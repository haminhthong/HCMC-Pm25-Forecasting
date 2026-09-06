"""Master Pipeline CLI: Data Ingestion -> Regularization -> Features -> Backtest -> Calibration -> Artifact -> Evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts.schema import ForecastContext
from src.artifacts.writer import save_artifacts
from src.calibration.conformal import conformal_quantile, split_conformal_residuals
from src.config import load_config
from src.data.loader import load_air_quality, resolve_data_path
from src.data.quality import audit_air_quality
from src.data.regularization import regularize_hourly_series
from src.evaluation.metrics import (
    compute_mase,
    compute_skill_score,
    conformal_interval_metrics,
    regression_and_classification_metrics,
)
from src.evaluation.slices import sliced_error_analysis
from src.evaluation.station_metrics import metrics_by_station
from src.features.builder import build_features
from src.forecasting.baselines import (
    persistence_predictions,
    seasonal_naive_predictions,
)
from src.forecasting.selector import build_quality_gate, resolve_model_statuses
from src.forecasting.trainer import make_pipeline
from src.serving.predictor import Predictor
from src.utils import sha256_file
from src.validation.backtest import evaluate_candidate
from src.validation.split import generate_split_manifest, split_by_time

DEFAULT_COVERAGE = 0.9


def generate_model_version(data_path: Path) -> str:
    """Sinh mã phiên bản tự động từ ngày, git commit SHA (nếu có) và data SHA-256."""
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    git_sha = "local"
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            git_sha = result.stdout.strip()
    except Exception:
        pass
    data_hash = sha256_file(data_path)[:7]
    return f"pm25-{date_str}-{git_sha}-{data_hash}"


def run_train_pipeline(
    config_path: str = "configs/config.yaml",
    persist_artifacts: bool = True,
) -> dict[str, Any]:
    """Chạy toàn bộ pipeline huấn luyện, calibration, backtest và lưu artifact."""
    config = load_config(config_path)
    data_path = resolve_data_path(config["data"]["path"])
    raw = load_air_quality(config)

    timestamp = config["data"]["timestamp_column"]
    station = config["data"]["station_column"]
    target = config["data"]["target_column"]

    # 1. Regularize hourly series (P0.3: shared train/serving gap policy)
    regularized = regularize_hourly_series(
        raw,
        timestamp_column=timestamp,
        group_columns=[station],
    )

    # 2. Build leakage-safe features
    frame = (
        build_features(regularized, config, include_target=True)
        .dropna(subset=[target, "target_next_hour"])
        .sort_values(timestamp, kind="stable")
        .reset_index(drop=True)
    )

    # 3. Time-ordered train / calibration / test split
    test_fraction = config["split"]["test_fraction"]
    calibration_fraction = config["split"].get("calibration_fraction", 0.1)
    coverage_target = float(config["split"].get("coverage", DEFAULT_COVERAGE))

    train_frame, cal_frame, test_frame = split_by_time(
        frame,
        test_fraction=test_fraction,
        calibration_fraction=calibration_fraction,
        timestamp_column=timestamp,
    )

    # P0.1 — zero calibration leakage
    if cal_frame.empty:
        raise ValueError(
            "calibration set rỗng sau split_by_time; P0.1 không cho phép "
            "fallback sang train_frame hoặc test_frame. Hãy tăng "
            "split.calibration_fraction hoặc cung cấp thêm dữ liệu."
        )
    if test_frame.empty:
        raise ValueError(
            "test set rỗng sau split_by_time; không thể đánh giá. "
            "Hãy tăng split.test_fraction hoặc cung cấp thêm dữ liệu."
        )

    # Generate split manifest
    split_manifest = generate_split_manifest(
        train_frame,
        cal_frame,
        test_frame,
        timestamp_column=timestamp,
        station_column=station,
    )

    # 4. Expanding-window backtest on candidates
    _, columns = make_pipeline(config, config["model"]["name"])
    candidates = config.get("model_comparison", {}).get("candidates", [config["model"]["name"]])
    backtest = {name: evaluate_candidate(name, train_frame, config, columns) for name in candidates}
    candidate_champion = min(backtest, key=lambda name: backtest[name]["mae_mean"])

    # 5. Fit champion on full development train frame
    pipeline, columns = make_pipeline(config, candidate_champion)
    pipeline.fit(train_frame[columns], train_frame["target_next_hour"])

    # 6. Conformal Calibration on dedicated future calibration window (P0.1 + P0.2)
    ml_cal_residuals = split_conformal_residuals(
        cal_frame["target_next_hour"],
        pipeline.predict(cal_frame[columns]),
    )
    pers_cal_residuals = split_conformal_residuals(
        cal_frame["target_next_hour"],
        persistence_predictions(cal_frame, target),
    )
    ml_residual_q = conformal_quantile(ml_cal_residuals, coverage_target)
    pers_residual_q = conformal_quantile(pers_cal_residuals, coverage_target)

    # Calibration evaluation
    cal_ml_metrics = regression_and_classification_metrics(
        cal_frame["target_next_hour"],
        pipeline.predict(cal_frame[columns]),
        config["thresholds"],
    )
    cal_pers_metrics = regression_and_classification_metrics(
        cal_frame["target_next_hour"],
        persistence_predictions(cal_frame, target),
        config["thresholds"],
    )

    # Conformal calibration coverage check
    cal_lower = np.maximum(0.0, pipeline.predict(cal_frame[columns]) - ml_residual_q)
    cal_upper = pipeline.predict(cal_frame[columns]) + ml_residual_q
    cal_picp = float(np.mean((cal_frame["target_next_hour"] >= cal_lower) & (cal_frame["target_next_hour"] <= cal_upper)))

    quality_gate = build_quality_gate(
        cal_ml_metrics,
        cal_pers_metrics,
        backtest[candidate_champion]["mae_std"],
        config,
        conformal_picp=cal_picp,
        coverage_target=coverage_target,
    )

    # 7. Independent Final Test Evaluation
    candidate_ml_pred = pipeline.predict(test_frame[columns])
    pers_test_pred = persistence_predictions(test_frame, target)
    seasonal_test_pred = seasonal_naive_predictions(test_frame, target)

    candidate_ml_test = regression_and_classification_metrics(
        test_frame["target_next_hour"],
        candidate_ml_pred,
        config["thresholds"],
    )
    persistence_test = regression_and_classification_metrics(
        test_frame["target_next_hour"],
        pers_test_pred,
        config["thresholds"],
    )
    seasonal_naive_test = regression_and_classification_metrics(
        test_frame["target_next_hour"],
        seasonal_test_pred,
        config["thresholds"],
    )

    candidate_ml_test["mase"] = compute_mase(
        test_frame["target_next_hour"], candidate_ml_pred, pers_test_pred
    )
    candidate_ml_test["skill_score_vs_persistence"] = compute_skill_score(
        test_frame["target_next_hour"], candidate_ml_pred, pers_test_pred
    )
    persistence_test["mase"] = 1.0
    persistence_test["skill_score_vs_persistence"] = 0.0
    seasonal_naive_test["mase"] = compute_mase(
        test_frame["target_next_hour"], seasonal_test_pred, pers_test_pred
    )
    seasonal_naive_test["skill_score_vs_persistence"] = compute_skill_score(
        test_frame["target_next_hour"], seasonal_test_pred, pers_test_pred
    )

    # Model status and serving champion resolution
    is_smoke = bool("sample" in str(data_path).lower() or "synthetic" in str(data_path).lower())
    statuses = resolve_model_statuses(
        candidate_champion=candidate_champion,
        passes_quality_gate=quality_gate["passes_baseline"],
        smoke_only=is_smoke,
    )

    serving_champion = candidate_champion if quality_gate["passes_baseline"] else "persistence"
    serving_residual_q = ml_residual_q if quality_gate["passes_baseline"] else pers_residual_q
    serving_champion_pred = candidate_ml_pred if quality_gate["passes_baseline"] else pers_test_pred
    serving_champion_test = candidate_ml_test.copy() if quality_gate["passes_baseline"] else persistence_test.copy()

    # Conformal test interval evaluation
    test_lower = np.maximum(0.0, serving_champion_pred - serving_residual_q)
    test_upper = serving_champion_pred + serving_residual_q
    conformal_test_metrics = conformal_interval_metrics(
        test_frame["target_next_hour"], test_lower, test_upper
    )
    serving_champion_test["conformal_interval"] = conformal_test_metrics

    station_metrics = metrics_by_station(
        test_frame,
        serving_champion_pred,
        station,
        config["thresholds"],
        conformal_residual_q90=serving_residual_q,
    )
    sliced_errors = sliced_error_analysis(
        test_frame,
        serving_champion_pred,
        config["thresholds"],
        timestamp_column=timestamp,
        station_column=station,
    )

    model_version = generate_model_version(data_path)
    trained_stations = sorted(train_frame[station].unique().tolist())

    forecast_context = ForecastContext(
        horizon_hours=1,
        frequency="1h",
        timezone="Asia/Ho_Chi_Minh",
        required_history_hours=25,
        allowed_gap_hours=6,
        feature_schema_version="2.0",
    )

    evaluation = {
        "data_audit": audit_air_quality(raw, config),
        "split_manifest": split_manifest,
        "baselines": {
            "persistence": persistence_test,
            "seasonal_naive_24h": seasonal_naive_test,
        },
        "backtest": backtest,
        "calibration": {
            "ml_metrics": cal_ml_metrics,
            "persistence_metrics": cal_pers_metrics,
            "ml_residual_quantile": ml_residual_q,
            "persistence_residual_quantile": pers_residual_q,
            "ml_residual_quantile_90": ml_residual_q,
            "persistence_residual_quantile_90": pers_residual_q,
            "calibration_rows": int(len(cal_frame)),
            "calibration_window": [
                str(cal_frame[timestamp].min()),
                str(cal_frame[timestamp].max()),
            ],
            "conformal_quantile_method": "finite_sample_split_conformal",
            "coverage_target": coverage_target,
            "calibration_picp": cal_picp,
        },
        "candidate_ml_test": candidate_ml_test,
        "persistence_test": persistence_test,
        "seasonal_naive_test": seasonal_naive_test,
        "serving_champion": serving_champion,
        "serving_champion_test": serving_champion_test,
        "champion_test": serving_champion_test,
        "conformal_test_evaluation": conformal_test_metrics,
        "quality_gate": quality_gate,
        "metrics_by_station": station_metrics,
        "sliced_error_analysis": sliced_errors,
    }

    metadata = {
        "model_version": model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": candidate_champion,
        "serving_champion": serving_champion,
        "serving_strategy": statuses["serving_strategy"],
        "candidate_champion": statuses["candidate_champion"],
        "calibration_gate": statuses["calibration_gate"],
        "production_readiness": statuses["production_readiness"],
        "trained_stations": trained_stations,
        "prediction_interval": {
            "method": "split_conformal",
            "residual_quantile": round(serving_residual_q, 4),
            "coverage_target": coverage_target,
            "finite_sample_corrected": True,
        },
        "features": columns,
        "data_provenance": {
            "data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "rows_raw": int(len(raw)),
            "rows_train": int(len(train_frame)),
            "rows_calibration": int(len(cal_frame)),
            "rows_test": int(len(test_frame)),
            "smoke_only": is_smoke,
        },
    }

    if persist_artifacts:
        save_artifacts(
            pipeline=pipeline,
            metadata=metadata,
            evaluation=evaluation,
            config=config,
            split_manifest=split_manifest,
            forecast_context=forecast_context,
        )

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "evaluation": evaluation,
        "metrics": serving_champion_test,
        "candidate_champion": candidate_champion,
        "serving_champion": serving_champion,
        "prediction_interval": metadata["prediction_interval"],
    }


def main() -> None:
    """CLI entry point for master pipeline."""
    parser = argparse.ArgumentParser(description="Leakage-Safe Next-Hour PM2.5 Master Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train command
    train_parser = subparsers.add_parser("train", help="Run full training, backtest, and artifact freezing")
    train_parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    train_parser.add_argument("--no-artifacts", action="store_true", help="Do not save artifacts")

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Predict PM2.5 from input CSV")
    predict_parser.add_argument("--artifact-dir", default="artifacts", help="Path to artifact bundle directory")
    predict_parser.add_argument("--input", required=True, help="Path to CSV containing observations")

    args = parser.parse_args()

    if args.command == "train":
        result = run_train_pipeline(
            config_path=args.config,
            persist_artifacts=not args.no_artifacts,
        )
        print(f"Pipeline finished successfully. Serving champion: {result['serving_champion']}")
        print(f"Production readiness: {result['metadata']['production_readiness']}")
        print(f"Calibration gate: {result['metadata']['calibration_gate']}")

    elif args.command == "predict":
        predictor = Predictor.from_artifact(args.artifact_dir)
        df = pd.read_csv(args.input)
        res = predictor.predict(df)
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
