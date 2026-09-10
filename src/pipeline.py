"""Pipeline huấn luyện, đánh giá và dự báo PM2.5 một giờ tiếp theo."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts.writer import save_artifacts
from src.calibration.conformal import (
    split_conformal_residuals,
    station_conformal_quantiles,
)
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
from src.forecasting.baselines import persistence_predictions, seasonal_naive_predictions
from src.forecasting.selection import select_forecast_strategy
from src.forecasting.trainer import make_pipeline
from src.serving.predictor import Predictor
from src.utils import sha256_file
from src.validation.backtest import evaluate_candidate
from src.validation.split import generate_split_manifest, split_by_time

DEFAULT_COVERAGE = 0.9


def run_train_pipeline(
    config_path: str = "configs/config.yaml",
    persist_artifacts: bool = True,
) -> dict[str, Any]:
    """Chạy load, regularization, feature engineering, backtest và calibration."""
    config = load_config(config_path)
    data_path = resolve_data_path(config["data"]["path"])
    raw = load_air_quality(config)

    timestamp = config["data"]["timestamp_column"]
    station = config["data"]["station_column"]
    target = config["data"]["target_column"]

    regularized = regularize_hourly_series(
        raw,
        timestamp_column=timestamp,
        group_columns=[station],
    )
    frame = (
        build_features(regularized, config, include_target=True)
        .dropna(subset=[target, "target_next_hour"])
        .sort_values(timestamp, kind="stable")
        .reset_index(drop=True)
    )

    split_config = config["split"]
    coverage_target = float(split_config.get("coverage", DEFAULT_COVERAGE))
    train_frame, calibration_frame, test_frame = split_by_time(
        frame,
        test_fraction=split_config.get("test_fraction"),
        calibration_fraction=split_config.get("calibration_fraction", 0.1),
        timestamp_column=timestamp,
        train_end=split_config.get("train_end"),
        calibration_end=split_config.get("calibration_end"),
        test_end=split_config.get("test_end"),
    )
    if calibration_frame.empty:
        raise ValueError(
            "Tập calibration không được rỗng; hãy tăng calibration_fraction "
            "hoặc cung cấp thêm dữ liệu."
        )
    if test_frame.empty:
        raise ValueError("Tập test không được rỗng; hãy cung cấp thêm dữ liệu.")

    split_manifest = generate_split_manifest(
        train_frame,
        calibration_frame,
        test_frame,
        timestamp_column=timestamp,
        station_column=station,
    )

    candidates = config["model_comparison"]["candidates"]
    _, feature_columns = make_pipeline(config, candidates[0])
    backtest = {
        name: evaluate_candidate(name, train_frame, config, feature_columns)
        for name in candidates
    }
    best_cv_model = min(backtest, key=lambda name: backtest[name]["mae_mean"])

    model_pipeline, feature_columns = make_pipeline(config, best_cv_model)
    model_pipeline.fit(train_frame[feature_columns], train_frame["target_next_hour"])

    model_cal_pred = model_pipeline.predict(calibration_frame[feature_columns])
    persistence_cal_pred = persistence_predictions(calibration_frame, target)
    model_cal_residuals = split_conformal_residuals(
        calibration_frame["target_next_hour"], model_cal_pred
    )
    persistence_cal_residuals = split_conformal_residuals(
        calibration_frame["target_next_hour"], persistence_cal_pred
    )
    minimum_calibration_samples = int(
        config.get("calibration", {}).get("minimum_calibration_samples_per_station", 20)
    )
    model_global_q, model_station_q = station_conformal_quantiles(
        calibration_frame,
        model_cal_residuals,
        station_column=station,
        coverage=coverage_target,
        minimum_samples=minimum_calibration_samples,
    )
    persistence_global_q, persistence_station_q = station_conformal_quantiles(
        calibration_frame,
        persistence_cal_residuals,
        station_column=station,
        coverage=coverage_target,
        minimum_samples=minimum_calibration_samples,
    )

    model_cal_metrics = regression_and_classification_metrics(
        calibration_frame["target_next_hour"], model_cal_pred, config["thresholds"]
    )
    persistence_cal_metrics = regression_and_classification_metrics(
        calibration_frame["target_next_hour"], persistence_cal_pred, config["thresholds"]
    )
    calibration_lower = np.maximum(0.0, model_cal_pred - model_global_q)
    calibration_upper = model_cal_pred + model_global_q
    calibration_picp = float(
        np.mean(
            (calibration_frame["target_next_hour"] >= calibration_lower)
            & (calibration_frame["target_next_hour"] <= calibration_upper)
        )
    )
    model_selection = select_forecast_strategy(
        best_cv_model,
        model_cal_metrics,
        persistence_cal_metrics,
        backtest[best_cv_model]["mae_std"],
        config,
        conformal_picp=calibration_picp,
        coverage_target=coverage_target,
    )
    forecast_strategy = model_selection["forecast_strategy"]

    model_test_pred = model_pipeline.predict(test_frame[feature_columns])
    persistence_test_pred = persistence_predictions(test_frame, target)
    seasonal_test_pred = seasonal_naive_predictions(test_frame, target)
    model_test = regression_and_classification_metrics(
        test_frame["target_next_hour"], model_test_pred, config["thresholds"]
    )
    persistence_test = regression_and_classification_metrics(
        test_frame["target_next_hour"], persistence_test_pred, config["thresholds"]
    )
    seasonal_test = regression_and_classification_metrics(
        test_frame["target_next_hour"], seasonal_test_pred, config["thresholds"]
    )

    model_test["mase"] = compute_mase(
        test_frame["target_next_hour"], model_test_pred, persistence_test_pred
    )
    model_test["skill_score_vs_persistence"] = compute_skill_score(
        test_frame["target_next_hour"], model_test_pred, persistence_test_pred
    )
    persistence_test["mase"] = 1.0
    persistence_test["skill_score_vs_persistence"] = 0.0
    seasonal_test["mase"] = compute_mase(
        test_frame["target_next_hour"], seasonal_test_pred, persistence_test_pred
    )
    seasonal_test["skill_score_vs_persistence"] = compute_skill_score(
        test_frame["target_next_hour"], seasonal_test_pred, persistence_test_pred
    )

    if forecast_strategy == "persistence":
        selected_prediction = persistence_test_pred
        selected_test = persistence_test.copy()
        selected_global_q = persistence_global_q
        selected_station_q = persistence_station_q
    else:
        selected_prediction = model_test_pred
        selected_test = model_test.copy()
        selected_global_q = model_global_q
        selected_station_q = model_station_q

    test_quantiles = (
        test_frame[station]
        .map(selected_station_q)
        .fillna(selected_global_q)
        .to_numpy()
    )
    test_lower = np.maximum(0.0, selected_prediction - test_quantiles)
    test_upper = selected_prediction + test_quantiles
    conformal_test = conformal_interval_metrics(
        test_frame["target_next_hour"], test_lower, test_upper
    )
    selected_test["conformal_interval"] = conformal_test
    selected_station_metrics = metrics_by_station(
        test_frame,
        selected_prediction,
        station,
        config["thresholds"],
        conformal_residual_q90={
            str(key): selected_station_q.get(str(key), selected_global_q)
            for key in test_frame[station].unique()
        },
    )
    sliced_errors = sliced_error_analysis(
        test_frame,
        selected_prediction,
        config["thresholds"],
        timestamp_column=timestamp,
    )

    dataset_scope = "sample" if "sample" in str(data_path).lower() else "external"
    trained_stations = sorted(train_frame[station].astype(str).unique().tolist())
    evaluation = {
        "data_audit": audit_air_quality(raw, config),
        "split_manifest": split_manifest,
        "baselines": {
            "persistence": persistence_test,
            "seasonal_naive_24h": seasonal_test,
        },
        "backtest": backtest,
        "calibration": {
            "model_metrics": model_cal_metrics,
            "persistence_metrics": persistence_cal_metrics,
            "model_residual_quantile": model_global_q,
            "persistence_residual_quantile": persistence_global_q,
            "calibration_rows": int(len(calibration_frame)),
            "calibration_window": [
                str(calibration_frame[timestamp].min()),
                str(calibration_frame[timestamp].max()),
            ],
            "conformal_quantile_method": "finite_sample_split_conformal",
            "coverage_target": coverage_target,
            "calibration_picp": calibration_picp,
        },
        "model_test": model_test,
        "persistence_test": persistence_test,
        "seasonal_naive_test": seasonal_test,
        "selected_strategy_test": selected_test,
        "conformal_test_evaluation": conformal_test,
        "model_selection": model_selection,
        "metrics_by_station": selected_station_metrics,
        "sliced_error_analysis": sliced_errors,
    }

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "best_cv_model": best_cv_model,
        "forecast_strategy": forecast_strategy,
        "dataset_scope": dataset_scope,
        "trained_stations": trained_stations,
        "prediction_interval": {
            "method": "split_conformal_prediction_interval",
            "residual_quantile": round(float(selected_global_q), 4),
            "global_q90": round(float(selected_global_q), 4),
            "station_q90": {key: round(value, 4) for key, value in selected_station_q.items()},
            "coverage_target": coverage_target,
            "finite_sample_corrected": True,
        },
        "features": feature_columns,
        "input_policy": {
            "required_history_hours": int(
                config.get("serving", {}).get("required_history_hours", 25)
            ),
            "allowed_gap_hours": int(config.get("serving", {}).get("allowed_gap_hours", 6)),
        },
        "data_provenance": {
            "source_file": str(data_path),
            "data_sha256": sha256_file(data_path),
            "rows_raw": int(len(raw)),
            "rows_train": int(len(train_frame)),
            "rows_calibration": int(len(calibration_frame)),
            "rows_test": int(len(test_frame)),
            "storage_timezone": "UTC",
            "calendar_timezone": config["data"].get(
                "calendar_timezone", "Asia/Ho_Chi_Minh"
            ),
        },
    }

    if persist_artifacts:
        save_artifacts(
            pipeline=model_pipeline,
            metadata=metadata,
            evaluation=evaluation,
            config=config,
        )

    return {
        "pipeline": model_pipeline,
        "metadata": metadata,
        "evaluation": evaluation,
        "metrics": selected_test,
        "best_cv_model": best_cv_model,
        "forecast_strategy": forecast_strategy,
        "prediction_interval": metadata["prediction_interval"],
    }


def main() -> None:
    """CLI canonical của dự án."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="HCMC PM2.5 next-hour forecasting")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Huấn luyện và đánh giá theo thời gian")
    train_parser.add_argument("--config", default="configs/config.yaml")
    train_parser.add_argument("--no-artifacts", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="Dự báo từ một CSV history")
    predict_parser.add_argument("--artifact-dir", default="artifacts")
    predict_parser.add_argument("--input", required=True)

    args = parser.parse_args()
    if args.command == "train":
        result = run_train_pipeline(args.config, persist_artifacts=not args.no_artifacts)
        print(
            "Pipeline hoàn tất: "
            f"best_cv_model={result['best_cv_model']}, "
            f"forecast_strategy={result['forecast_strategy']}"
        )
    else:
        predictor = Predictor.from_artifact(args.artifact_dir)
        result = predictor.predict(pd.read_csv(args.input))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
