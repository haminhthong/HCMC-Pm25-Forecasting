"""Serving Predictor for Next-Hour PM2.5 forecasting."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.artifacts.loader import load_artifact_bundle
from src.artifacts.schema import ForecastContext
from src.data.regularization import regularize_hourly_series
from src.data.runtime_gate import audit_runtime_history
from src.data.schema import normalize_timestamp_series
from src.evaluate import classify_pm25, get_threshold_params
from src.features.builder import build_features, model_feature_columns
from src.monitoring.forecast_log import append_forecast_event


class Predictor:
    """Đóng gói preprocessing, model và metadata cho inference tự chứa (P0.4)."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        artifact_dir: Path | None = None,
        is_from_artifact: bool = False,
    ):
        self.config = config
        if artifact_dir is None:
            artifact_dir = Path(config["artifacts"]["directory"])
        self.artifact_dir = Path(artifact_dir)

        model_path = self.artifact_dir / config["artifacts"]["model_file"]
        self.model = joblib.load(model_path)

        metadata_path = self.artifact_dir / config["artifacts"].get("metadata_file", "metadata.json")
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self._feature_schema = self._load_feature_schema(self.artifact_dir, config)
        self.is_from_artifact = is_from_artifact

        # Only cross-check consistency when loaded from an artifact bundle with config snapshot
        if self.is_from_artifact:
            self._assert_self_consistent()

        # ForecastContext & Station coverage
        raw_ctx = self.metadata.get("forecast_context")
        self.forecast_context = (
            ForecastContext.from_dict(raw_ctx) if raw_ctx else ForecastContext()
        )
        self.trained_stations = set(self.metadata.get("trained_stations", []))

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        production_pointer: str | Path | None = None,
    ) -> Predictor:
        """Nạp predictor tự chứa hoàn toàn từ artifact versioned."""
        bundle = load_artifact_bundle(artifact_dir)
        config = bundle["config"]
        if not config:
            snapshot_path = bundle["artifact_dir"] / "config_snapshot.yaml"
            if snapshot_path.is_file():
                config = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(
                    f"Artifact thiếu config_snapshot.yaml tại {snapshot_path}; "
                    "P0.4 yêu cầu artifact tự chứa config."
                )

        predictor = cls(
            config,
            artifact_dir=bundle["artifact_dir"],
            is_from_artifact=True,
        )
        predictor.metadata = bundle["metadata"]
        predictor.forecast_context = bundle["forecast_context"]
        predictor.trained_stations = set(predictor.metadata.get("trained_stations", []))
        return predictor

    @staticmethod
    def _load_feature_schema(artifact_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
        schema_path = artifact_dir / config["artifacts"].get(
            "feature_schema_file", "feature_schema.json"
        )
        if not schema_path.is_file():
            return {}
        with schema_path.open(encoding="utf-8") as file:
            return json.load(file)

    def _assert_self_consistent(self) -> None:
        """Cross-check that the artifact snapshot matches what we trained on."""
        schema_features = self._feature_schema.get("model_feature_columns")
        if not schema_features:
            return
        # Schema artifact lưu cả numeric features và categorical station key,
        # đúng với thứ tự columns mà make_pipeline trả về.
        live_features = [
            *model_feature_columns(self.config),
            self.config["data"]["station_column"],
        ]
        if list(schema_features) != list(live_features):
            raise RuntimeError(
                "feature_schema.json và configs/config.yaml không đồng bộ. "
                "Predictor phải được nạp từ artifact (Predictor.from_artifact) "
                "để tránh training-serving skew (P0.4)."
            )

    def predict(self, observations: pd.DataFrame) -> dict[str, Any]:
        """Dự báo PM2.5 cho mốc tiếp theo từ chuỗi quan sát lịch sử."""
        data_config = self.config["data"]
        station_col = data_config["station_column"]
        timestamp_col = data_config["timestamp_column"]
        target_col = data_config["target_column"]

        if observations.empty:
            raise ValueError("Bảng quan sát không được rỗng.")

        stations = observations[station_col].dropna().unique()
        if len(stations) != 1:
            raise ValueError("Dữ liệu đầu vào phải thuộc về một trạm duy nhất.")

        station_name = str(stations[0])
        is_ood_station = bool(self.trained_stations and station_name not in self.trained_stations)

        # 1. P0.3: Regularize hourly series instead of rejecting irregular gaps
        work = observations.copy()
        work[timestamp_col] = normalize_timestamp_series(
            work[timestamp_col],
            source_timezone=self.config.get("data", {}).get(
                "source_timezone", self.forecast_context.timezone
            ),
        )
        if work[timestamp_col].duplicated().any():
            raise ValueError("Chuỗi quan sát chứa timestamp trùng lặp.")

        work = regularize_hourly_series(
            work,
            timestamp_column=timestamp_col,
            group_columns=[station_col],
        )

        min_required = self.forecast_context.required_history_hours
        if len(work) < min_required:
            raise ValueError(
                f"Cần tối thiểu {min_required} quan trắc (giờ) để tính toán đặc trưng lag; "
                f"nhận {len(work)}."
            )

        runtime_quality = audit_runtime_history(
            work,
            timestamp_column=timestamp_col,
            target_column=target_col,
            required_history_hours=min_required,
            allowed_gap_hours=self.forecast_context.allowed_gap_hours,
            exogenous_columns=self.config.get("features", {}).get("exogenous_columns", []),
        )
        if runtime_quality["status"] == "UNUSABLE":
            raise ValueError(
                "Dữ liệu hiện tại không đủ để dự báo an toàn: "
                + ", ".join(runtime_quality["warnings"] or ["runtime_data_unusable"])
            )

        # 2. Build features on regularized series
        featured = build_features(work, self.config, include_target=False)
        latest_row = featured.iloc[[-1]]

        current_val = latest_row[target_col].values[0]
        if pd.isna(current_val) or current_val < 0:
            raise ValueError("Giá trị PM2.5 không được âm hoặc thiếu tại mốc quan sát hiện tại.")

        # 3. Serving champion strategy
        serving_strategy = self.metadata.get("serving_strategy", "ml_model")
        serving_champion = self.metadata.get("serving_champion", self.metadata.get("model_name", "model"))
        if runtime_quality["fallback_required"]:
            serving_strategy = "persistence_fallback"
            serving_champion = "persistence"

        if serving_strategy == "persistence_fallback" or serving_champion == "persistence":
            predicted_pm25 = float(current_val)
        else:
            feature_cols = model_feature_columns(self.config)
            input_df = latest_row[[*feature_cols, station_col]]
            predicted_pm25 = float(self.model.predict(input_df)[0])

        predicted_pm25 = max(0.0, predicted_pm25)

        # 4. Conformal Prediction Interval
        interval_info = self.metadata.get("prediction_interval", {})
        station_q90 = interval_info.get("station_q90", {})
        residual_q = float(
            station_q90.get(station_name, interval_info.get("global_q90", interval_info.get("residual_quantile", 5.0)))
        )
        coverage_val = float(
            interval_info.get("coverage", interval_info.get("coverage_target", 0.90))
        )

        lower = max(0.0, predicted_pm25 - residual_q)
        upper = predicted_pm25 + residual_q

        low_max, medium_max, labels = get_threshold_params(self.config["thresholds"])
        level = classify_pm25([predicted_pm25], low_max, medium_max, labels)[0]

        # Dùng iloc thay vì values để giữ timezone UTC của Timestamp canonical.
        forecast_origin_ts = pd.to_datetime(latest_row[timestamp_col].iloc[0])
        forecast_for_ts = forecast_origin_ts + pd.to_timedelta(self.forecast_context.horizon_hours, unit="h")

        result = {
            "station": station_name,
            "station_id": station_name,
            "forecast_origin": str(forecast_origin_ts),
            "forecast_for": str(forecast_for_ts),
            "current_pm25": float(current_val),
            "predicted_pm25": round(predicted_pm25, 2),
            "level": str(level),
            "forecast_strategy": serving_strategy,
            "serving_champion": serving_champion,
            "is_out_of_distribution": is_ood_station,
            "interval": {
                "method": interval_info.get("method", "split_conformal_prediction_interval"),
                "coverage_target": coverage_val,
                "coverage": coverage_val,
                "lower": round(lower, 2),
                "upper": round(upper, 2),
                "width": round(upper - lower, 2),
            },
            "model_version": self.metadata.get("model_version", "unknown"),
            "updated_at": datetime.now(UTC).isoformat(),
            "production_readiness": self.metadata.get("production_readiness", "unknown"),
            "calibration_gate": self.metadata.get("calibration_gate", "unknown"),
            "data_quality": runtime_quality,
        }
        log_path = self.config.get("monitoring", {}).get("forecast_log_path")
        if log_path:
            append_forecast_event(
                {
                    "forecast_id": (
                        f"{self.metadata.get('model_version', 'unknown')}:"
                        f"{station_name}:{forecast_origin_ts.isoformat()}"
                    ),
                    "model_version": result["model_version"],
                    "station_id": station_name,
                    "forecast_origin": forecast_origin_ts.isoformat(),
                    "forecast_for": forecast_for_ts.isoformat(),
                    "current_pm25": float(current_val),
                    "prediction": result["predicted_pm25"],
                    "lower": result["interval"]["lower"],
                    "upper": result["interval"]["upper"],
                    "strategy": serving_strategy,
                    "data_quality_status": runtime_quality["status"],
                    "created_at": result["updated_at"],
                    "persistence_prediction": float(current_val),
                },
                log_path,
            )
        return result
