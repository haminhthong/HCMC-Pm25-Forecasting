"""Đóng gói preprocessing, model và interval cho inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.artifacts.loader import load_artifact_bundle
from src.config import load_config
from src.data.regularization import regularize_hourly_series
from src.data.schema import normalize_timestamp_series
from src.evaluation.metrics import classify_pm25, get_threshold_params
from src.features.builder import build_features, model_feature_columns
from src.serving.input_validation import check_forecast_input


class Predictor:
    """Dự báo một giờ tiếp theo từ history của đúng một trạm."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        artifact_dir: Path | None = None,
        model: Any | None = None,
        metadata: dict[str, Any] | None = None,
        feature_schema: dict[str, Any] | None = None,
    ):
        self.config = config
        self.artifact_dir = Path(artifact_dir or config["artifacts"]["directory"])
        model_path = self.artifact_dir / config["artifacts"].get("model_file", "model.joblib")
        self.model = model if model is not None else joblib.load(model_path)
        self.metadata = metadata if metadata is not None else self._read_json("metadata.json")
        self._feature_schema = (
            feature_schema if feature_schema is not None else self._read_json("feature_schema.json")
        )
        self.trained_stations = set(self.metadata.get("trained_stations", []))
        if feature_schema is not None:
            self._assert_feature_schema()

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path = "artifacts",
        *,
        config_path: str | Path = "configs/config.yaml",
    ) -> Predictor:
        """Nạp model hiện tại và cấu hình nguồn của repository."""
        bundle = load_artifact_bundle(artifact_dir)
        config_file = Path(config_path)
        if not config_file.is_file():
            repo_config = Path(__file__).resolve().parents[2] / config_file
            config_file = repo_config if repo_config.is_file() else config_file
        config = load_config(config_file)
        return cls(
            config,
            artifact_dir=bundle["artifact_dir"],
            model=bundle["pipeline"],
            metadata=bundle["metadata"],
            feature_schema=bundle["feature_schema"],
        )

    def _read_json(self, filename: str) -> dict[str, Any]:
        path = self.artifact_dir / filename
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _assert_feature_schema(self) -> None:
        """Báo lỗi sớm nếu thứ tự cột train và serving không khớp."""
        saved_columns = self._feature_schema.get("model_feature_columns")
        if not saved_columns:
            return
        live_columns = [
            *model_feature_columns(self.config),
            self.config["data"]["station_column"],
        ]
        if list(saved_columns) != live_columns:
            raise RuntimeError(
                "feature_schema.json không khớp với cấu hình hiện tại; "
                "hãy train lại trước khi dự báo."
            )

    def predict(self, observations: pd.DataFrame) -> dict[str, Any]:
        """Tạo điểm dự báo và conformal interval từ các quan trắc gần nhất."""
        data_config = self.config["data"]
        station_col = data_config["station_column"]
        timestamp_col = data_config["timestamp_column"]
        target_col = data_config["target_column"]
        if observations.empty:
            raise ValueError("Bảng quan sát không được rỗng.")
        if observations[station_col].isna().any():
            raise ValueError(f"Cột {station_col} không được chứa giá trị rỗng.")

        stations = observations[station_col].dropna().unique()
        if len(stations) != 1:
            raise ValueError("Dữ liệu đầu vào phải thuộc về một trạm duy nhất.")
        station_name = str(stations[0]).strip()
        is_ood_station = bool(self.trained_stations and station_name not in self.trained_stations)

        work = observations.copy()
        work[station_col] = work[station_col].astype(str).str.strip()
        if work[station_col].eq("").any():
            raise ValueError(f"Cột {station_col} không được chứa station_id rỗng.")
        work[timestamp_col] = normalize_timestamp_series(
            work[timestamp_col],
            source_timezone=data_config.get("source_timezone", "Asia/Ho_Chi_Minh"),
        )
        if work[timestamp_col].isna().any():
            raise ValueError("Cột timestamp chứa giá trị không hợp lệ.")
        if "available_at" in work.columns:
            original_available = work["available_at"].notna()
            work["available_at"] = normalize_timestamp_series(
                work["available_at"],
                source_timezone=data_config.get("source_timezone", "Asia/Ho_Chi_Minh"),
            )
            if work.loc[original_available, "available_at"].isna().any():
                raise ValueError("Cột available_at chứa giá trị không hợp lệ.")
        for column in [target_col, *data_config.get("optional_columns", [])]:
            if column in work.columns:
                work[column] = pd.to_numeric(work[column], errors="coerce")
        if work[timestamp_col].duplicated().any():
            raise ValueError("Chuỗi quan sát chứa timestamp trùng lặp.")

        work = regularize_hourly_series(
            work,
            timestamp_column=timestamp_col,
            group_columns=[station_col],
        )

        serving_config = self.config.get("serving", {})
        required_history = int(serving_config.get("required_history_hours", 25))
        allowed_gap = int(serving_config.get("allowed_gap_hours", 6))
        if len(work) < required_history:
            raise ValueError(
                f"Cần tối thiểu {required_history} quan trắc (giờ); nhận {len(work)}."
            )
        quality = check_forecast_input(
            work,
            timestamp_column=timestamp_col,
            target_column=target_col,
            required_history_hours=required_history,
            allowed_gap_hours=allowed_gap,
            exogenous_columns=self.config.get("features", {}).get("exogenous_columns", []),
        )
        if quality["status"] == "invalid":
            raise ValueError(
                "Dữ liệu hiện tại không đủ để dự báo an toàn: "
                + ", ".join(quality["warnings"] or ["forecast_input_invalid"])
            )

        featured = build_features(work, self.config, include_target=False)
        latest_row = featured.iloc[[-1]]
        current_value = latest_row[target_col].iloc[0]
        if pd.isna(current_value) or current_value < 0:
            raise ValueError("Giá trị PM2.5 hiện tại không được âm hoặc thiếu.")

        configured_strategy = self.metadata.get(
            "forecast_strategy",
            self.metadata.get("best_cv_model", "persistence"),
        )
        strategy = "persistence" if quality["use_persistence"] else str(configured_strategy)
        if strategy == "persistence":
            prediction = float(current_value)
        else:
            columns = model_feature_columns(self.config)
            prediction = float(self.model.predict(latest_row[[*columns, station_col]])[0])
        prediction = max(0.0, prediction)

        interval_info = self.metadata.get("prediction_interval", {})
        station_quantiles = interval_info.get("station_q90", {})
        residual_quantile = float(
            station_quantiles.get(
                station_name,
                interval_info.get("global_q90", interval_info.get("residual_quantile", 5.0)),
            )
        )
        coverage = float(interval_info.get("coverage_target", interval_info.get("coverage", 0.9)))
        lower = max(0.0, prediction - residual_quantile)
        upper = prediction + residual_quantile

        low_max, medium_max, labels = get_threshold_params(self.config["thresholds"])
        level = classify_pm25([prediction], low_max, medium_max, labels)[0]
        origin = pd.Timestamp(latest_row[timestamp_col].iloc[0])
        forecast_for = origin + pd.to_timedelta(1, unit="h")

        return {
            "station": station_name,
            "station_id": station_name,
            "forecast_origin": str(origin),
            "forecast_for": str(forecast_for),
            "current_pm25": float(current_value),
            "predicted_pm25": round(prediction, 2),
            "level": str(level),
            "forecast_strategy": strategy,
            "best_cv_model": self.metadata.get("best_cv_model"),
            "dataset_scope": self.metadata.get("dataset_scope", "sample"),
            "is_out_of_distribution": is_ood_station,
            "interval": {
                "method": interval_info.get("method", "split_conformal_prediction_interval"),
                "coverage_target": coverage,
                "coverage": coverage,
                "lower": round(lower, 2),
                "upper": round(upper, 2),
                "width": round(upper - lower, 2),
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "data_quality": quality,
        }
