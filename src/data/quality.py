"""Kiểm tra chất lượng, schema và miền giá trị vật lý của dữ liệu."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.regularization import audit_hourly_gaps
from src.data.schema import DEFAULT_SOURCE_TIMEZONE, PHYSICAL_RANGES, normalize_timestamp_series


def validate_schema(frame: pd.DataFrame, required_columns: list[str]) -> None:
    """Kiểm tra sự hiện diện của các cột bắt buộc trong bảng dữ liệu."""
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Dữ liệu thiếu cột bắt buộc: {', '.join(missing)}")


def validate_physical_ranges(frame: pd.DataFrame) -> dict[str, int]:
    """Kiểm tra số lượng bản ghi vi phạm miền giá trị vật lý theo từng biến."""
    anomalies: dict[str, int] = {}
    for col, (min_val, max_val) in PHYSICAL_RANGES.items():
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            invalid_count = int(((series < min_val) | (series > max_val)).sum())
            if invalid_count > 0:
                anomalies[col] = invalid_count
    return anomalies


def audit_air_quality(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Tạo báo cáo kiểm tra chất lượng dữ liệu đầu vào."""
    data_config = config["data"]
    timestamp = data_config["timestamp_column"]
    station = data_config["station_column"]
    target = data_config["target_column"]
    target_values = pd.to_numeric(frame[target], errors="coerce")

    duplicate_mask = frame.duplicated([station, timestamp], keep=False)
    gap_audit = audit_hourly_gaps(frame, timestamp, group_columns=[station])
    physical_anomalies = validate_physical_ranges(frame)
    tracked_columns = [target, *config.get("features", {}).get("exogenous_columns", [])]
    missing_rate = {
        column: float(frame[column].isna().mean())
        for column in tracked_columns
        if column in frame.columns
    }
    availability_violations = 0
    if "available_at" in frame.columns:
        observed_at = normalize_timestamp_series(
            frame[timestamp],
            source_timezone=data_config.get("source_timezone", DEFAULT_SOURCE_TIMEZONE),
        )
        available_at = normalize_timestamp_series(
            frame["available_at"],
            source_timezone=data_config.get("source_timezone", DEFAULT_SOURCE_TIMEZONE),
        )
        availability_violations = int((available_at > observed_at).sum())

    return {
        "rows": int(len(frame)),
        "stations": int(frame[station].nunique()),
        "period": [str(frame[timestamp].min()), str(frame[timestamp].max())],
        "duplicate_station_timestamps": int(duplicate_mask.sum()),
        "missing_target": int(target_values.isna().sum()),
        "negative_target": int((target_values < 0).sum()),
        "irregular_hourly_gaps": gap_audit["irregular_hourly_gaps"],
        "total_gaps_observed": gap_audit["total_gaps_observed"],
        "physical_anomalies": physical_anomalies,
        "missing_rate": missing_rate,
        "availability_violations": availability_violations,
        "columns": list(frame.columns),
    }
