"""Leakage-safe feature builder for PM2.5 next-hour forecasting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.exogenous import prepare_exogenous_columns
from src.features.lag import lookup_pm25_at_offset
from src.features.rolling import add_rolling_features, add_trend_features
from src.features.temporal import add_time_features


def add_missingness_features(
    frame: pd.DataFrame,
    target_column: str,
    exogenous_columns: list[str],
) -> pd.DataFrame:
    """Tạo các đặc trưng missingness giúp model nhận biết trạng thái gián đoạn của sensor."""
    result = frame.copy()
    lag_1_col = f"{target_column}_lag_1"
    if lag_1_col in result.columns:
        result[f"{target_column}_lag_1_missing"] = result[lag_1_col].isna().astype(float)

    # Missingness count in exogenous telemetry
    existing_exo = [c for c in exogenous_columns if c in result.columns]
    if existing_exo:
        result["exogenous_missing_count"] = result[existing_exo].isna().sum(axis=1).astype(float)
    else:
        result["exogenous_missing_count"] = 0.0

    return result


def build_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
    include_target: bool = True,
) -> pd.DataFrame:
    """Tạo đặc trưng chỉ từ quan sát hiện tại/quá khứ trong từng trạm mà không rò rỉ tương lai."""
    data_config = config["data"]
    station = data_config["station_column"]
    timestamp = data_config["timestamp_column"]
    target = data_config["target_column"]
    result = frame.sort_values([station, timestamp], kind="stable").copy()

    # 1. Clock-time lags: tra cứu theo số giờ thực tế, không dịch chuyển vị trí dòng
    for lag in config["features"]["lags"]:
        result[f"{target}_lag_{lag}"] = lookup_pm25_at_offset(
            result,
            station,
            timestamp,
            target,
            offset_hours=-lag,
        )

    # 2. Trend differences: chênh lệch nồng độ so với 1h và 3h trước
    delta_lags = config["features"].get("delta_lags", [1, 3])
    result = add_trend_features(result, target, delta_lags=delta_lags)

    # 3. Rolling statistics (mean & std) với closed='left' (chỉ dùng lịch sử trước t)
    include_rolling_std = config["features"].get("include_rolling_std", True)
    result = add_rolling_features(
        result,
        station,
        timestamp,
        target,
        config["features"]["rolling_windows"],
        include_std=include_rolling_std,
    )

    # 4. Cyclic time features
    result = add_time_features(result, timestamp)

    # 5. Exogenous features availability
    result = prepare_exogenous_columns(result, config)

    # 6. Optional explicit missingness indicators
    if config.get("features", {}).get("include_missingness_features", False):
        exo_cols = config.get("features", {}).get("exogenous_columns", [])
        result = add_missingness_features(result, target, exo_cols)

    # 7. Target & Baselines
    if include_target:
        result["target_timestamp"] = result[timestamp] + pd.to_timedelta(1, unit="h")
        result["target_next_hour"] = lookup_pm25_at_offset(
            result,
            station,
            timestamp,
            target,
            offset_hours=1,
        )
        # Seasonal Naive 24h cho target(t+1):
        # \hat{y}_{t+1}^{seasonal24} = y_{(t+1)-24} = y_{t-23}
        # Tra cứu quan trắc tại cùng trạm ở mốc t - 23h.
        result["seasonal_naive_24h"] = lookup_pm25_at_offset(
            result,
            station,
            timestamp,
            target,
            offset_hours=-23,
        )
    return result


def model_feature_columns(config: dict[str, Any]) -> list[str]:
    """Trả về danh sách cột đầu vào theo đúng thứ tự của model."""
    target = config["data"]["target_column"]
    history = [f"{target}_lag_{lag}" for lag in config["features"]["lags"]]
    rolling_means = [
        f"{target}_rolling_mean_{window}" for window in config["features"]["rolling_windows"]
    ]
    include_rolling_std = config["features"].get("include_rolling_std", True)
    rolling_stds = (
        [f"{target}_rolling_std_{window}" for window in config["features"]["rolling_windows"]]
        if include_rolling_std
        else []
    )
    delta_lags = config["features"].get("delta_lags", [1, 3])
    deltas = [
        f"{target}_delta_{lag}h"
        for lag in delta_lags
        if lag in config["features"].get("lags", [])
    ]
    base_cols = [
        target,
        *history,
        *deltas,
        *rolling_means,
        *rolling_stds,
        *config["features"]["exogenous_columns"],
        "hour_sin",
        "hour_cos",
        "day_of_week",
    ]
    if config.get("features", {}).get("include_missingness_features", False):
        base_cols.extend([f"{target}_lag_1_missing", "exogenous_missing_count"])

    return base_cols
