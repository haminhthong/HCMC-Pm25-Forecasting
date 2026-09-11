"""Tạo đặc trưng PM2.5 giờ kế tiếp mà không rò rỉ dữ liệu tương lai."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.schema import DEFAULT_SOURCE_TIMEZONE
from src.features.exogenous import lookup_feature_at_offset, prepare_exogenous_columns
from src.features.lag import lookup_pm25_at_offset
from src.features.rolling import add_rolling_features, add_trend_features
from src.features.temporal import add_time_features


def add_missingness_features(
    frame: pd.DataFrame,
    target_column: str,
    exogenous_columns: list[str],
) -> pd.DataFrame:
    """Tạo cờ dữ liệu thiếu để mô hình nhận biết trạng thái gián đoạn của cảm biến."""
    result = frame.copy()
    lag_1_col = f"{target_column}_lag_1"
    if lag_1_col in result.columns:
        result[f"{target_column}_lag_1_missing"] = result[lag_1_col].isna().astype(float)

    # Đếm số biến ngoại sinh đang bị thiếu.
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
    source_timezone = data_config.get("source_timezone", DEFAULT_SOURCE_TIMEZONE)
    result = frame.sort_values([station, timestamp], kind="stable").copy()
    label_source = result.copy()

    # Đặc trưng hiện tại cũng phải tuân thủ available_at. Nhãn t+1 bên dưới
    # được lookup riêng và không bị giới hạn bởi thời điểm phát hành nhãn.
    if "available_at" in result.columns:
        result[target] = lookup_feature_at_offset(
            result,
            station,
            timestamp,
            target,
            offset_hours=0,
            source_timezone=source_timezone,
        )

    # 1. Lag theo mốc giờ thực: không dịch chuyển theo vị trí dòng.
    for lag in config["features"]["lags"]:
        result[f"{target}_lag_{lag}"] = lookup_pm25_at_offset(
            result,
            station,
            timestamp,
            target,
            offset_hours=-lag,
            source_timezone=source_timezone,
        )

    # 2. Độ lệch xu hướng: chênh lệch nồng độ so với 1h và 3h trước.
    delta_lags = config["features"].get("delta_lags", [1, 3])
    result = add_trend_features(result, target, delta_lags=delta_lags)

    # 3. Thống kê rolling (mean và std) với closed='left' (chỉ dùng lịch sử trước t).
    include_rolling_std = config["features"].get("include_rolling_std", True)
    result = add_rolling_features(
        result,
        station,
        timestamp,
        target,
        config["features"]["rolling_windows"],
        include_std=include_rolling_std,
    )

    # 4. Đặc trưng thời gian tuần hoàn.
    result = add_time_features(
        result,
        timestamp,
        calendar_timezone=config.get("data", {}).get("calendar_timezone", "Asia/Ho_Chi_Minh"),
    )

    # 5. Đặc trưng ngoại sinh theo thời điểm đã có dữ liệu.
    result = prepare_exogenous_columns(result, config)

    # 6. Cờ tùy chọn cho dữ liệu thiếu.
    if config.get("features", {}).get("include_missingness_features", False):
        exo_cols = config.get("features", {}).get("exogenous_columns", [])
        result = add_missingness_features(result, target, exo_cols)

    # 7. Nhãn tương lai và baseline.
    if include_target:
        result["target_timestamp"] = result[timestamp] + pd.to_timedelta(1, unit="h")
        result["target_next_hour"] = lookup_pm25_at_offset(
            label_source,
            station,
            timestamp,
            target,
            offset_hours=1,
            # Target tương lai là nhãn quan sát, không phải feature tại t.
            # Không dùng available_at của t để loại nhãn t+1.
            enforce_availability=False,
            source_timezone=source_timezone,
        )
        # Seasonal Naive 24h cho target(t+1):
        # \hat{y}_{t+1}^{seasonal24} = y_{(t+1)-24} = y_{t-23}
        # Tra cứu quan trắc tại cùng trạm ở mốc t - 23h.
        result["seasonal_naive_24h"] = lookup_pm25_at_offset(
            label_source,
            station,
            timestamp,
            target,
            offset_hours=-23,
            enforce_availability=False,
            source_timezone=source_timezone,
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
        f"{target}_delta_{lag}h" for lag in delta_lags if lag in config["features"].get("lags", [])
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
