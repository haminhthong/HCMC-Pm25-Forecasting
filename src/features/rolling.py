"""Rolling window statistical features with strict closed='left' causality."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_rolling_features(
    frame: pd.DataFrame,
    station_column: str,
    timestamp_column: str,
    target_column: str,
    windows: list[int],
    include_std: bool = True,
) -> pd.DataFrame:
    """Tạo rolling mean và rolling std theo cửa sổ giờ với closed='left'.

    closed='left' đảm bảo quan trắc tại thời điểm t hiện tại bị loại trừ khỏi cửa sổ rolling.
    """
    result = frame.copy()
    for window in windows:
        mean_col = f"{target_column}_rolling_mean_{window}"
        result[mean_col] = np.nan
        std_col = f"{target_column}_rolling_std_{window}" if include_std else None
        if std_col:
            result[std_col] = np.nan

        for _, station_frame in result.groupby(station_column, sort=False):
            series = station_frame.set_index(timestamp_column)[target_column]
            rolling = series.rolling(
                f"{window}h",
                closed="left",
                min_periods=1,
            )
            result.loc[station_frame.index, mean_col] = rolling.mean().to_numpy()
            if std_col:
                result.loc[station_frame.index, std_col] = rolling.std().to_numpy()
    return result


def add_trend_features(
    frame: pd.DataFrame,
    target_column: str,
    delta_lags: list[int] = (1, 3),
) -> pd.DataFrame:
    """Tạo đặc trưng độ dốc/chênh lệch PM2.5 theo mốc giờ thực: delta = y(t) - y(t-lag)."""
    result = frame.copy()
    for lag in delta_lags:
        lag_col = f"{target_column}_lag_{lag}"
        if lag_col in result.columns:
            result[f"{target_column}_delta_{lag}h"] = result[target_column] - result[lag_col]
    return result
