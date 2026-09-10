"""Mã hóa chu kỳ thời gian theo giờ trong ngày và thứ trong tuần."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(
    frame: pd.DataFrame,
    timestamp_column: str,
    calendar_timezone: str = "Asia/Ho_Chi_Minh",
) -> pd.DataFrame:
    """Mã hóa lịch theo giờ TP.HCM, không phụ thuộc timezone lưu trữ UTC."""
    result = frame.copy()
    timestamps = result[timestamp_column]
    if getattr(timestamps.dtype, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert(calendar_timezone)
    hour = timestamps.dt.hour
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_of_week"] = timestamps.dt.dayofweek
    return result
