"""Cyclic temporal encodings for diurnal and day-of-week patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Mã hóa giờ theo chu kỳ (cyclic encoding sin/cos) và bổ sung thứ trong tuần."""
    result = frame.copy()
    hour = result[timestamp_column].dt.hour
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_of_week"] = result[timestamp_column].dt.dayofweek
    return result
