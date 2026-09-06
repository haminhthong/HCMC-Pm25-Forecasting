"""Exact Clock-Time Lag Lookups (Leakage-Safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def lookup_pm25_at_offset(
    frame: pd.DataFrame,
    station_column: str,
    timestamp_column: str,
    target_column: str,
    offset_hours: int,
) -> np.ndarray:
    """Tra PM2.5 tại một độ lệch giờ chính xác trong cùng trạm (Exact Clock-Time Lookup).

    Hàm dùng khóa ``(station, timestamp + offset_hours)`` thay vì dịch theo số dòng
    (row-position shift). Vì vậy, nếu dữ liệu bị khuyết một giờ, lag 1 giờ sẽ là NaN
    thay vì lấy nhầm quan trắc gần nhất cách đó nhiều giờ.
    """
    lookup = frame.set_index([station_column, timestamp_column])[target_column]
    lookup_keys = pd.MultiIndex.from_arrays(
        [
            frame[station_column],
            frame[timestamp_column] + pd.to_timedelta(offset_hours, unit="h"),
        ],
        names=[station_column, timestamp_column],
    )
    return lookup.reindex(lookup_keys).to_numpy()
