"""Exact Clock-Time Lag Lookups (Leakage-Safe)."""

from __future__ import annotations

import pandas as pd

from src.features.exogenous import lookup_feature_at_offset


def lookup_pm25_at_offset(
    frame: pd.DataFrame,
    station_column: str,
    timestamp_column: str,
    target_column: str,
    offset_hours: int,
    *,
    enforce_availability: bool = True,
) -> pd.Series:
    """Tra PM2.5 tại một độ lệch giờ chính xác trong cùng trạm (Exact Clock-Time Lookup).

    Hàm dùng khóa ``(station, timestamp + offset_hours)`` thay vì dịch theo số dòng
    (row-position shift). Vì vậy, nếu dữ liệu bị khuyết một giờ, lag 1 giờ sẽ là NaN
    thay vì lấy nhầm quan trắc gần nhất cách đó nhiều giờ.
    """
    return lookup_feature_at_offset(
        frame,
        station_column,
        timestamp_column,
        target_column,
        offset_hours,
        enforce_availability=enforce_availability,
    )
