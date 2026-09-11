"""Schema chuẩn và hợp đồng dữ liệu cho chuỗi thời gian chất lượng không khí.

Định nghĩa các cột chuẩn và quy tắc timestamp của nguồn CSV mẫu.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_SOURCE_TIMEZONE = "Asia/Ho_Chi_Minh"
CANONICAL_STORAGE_TIMEZONE = "UTC"

CANONICAL_COLUMNS = [
    "timestamp",
    "available_at",
    "station_id",
    "latitude",
    "longitude",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "rainfall",
]

PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    "PM2.5": (0.0, 1000.0),
    "PM10": (0.0, 1500.0),
    "NO2": (0.0, 1000.0),
    "SO2": (0.0, 1000.0),
    "CO": (0.0, 200.0),
    "O3": (0.0, 1000.0),
    "temperature": (-20.0, 60.0),
    "humidity": (0.0, 100.0),
    "wind_speed": (0.0, 100.0),
    "wind_direction": (0.0, 360.0),
    "rainfall": (0.0, 500.0),
}


def normalize_timestamp_series(
    values: pd.Series,
    *,
    source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> pd.Series:
    """Chuẩn hóa timestamp về UTC có timezone.

    Timestamp không có timezone được hiểu là giờ địa phương TP.HCM. Timestamp
    đã có timezone được đổi sang UTC. Quy ước này phải được dùng ở mọi điểm
    vào pipeline để huấn luyện, backtest và dự báo dùng cùng một trục thời gian.
    """
    parsed = pd.to_datetime(values, errors="coerce")
    if getattr(parsed.dtype, "tz", None) is None:
        try:
            return parsed.dt.tz_localize(source_timezone).dt.tz_convert(
                CANONICAL_STORAGE_TIMEZONE
            )
        except (AttributeError, TypeError):
            # Input vừa naive vừa aware tạo object dtype; chuẩn hóa từng phần tử
            # để không lặng lẽ biến một phần timestamp thành UTC giả.
            normalized = []
            for value in parsed:
                timestamp = pd.Timestamp(value) if pd.notna(value) else pd.NaT
                if pd.isna(timestamp):
                    normalized.append(pd.NaT)
                elif timestamp.tzinfo is None:
                    normalized.append(
                        timestamp.tz_localize(source_timezone).tz_convert(
                            CANONICAL_STORAGE_TIMEZONE
                        )
                    )
                else:
                    normalized.append(timestamp.tz_convert(CANONICAL_STORAGE_TIMEZONE))
            return pd.Series(normalized, index=values.index)
    return parsed.dt.tz_convert(CANONICAL_STORAGE_TIMEZONE)
