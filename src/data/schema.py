"""Canonical Data Contract & Schema for Air Quality Time Series.

Defines the universal representation for air quality and meteorological observations
regardless of origin (historical CSV, telemetry API, weather API).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

DEFAULT_SOURCE_TIMEZONE = "Asia/Ho_Chi_Minh"
CANONICAL_STORAGE_TIMEZONE = "UTC"

CANONICAL_COLUMNS = [
    "timestamp",
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
    vào hệ thống để train, backtest và serving dùng cùng một trục thời gian.
    """
    parsed = pd.to_datetime(values, errors="coerce")
    if getattr(parsed.dtype, "tz", None) is None:
        try:
            return parsed.dt.tz_localize(source_timezone).dt.tz_convert(
                CANONICAL_STORAGE_TIMEZONE
            )
        except (AttributeError, TypeError):
            # Mixed naive/aware input tạo object dtype; chuẩn hóa từng phần tử
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


@dataclass
class StationMetadata:
    """Metadata describing an air quality monitoring station."""

    station_id: str
    station_name: str
    latitude: float | None = None
    longitude: float | None = None
    station_type: str = "urban_background"
    data_provider: str = "unknown"
    timezone: str = "Asia/Ho_Chi_Minh"

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "station_type": self.station_type,
            "data_provider": self.data_provider,
            "timezone": self.timezone,
        }


@dataclass
class AirQualityDataset:
    """Canonical contract for ingested and validated air quality datasets."""

    frame: pd.DataFrame
    source: str
    snapshot_id: str
    frequency: str = "1h"
    timezone: str = "Asia/Ho_Chi_Minh"
    station_ids: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.station_ids and "station_id" in self.frame.columns:
            self.station_ids = sorted(self.frame["station_id"].dropna().unique().tolist())
        if self.start_time is None and "timestamp" in self.frame.columns and not self.frame.empty:
            self.start_time = pd.to_datetime(self.frame["timestamp"].min()).to_pydatetime()
        if self.end_time is None and "timestamp" in self.frame.columns and not self.frame.empty:
            self.end_time = pd.to_datetime(self.frame["timestamp"].max()).to_pydatetime()

    def to_manifest(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "source": self.source,
            "frequency": self.frequency,
            "timezone": self.timezone,
            "station_ids": self.station_ids,
            "station_count": len(self.station_ids),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_rows": int(len(self.frame)),
            "columns": list(self.frame.columns),
            "metadata": self.metadata,
        }
