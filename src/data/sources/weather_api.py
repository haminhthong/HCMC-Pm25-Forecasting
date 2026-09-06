"""Meteorological weather API connector."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.data.schema import AirQualityDataset
from src.data.sources.base import BaseSource


class WeatherAPISource(BaseSource):
    """Source for fetching surface meteorology feeds (e.g. Open-Meteo, ECMWF)."""

    def __init__(
        self,
        api_url: str = "https://api.open-meteo.com/v1/forecast",
        latitude: float = 10.7769,
        longitude: float = 106.7009,
    ):
        self.api_url = api_url
        self.latitude = latitude
        self.longitude = longitude

    def fetch(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> AirQualityDataset:
        snap_id = f"weather-api-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        empty_frame = pd.DataFrame(
            columns=["timestamp", "temperature", "humidity", "wind_speed", "wind_direction", "rainfall"]
        )
        return AirQualityDataset(
            frame=empty_frame,
            source=f"weather:{self.api_url}",
            snapshot_id=snap_id,
            metadata={
                "latitude": self.latitude,
                "longitude": self.longitude,
                "status": "connector_initialized",
            },
        )
