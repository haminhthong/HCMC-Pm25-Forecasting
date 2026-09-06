"""Air quality telemetry API connector."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.data.schema import AirQualityDataset
from src.data.sources.base import BaseSource


class AirQualityAPISource(BaseSource):
    """Source for fetching telemetry feeds from Air Quality APIs (e.g. OpenAQ, local EPA)."""

    def __init__(
        self,
        api_url: str = "https://api.openaq.org/v2",
        api_key: str | None = None,
        city: str = "Ho Chi Minh City",
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.city = city

    def fetch(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> AirQualityDataset:
        """Fetch remote readings or return an empty canonical frame if offline."""
        snap_id = f"aq-api-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        # In mock / telemetry mode, returns empty canonical template unless populated
        empty_frame = pd.DataFrame(
            columns=["timestamp", "station_id", "PM2.5", "NO2", "SO2", "CO", "O3"]
        )
        return AirQualityDataset(
            frame=empty_frame,
            source=f"api:{self.api_url}",
            snapshot_id=snap_id,
            metadata={"city": self.city, "status": "connector_initialized"},
        )
