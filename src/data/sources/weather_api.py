"""Connector khí tượng hourly từ Open-Meteo Archive API."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.data.schema import AirQualityDataset, normalize_timestamp_series
from src.data.sources.base import BaseSource


def _get_json(url: str, timeout: int) -> dict[str, Any]:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "hcmc-pm25-forecasting/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL do cấu hình
        return json.loads(response.read().decode("utf-8"))


class WeatherAPISource(BaseSource):
    """Lấy weather observation theo tọa độ và chuẩn hóa về một station_id."""

    def __init__(
        self,
        api_url: str = "https://archive-api.open-meteo.com/v1/archive",
        latitude: float = 10.7769,
        longitude: float = 106.7009,
        station_id: str | None = None,
        timeout: int = 30,
    ):
        self.api_url = api_url
        self.latitude = latitude
        self.longitude = longitude
        self.station_id = station_id or f"weather_{latitude:.4f}_{longitude:.4f}"
        self.timeout = timeout

    def fetch(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> AirQualityDataset:
        """Fetch dữ liệu thật, không trả trạng thái thành công cho frame rỗng."""
        end = end_time or datetime.now(UTC)
        start = start_time or (end - timedelta(hours=24))
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
            "timezone": "UTC",
        }
        payload = _get_json(
            f"{self.api_url}?{urlencode(params)}",
            timeout=self.timeout,
        )
        hourly = payload.get("hourly", {})
        frame = pd.DataFrame(
            {
                "timestamp": hourly.get("time", []),
                "station_id": self.station_id,
                "temperature": hourly.get("temperature_2m", []),
                "humidity": hourly.get("relative_humidity_2m", []),
                "wind_speed": hourly.get("wind_speed_10m", []),
                "wind_direction": hourly.get("wind_direction_10m", []),
                "rainfall": hourly.get("precipitation", []),
            }
        )
        fetched_at = datetime.now(UTC)
        if frame.empty:
            frame = pd.DataFrame(
                columns=[
                    "timestamp",
                    "station_id",
                    "temperature",
                    "humidity",
                    "wind_speed",
                    "wind_direction",
                    "rainfall",
                    "available_at",
                ]
            )
        else:
            frame["timestamp"] = normalize_timestamp_series(frame["timestamp"])
            # Archive data đại diện cho quan trắc đã có sẵn tại timestamp.
            frame["available_at"] = frame["timestamp"]

        return AirQualityDataset(
            frame=frame,
            source=f"weather:{self.api_url}",
            snapshot_id=f"weather-api-{fetched_at.strftime('%Y%m%d%H%M%S')}",
            timezone="UTC",
            metadata={
                "latitude": self.latitude,
                "longitude": self.longitude,
                "station_id": self.station_id,
                "status": "fetched",
                "request_start": start.isoformat(),
                "request_end": end.isoformat(),
                "fetched_at_utc": fetched_at.isoformat(),
                "row_count": int(len(frame)),
            },
        )
