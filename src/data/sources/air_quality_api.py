"""Connector đọc quan trắc PM2.5 từ OpenAQ-compatible API."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.data.schema import AirQualityDataset, normalize_timestamp_series
from src.data.sources.base import BaseSource

PARAMETER_MAP = {
    "pm25": "PM2.5",
    "pm2.5": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "o3": "O3",
}


def _get_json(url: str, *, api_key: str | None, timeout: int) -> dict[str, Any]:
    """Gọi HTTP bằng thư viện chuẩn để connector không phụ thuộc requests."""
    headers = {"Accept": "application/json", "User-Agent": "hcmc-pm25-forecasting/1.0"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL do cấu hình
        return json.loads(response.read().decode("utf-8"))


class AirQualityAPISource(BaseSource):
    """Đọc phép đo và gom về một dòng cho mỗi trạm/timestamp."""

    def __init__(
        self,
        api_url: str = "https://api.openaq.org/v2/measurements",
        api_key: str | None = None,
        city: str = "Ho Chi Minh City",
        timeout: int = 30,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.city = city
        self.timeout = timeout

    def fetch(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> AirQualityDataset:
        """Fetch dữ liệu thật; lỗi mạng được báo rõ thay vì trả DataFrame rỗng."""
        end = end_time or datetime.now(timezone.utc)
        start = start_time or (end - timedelta(hours=24))
        params: dict[str, Any] = {
            "city": self.city,
            "date_from": start.isoformat(),
            "date_to": end.isoformat(),
            "limit": int(kwargs.get("limit", 10000)),
        }
        parameters = kwargs.get("parameters", list(PARAMETER_MAP))
        if parameters:
            params["parameter"] = ",".join(parameters)
        payload = _get_json(
            f"{self.api_url}?{urlencode(params)}",
            api_key=self.api_key,
            timeout=self.timeout,
        )

        rows: list[dict[str, Any]] = []
        fetched_at = datetime.now(timezone.utc)
        for item in payload.get("results", []):
            raw_parameter = item.get("parameter", "")
            if isinstance(raw_parameter, dict):
                raw_parameter = raw_parameter.get("name", raw_parameter.get("id", ""))
            parameter = str(raw_parameter).lower()
            canonical_parameter = PARAMETER_MAP.get(parameter)
            if canonical_parameter is None:
                continue
            date_info = item.get("date", {})
            observed_at = item.get("datetime") or date_info.get("utc") or date_info.get("local")
            if observed_at is None:
                continue
            station_id = (
                item.get("location_id")
                or item.get("location")
                or item.get("station_id")
                or "unknown"
            )
            coordinates = item.get("coordinates") or {}
            rows.append(
                {
                    "timestamp": observed_at,
                    "station_id": str(station_id),
                    canonical_parameter: item.get("value"),
                    "available_at": fetched_at,
                    "latitude": coordinates.get("latitude"),
                    "longitude": coordinates.get("longitude"),
                }
            )

        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(
                columns=[
                    "timestamp",
                    "station_id",
                    "PM2.5",
                    "NO2",
                    "SO2",
                    "CO",
                    "O3",
                    "available_at",
                ]
            )
        else:
            frame["timestamp"] = normalize_timestamp_series(frame["timestamp"])
            frame["available_at"] = normalize_timestamp_series(frame["available_at"])
            value_columns = [column for column in PARAMETER_MAP.values() if column in frame.columns]
            aggregations: dict[str, str] = {
                column: "mean" for column in value_columns
            }
            aggregations.update({"available_at": "max", "latitude": "first", "longitude": "first"})
            frame = frame.groupby(["station_id", "timestamp"], as_index=False).agg(aggregations)

        return AirQualityDataset(
            frame=frame,
            source=f"api:{self.api_url}",
            snapshot_id=f"aq-api-{fetched_at.strftime('%Y%m%d%H%M%S')}",
            timezone="UTC",
            metadata={
                "city": self.city,
                "status": "fetched",
                "request_start": start.isoformat(),
                "request_end": end.isoformat(),
                "fetched_at_utc": fetched_at.isoformat(),
                "row_count": int(len(frame)),
            },
        )
