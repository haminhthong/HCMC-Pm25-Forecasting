"""Historical CSV source ingestor."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import AirQualityDataset
from src.data.sources.base import BaseSource
from src.utils import sha256_file


class HistoricalCSVSource(BaseSource):
    """Ingest air quality observations from historical CSV archives."""

    def __init__(
        self,
        filepath: str | Path,
        column_mapping: dict[str, str] | None = None,
        snapshot_id: str | None = None,
    ):
        self.filepath = Path(filepath)
        self.column_mapping = column_mapping or {}
        self.snapshot_id = snapshot_id

    def fetch(self, **kwargs: Any) -> AirQualityDataset:
        if not self.filepath.is_file():
            raise FileNotFoundError(f"Không tìm thấy file CSV tại: {self.filepath}")

        df = pd.read_csv(self.filepath)

        # Apply rename mapping to canonical columns if provided
        if self.column_mapping:
            df = df.rename(columns=self.column_mapping)

        # Standardize station column if named 'station'
        if "station" in df.columns and "station_id" not in df.columns:
            df["station_id"] = df["station"]

        # Parse timestamps to UTC/ISO
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        snap_id = self.snapshot_id or f"csv-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        file_hash = sha256_file(self.filepath)

        return AirQualityDataset(
            frame=df,
            source=f"csv:{self.filepath.name}",
            snapshot_id=snap_id,
            frequency="1h",
            metadata={"filepath": str(self.filepath), "sha256": file_hash},
        )
