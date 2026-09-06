"""Artifact schemas and ForecastContext definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ForecastContext:
    """Operational forecasting context governing serving and feature construction (Point 13)."""

    horizon_hours: int = 1
    frequency: str = "1h"
    timezone: str = "Asia/Ho_Chi_Minh"
    required_history_hours: int = 25
    allowed_gap_hours: int = 6
    feature_schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastContext:
        return cls(
            horizon_hours=int(data.get("horizon_hours", 1)),
            frequency=str(data.get("frequency", "1h")),
            timezone=str(data.get("timezone", "Asia/Ho_Chi_Minh")),
            required_history_hours=int(data.get("required_history_hours", 25)),
            allowed_gap_hours=int(data.get("allowed_gap_hours", 6)),
            feature_schema_version=str(data.get("feature_schema_version", "2.0")),
        )
