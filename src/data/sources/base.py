"""Base source abstraction for data ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.data.schema import AirQualityDataset


class BaseSource(ABC):
    """Abstract interface for raw observation sources."""

    @abstractmethod
    def fetch(self, **kwargs: Any) -> AirQualityDataset:
        """Fetch raw observations and return a standardized AirQualityDataset."""
        raise NotImplementedError
