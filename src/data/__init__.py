"""Data package: schema, sources, quality audits, hourly regularization, and snapshots."""

from src.config import load_config, validate_config
from src.data.loader import load_air_quality, resolve_data_path
from src.data.quality import audit_air_quality, validate_physical_ranges, validate_schema
from src.data.regularization import (
    HOURLY_FREQ,
    audit_hourly_gaps,
    regularize_hourly_series,
)
from src.data.runtime_gate import audit_runtime_history
from src.data.schema import (
    CANONICAL_COLUMNS,
    PHYSICAL_RANGES,
    AirQualityDataset,
    StationMetadata,
)
from src.data.snapshot import load_snapshot, save_snapshot

__all__ = [
    "CANONICAL_COLUMNS",
    "HOURLY_FREQ",
    "PHYSICAL_RANGES",
    "AirQualityDataset",
    "StationMetadata",
    "audit_air_quality",
    "audit_runtime_history",
    "audit_hourly_gaps",
    "load_air_quality",
    "load_config",
    "load_snapshot",
    "regularize_hourly_series",
    "resolve_data_path",
    "save_snapshot",
    "validate_config",
    "validate_physical_ranges",
    "validate_schema",
]
