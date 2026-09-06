"""Exogenous feature processing with Data Availability Contract.

Defines operational latency offsets for atmospheric and pollutant variables
to ensure offline training features match online telemetry availability.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def get_feature_availability(config: dict[str, Any]) -> dict[str, int]:
    """Return latency offset in hours for each exogenous feature."""
    return config.get("feature_availability", {})


def prepare_exogenous_columns(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Standardize exogenous features according to the availability contract."""
    result = frame.copy()
    availability = get_feature_availability(config)
    exogenous = config.get("features", {}).get("exogenous_columns", [])

    for col in exogenous:
        if col in result.columns:
            latency = availability.get(col, 0)
            if latency > 0:
                # If variable has latency (e.g. 1h), use delayed version
                result[col] = result[col].shift(latency)

    return result
