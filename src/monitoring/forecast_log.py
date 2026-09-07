"""Lưu forecast event và backfill nhãn trễ cho monitoring."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd

FORECAST_EVENT_COLUMNS = [
    "forecast_id",
    "model_version",
    "station_id",
    "forecast_origin",
    "forecast_for",
    "current_pm25",
    "prediction",
    "lower",
    "upper",
    "strategy",
    "data_quality_status",
    "created_at",
    "actual_pm25",
    "error",
    "persistence_prediction",
]


def append_forecast_event(event: dict[str, Any], path: str | Path) -> Path:
    """Append một forecast event vào CSV; event_id giúp retry không tạo dòng trùng."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = {column: event.get(column) for column in FORECAST_EVENT_COLUMNS}
    row["forecast_id"] = row["forecast_id"] or str(uuid.uuid4())
    if destination.is_file():
        existing = pd.read_csv(destination, usecols=["forecast_id"])
        if row["forecast_id"] in set(existing["forecast_id"].astype(str)):
            return destination
    pd.DataFrame([row], columns=FORECAST_EVENT_COLUMNS).to_csv(
        destination,
        mode="a",
        header=not destination.is_file(),
        index=False,
    )
    return destination


def backfill_actuals(
    event_path: str | Path,
    observations: pd.DataFrame,
    *,
    station_column: str,
    timestamp_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Join actual PM2.5 đúng một lần theo ``(station_id, forecast_for)``."""
    destination = Path(event_path)
    if not destination.is_file():
        return pd.DataFrame(columns=FORECAST_EVENT_COLUMNS)
    events = pd.read_csv(destination)
    if events.empty:
        return events

    actuals = observations[[station_column, timestamp_column, target_column]].copy()
    actuals["_forecast_key"] = pd.to_datetime(actuals[timestamp_column], utc=True, errors="coerce")
    actuals = actuals.rename(columns={station_column: "station_id", target_column: "actual_pm25"})
    actuals = actuals.drop_duplicates(["station_id", "_forecast_key"], keep="last")
    events["_forecast_key"] = pd.to_datetime(events["forecast_for"], utc=True, errors="coerce")
    events = events.drop(columns=["actual_pm25", "error"], errors="ignore")
    joined = events.merge(
        actuals[["station_id", "_forecast_key", "actual_pm25"]],
        on=["station_id", "_forecast_key"],
        how="left",
    )
    joined["error"] = joined["prediction"] - joined["actual_pm25"]
    joined = joined.drop(columns=["_forecast_key"])
    joined.to_csv(destination, index=False)
    return joined
