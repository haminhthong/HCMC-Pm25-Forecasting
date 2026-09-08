"""Metric monitoring cho forecast đã mature."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def rolling_forecast_metrics(events: pd.DataFrame, window: int = 168) -> dict[str, Any]:
    """Tính MAE, bias, skill và coverage trên các event đã có actual."""
    if events.empty or "actual_pm25" not in events.columns:
        return {"rows": 0, "mae": None, "bias": None, "skill_vs_persistence": None, "picp": None}
    work = events.dropna(subset=["actual_pm25", "prediction"]).tail(window).copy()
    if work.empty:
        return {"rows": 0, "mae": None, "bias": None, "skill_vs_persistence": None, "picp": None}
    error = work["prediction"].astype(float) - work["actual_pm25"].astype(float)
    if "persistence_prediction" not in work.columns:
        return {
            "rows": int(len(work)),
            "mae": float(np.abs(error).mean()),
            "bias": float(error.mean()),
            "skill_vs_persistence": None,
            "picp": None,
        }
    persistence_error = work["persistence_prediction"].astype(float) - work["actual_pm25"].astype(float)
    persistence_mae = float(np.abs(persistence_error).mean())
    lower = work["lower"].astype(float)
    upper = work["upper"].astype(float)
    picp = float(((work["actual_pm25"] >= lower) & (work["actual_pm25"] <= upper)).mean())
    return {
        "rows": int(len(work)),
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "skill_vs_persistence": (
            float(1 - np.abs(error).mean() / persistence_mae) if persistence_mae else 0.0
        ),
        "picp": picp,
        "interval_width": float((upper - lower).mean()),
        "fallback_rate": float((work["strategy"] == "persistence_fallback").mean()),
    }
