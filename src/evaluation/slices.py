"""Phân tích sai số theo thời gian và mức ô nhiễm."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import classify_pm25, get_threshold_params


def sliced_error_analysis(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    thresholds: dict[str, Any],
    timestamp_column: str = "timestamp",
) -> dict[str, Any]:
    """Phân tích sai số theo lát cắt giờ trong ngày và mức ô nhiễm."""
    work = test_frame.copy()
    work["_pred"] = predictions
    work["_abs_error"] = np.abs(work["target_next_hour"].to_numpy() - predictions)
    low_max, medium_max, labels = get_threshold_params(thresholds)
    work["_true_level"] = classify_pm25(
        work["target_next_hour"].to_numpy(), low_max, medium_max, labels
    )

    # Theo mức ô nhiễm
    by_level: dict[str, Any] = {}
    for level_name, group in work.groupby("_true_level", sort=False):
        by_level[str(level_name)] = {
            "count": int(len(group)),
            "mae": float(group["_abs_error"].mean()),
            "max_error": float(group["_abs_error"].max()),
        }

    # Theo giờ trong ngày
    by_hour: dict[str, Any] = {}
    work["_hour"] = pd.to_datetime(work[timestamp_column]).dt.hour
    for hour_val, group in work.groupby("_hour"):
        by_hour[str(hour_val)] = {
            "count": int(len(group)),
            "mae": float(group["_abs_error"].mean()),
        }

    return {
        "by_pollution_level": by_level,
        "by_hour_of_day": by_hour,
    }
