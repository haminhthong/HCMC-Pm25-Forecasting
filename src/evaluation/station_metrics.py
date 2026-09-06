"""Per-station sliced performance evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_and_classification_metrics


def metrics_by_station(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    station_column: str,
    thresholds: dict[str, Any],
    conformal_residual_q90: float | None = None,
) -> dict[str, Any]:
    """Tính toàn bộ metric cho từng trạm độc lập."""
    result: dict[str, Any] = {}
    work = test_frame.copy()
    work["_pred"] = predictions

    for station_name, group in work.groupby(station_column, sort=False):
        y_true = group["target_next_hour"].to_numpy()
        y_pred = group["_pred"].to_numpy()
        metrics = regression_and_classification_metrics(y_true, y_pred, thresholds)
        metrics["count"] = int(len(group))

        if conformal_residual_q90 is not None:
            lower = np.maximum(0.0, y_pred - conformal_residual_q90)
            upper = y_pred + conformal_residual_q90
            covered = (y_true >= lower) & (y_true <= upper)
            metrics["conformal_picp"] = float(np.mean(covered))
            metrics["conformal_mpiw"] = float(np.mean(upper - lower))

        result[str(station_name)] = metrics

    return result
