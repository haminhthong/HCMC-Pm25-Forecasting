"""Test evaluator and baseline benchmarking."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.metrics import (
    regression_and_classification_metrics,
)
from src.forecasting.baselines import (
    persistence_predictions,
    seasonal_naive_predictions,
)


def evaluate_baselines(
    test_frame: pd.DataFrame,
    target: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Đánh giá hai baseline bắt buộc trên cùng tập test cuối."""
    target_next_hour = test_frame["target_next_hour"]
    return {
        "persistence": regression_and_classification_metrics(
            target_next_hour,
            persistence_predictions(test_frame, target),
            thresholds,
        ),
        "seasonal_naive_24h": regression_and_classification_metrics(
            target_next_hour,
            seasonal_naive_predictions(test_frame, target),
            thresholds,
        ),
    }
