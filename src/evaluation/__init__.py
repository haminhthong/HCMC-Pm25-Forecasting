"""Evaluation package for regression, classification, slices, and intervals."""

from src.evaluation.evaluator import evaluate_baselines
from src.evaluation.metrics import (
    VALID_LABELS,
    classify_pm25,
    compute_mase,
    compute_skill_score,
    conformal_interval_metrics,
    get_threshold_params,
    regression_and_classification_metrics,
)
from src.evaluation.slices import sliced_error_analysis
from src.evaluation.station_metrics import metrics_by_station

__all__ = [
    "VALID_LABELS",
    "classify_pm25",
    "compute_mase",
    "compute_skill_score",
    "conformal_interval_metrics",
    "evaluate_baselines",
    "get_threshold_params",
    "metrics_by_station",
    "regression_and_classification_metrics",
    "sliced_error_analysis",
]
