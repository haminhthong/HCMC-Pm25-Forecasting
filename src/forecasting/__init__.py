"""Forecasting package for models, baselines, training, and champion selection."""

from src.forecasting.baselines import (
    persistence_predictions,
    seasonal_naive_predictions,
)
from src.forecasting.models import build_model
from src.forecasting.selector import build_quality_gate, resolve_model_statuses
from src.forecasting.trainer import make_pipeline, resolve_candidate_params

__all__ = [
    "build_model",
    "build_quality_gate",
    "make_pipeline",
    "persistence_predictions",
    "resolve_candidate_params",
    "resolve_model_statuses",
    "seasonal_naive_predictions",
]
