"""Validation package: time-ordered split, manifests, and expanding-window backtest."""

from src.validation.backtest import evaluate_candidate, expanding_time_folds
from src.validation.split import generate_split_manifest, split_by_time

__all__ = [
    "evaluate_candidate",
    "expanding_time_folds",
    "generate_split_manifest",
    "split_by_time",
]
