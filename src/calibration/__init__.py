"""Conformal Prediction module (P0.2: finite-sample split-conformal).

Public API lives in :mod:`src.calibration.conformal`.
"""

from __future__ import annotations

from src.calibration.conformal import (
    conformal_quantile,
    split_conformal_residuals,
)

__all__ = ["conformal_quantile", "split_conformal_residuals"]
