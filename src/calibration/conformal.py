"""Conformal Prediction utilities (P0.2: finite-sample split-conformal).

This module extracts conformal logic from ``src/train.py`` so that the same
implementation can be reused by training, evaluation, and serving. The
quantile formula follows the finite-sample correction:

    rank = ceil((n + 1) * coverage)
    rank = min(rank, n)
    quantile = sorted_residuals[rank - 1]

which is the standard recommendation for split-conformal prediction intervals
when the calibration set has size ``n``. This avoids the under-coverage bias
of the empirical ``np.quantile(residuals, coverage)`` estimator on small
calibration sets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def conformal_quantile(residuals: Sequence[float] | np.ndarray, coverage: float) -> float:
    """Compute the finite-sample split-conformal residual quantile.

    Parameters
    ----------
    residuals : array-like
        Absolute residuals ``|y - y_hat|`` observed on the calibration set.
        Must be non-empty. Caller is responsible for ensuring the calibration
        set is independent from training data.
    coverage : float
        Target marginal coverage in (0, 1). For coverage=0.9 and n=10 the
        returned quantile corresponds to the 10th smallest residual (rank
        ceil((10+1)*0.9)=10), which is the largest residual in the set.

    Returns
    -------
    float
        The conformal radius ``q`` such that the interval ``y_hat +/- q``
        achieves at least ``coverage`` coverage on the calibration sample.

    Raises
    ------
    ValueError
        If ``residuals`` is empty or ``coverage`` is outside (0, 1).
    """
    arr = np.asarray(list(residuals), dtype=float)
    if arr.size == 0:
        raise ValueError(
            "conformal_quantile yêu cầu calibration set không rỗng; "
            "không được fallback sang train/test (xem P0.1)."
        )
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage phải nằm trong (0, 1); nhận {coverage!r}.")

    n = arr.size
    rank = math.ceil((n + 1) * coverage)
    rank = min(rank, n)  # nếu ceil > n thì lấy residual lớn nhất
    sorted_residuals = np.sort(arr)
    return float(sorted_residuals[rank - 1])


def split_conformal_residuals(y_true: Sequence[float], y_pred: Sequence[float]) -> np.ndarray:
    """Compute the absolute residuals used for split-conformal calibration."""
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true và y_pred phải cùng shape; nhận {y_true_arr.shape} vs {y_pred_arr.shape}."
        )
    return np.abs(y_true_arr - y_pred_arr)
