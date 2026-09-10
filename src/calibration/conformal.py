"""Tiện ích split-conformal theo hiệu chỉnh finite-sample.

Module này chứa logic conformal dùng chung cho huấn luyện, đánh giá và serving.
Công thức quantile áp dụng hiệu chỉnh finite-sample:

    rank = ceil((n + 1) * coverage)
    rank = min(rank, n)
    quantile = sorted_residuals[rank - 1]

Đây là cách tính phù hợp cho khoảng dự báo split-conformal khi calibration set
có ``n`` mẫu. Cách này tránh xu hướng đánh giá thiếu coverage của ước lượng
``np.quantile(residuals, coverage)`` trên calibration set nhỏ.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def conformal_quantile(residuals: Sequence[float] | np.ndarray, coverage: float) -> float:
    """Tính quantile residual split-conformal theo hiệu chỉnh finite-sample.

    ``residuals`` là residual tuyệt đối ``|y - y_hat|`` trên calibration set
    và phải không rỗng. Calibration set phải độc lập với dữ liệu huấn luyện.
    Với coverage bằng 0.9 và ``n=10``, rank bằng
    ``ceil((10+1)*0.9)=10`` nên quantile là residual lớn nhất.
    """
    arr = np.asarray(list(residuals), dtype=float)
    if arr.size == 0:
        raise ValueError(
            "conformal_quantile yêu cầu calibration set không rỗng; "
            "không được thay bằng residual của train hoặc test."
        )
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage phải nằm trong (0, 1); nhận {coverage!r}.")

    n = arr.size
    rank = math.ceil((n + 1) * coverage)
    rank = min(rank, n)  # nếu ceil > n thì lấy residual lớn nhất
    sorted_residuals = np.sort(arr)
    return float(sorted_residuals[rank - 1])


def split_conformal_residuals(y_true: Sequence[float], y_pred: Sequence[float]) -> np.ndarray:
    """Tính residual tuyệt đối dùng cho bước hiệu chuẩn split-conformal."""
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true và y_pred phải cùng shape; nhận {y_true_arr.shape} vs {y_pred_arr.shape}."
        )
    return np.abs(y_true_arr - y_pred_arr)


def station_conformal_quantiles(
    frame: pd.DataFrame,
    residuals: Sequence[float] | np.ndarray,
    *,
    station_column: str,
    coverage: float,
    minimum_samples: int = 20,
) -> tuple[float, dict[str, float]]:
    """Tính q theo trạm và fallback về q toàn cục khi mẫu quá ít."""
    residual_array = np.asarray(residuals, dtype=float)
    if len(frame) != len(residual_array):
        raise ValueError("frame và residuals phải có cùng số dòng.")
    frame = frame.reset_index(drop=True)
    global_q = conformal_quantile(residual_array, coverage)
    station_q: dict[str, float] = {}
    for station, indexes in frame.groupby(station_column, sort=False).groups.items():
        values = residual_array[np.asarray(list(indexes), dtype=int)]
        if len(values) >= minimum_samples:
            station_q[str(station)] = conformal_quantile(values, coverage)
    return global_q, station_q
