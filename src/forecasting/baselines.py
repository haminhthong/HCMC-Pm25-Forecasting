"""Fundamental time-series forecasting baselines: Persistence and Seasonal Naive 24h."""

from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_predictions(frame: pd.DataFrame, target_column: str) -> np.ndarray:
    """Baseline persistence: giá trị giờ tới bằng quan trắc hiện tại (t+1 = t)."""
    return frame[target_column].to_numpy(dtype=float)


def seasonal_naive_predictions(frame: pd.DataFrame, target_column: str) -> np.ndarray:
    r"""Baseline chu kỳ 24 giờ: dự báo t+1 bằng quan trắc tại cùng giờ hôm trước (t - 23h).

    Công thức: \hat{y}_{t+1}^{seasonal24} = y_{(t+1)-24} = y_{t-23}.
    Fallback về persistence nếu lịch sử chưa đủ 24 giờ.
    """
    return frame["seasonal_naive_24h"].fillna(frame[target_column]).to_numpy(dtype=float)
