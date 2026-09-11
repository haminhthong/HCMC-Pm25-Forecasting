"""Tạo mô hình và chuyển tham số từ cấu hình."""

from __future__ import annotations

from typing import Any

from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

SUPPORTED_MODELS = frozenset({"ridge", "hist_gradient_boosting"})


def build_model(
    name: str,
    random_state: int,
    params: dict[str, Any] | None = None,
) -> RegressorMixin:
    """Tạo mô hình từ tên và các tham số đã khai báo."""
    params = dict(params or {})
    if name == "ridge":
        defaults = {"alpha": 1.0}
        return Ridge(random_state=random_state, **(defaults | params))

    if name == "hist_gradient_boosting":
        defaults = {"max_iter": 200, "learning_rate": 0.05}
        return HistGradientBoostingRegressor(random_state=random_state, **(defaults | params))

    raise ValueError(f"Mô hình không được hỗ trợ: {name}")
