"""Model builder with parameter forwarding for all candidate architectures."""

from __future__ import annotations

from typing import Any

from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge


def build_model(
    name: str,
    random_state: int,
    params: dict[str, Any] | None = None,
) -> RegressorMixin:
    """Tạo model từ tên với params được truyền đầy đủ (P0.5 fix)."""
    params = dict(params or {})
    if name == "ridge":
        defaults = {"alpha": 1.0}
        return Ridge(random_state=random_state, **(defaults | params))

    if name == "random_forest":
        defaults = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2, "n_jobs": -1}
        return RandomForestRegressor(random_state=random_state, **(defaults | params))

    if name == "extra_trees":
        defaults = {"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": -1}
        return ExtraTreesRegressor(random_state=random_state, **(defaults | params))

    if name == "hist_gradient_boosting":
        defaults = {"max_iter": 200, "learning_rate": 0.05}
        return HistGradientBoostingRegressor(random_state=random_state, **(defaults | params))

    raise ValueError(f"Mô hình không được hỗ trợ: {name}")
