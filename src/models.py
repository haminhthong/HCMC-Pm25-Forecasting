"""Backward-compatible facade for model building.

Re-exports ``build_model`` from ``src.forecasting.models``.
"""

from __future__ import annotations

from src.forecasting.models import build_model

__all__ = ["build_model"]
