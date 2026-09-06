"""Backward-compatible facade for serving Predictor.

Re-exports ``Predictor`` from ``src.serving.predictor``.
"""

from __future__ import annotations

from src.serving.predictor import Predictor

__all__ = ["Predictor"]
