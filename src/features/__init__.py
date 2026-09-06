"""Features package for leakage-safe air quality feature engineering."""

from src.features.builder import build_features, model_feature_columns
from src.features.exogenous import get_feature_availability, prepare_exogenous_columns
from src.features.lag import lookup_pm25_at_offset
from src.features.rolling import add_rolling_features, add_trend_features
from src.features.temporal import add_time_features

__all__ = [
    "add_rolling_features",
    "add_time_features",
    "add_trend_features",
    "build_features",
    "get_feature_availability",
    "lookup_pm25_at_offset",
    "model_feature_columns",
    "prepare_exogenous_columns",
]
