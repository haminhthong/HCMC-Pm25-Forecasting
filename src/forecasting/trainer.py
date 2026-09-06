"""Pipeline trainer and preprocessor constructor."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.builder import model_feature_columns
from src.forecasting.models import build_model


def resolve_candidate_params(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Return hyperparameters for ``model_name`` according to config contract (P0.5)."""
    models_section = config.get("models") or {}
    if model_name in models_section:
        params = models_section[model_name]
        return dict(params) if isinstance(params, dict) else {}
    legacy_champion = config.get("model", {}).get("name")
    if model_name == legacy_champion:
        legacy_params = config.get("model", {}).get("params") or {}
        return dict(legacy_params) if isinstance(legacy_params, dict) else {}
    return {}


def make_pipeline(config: dict[str, Any], model_name: str) -> tuple[Pipeline, list[str]]:
    """Tạo preprocessing và model trong cùng một sklearn Pipeline.

    Point 10:
    - Với Ridge: dùng SimpleImputer + StandardScaler.
    - Với Tree Ensembles (RandomForest, ExtraTrees, HistGradientBoosting):
      chỉ dùng SimpleImputer (StandardScaler không cần thiết và được lược bỏ).
    """
    station = config["data"]["station_column"]
    numeric = model_feature_columns(config)

    if model_name == "ridge":
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    else:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]

    numeric_pipeline = Pipeline(numeric_steps)
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            (
                "station",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [station],
            ),
        ]
    )
    params = resolve_candidate_params(config, model_name)
    model = build_model(model_name, config["project"]["random_state"], params)
    return Pipeline([("preprocess", preprocessor), ("model", model)]), [*numeric, station]
