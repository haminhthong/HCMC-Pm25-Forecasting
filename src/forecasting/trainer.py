"""Tạo pipeline tiền xử lý và mô hình cho từng ứng viên."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.builder import model_feature_columns
from src.forecasting.models import build_model


def resolve_candidate_params(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Lấy tham số đúng theo tên ứng viên trong cấu hình."""
    models_section = config.get("models") or {}
    params = models_section.get(model_name, {})
    return dict(params) if isinstance(params, dict) else {}


def make_pipeline(config: dict[str, Any], model_name: str) -> tuple[Pipeline, list[str]]:
    """Tạo tiền xử lý và mô hình trong cùng một sklearn Pipeline.

    - Ridge dùng SimpleImputer + StandardScaler.
    - HistGradientBoosting chỉ dùng SimpleImputer vì không cần StandardScaler.
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
