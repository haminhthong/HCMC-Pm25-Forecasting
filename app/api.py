"""FastAPI endpoint cho dự báo PM2.5 một giờ tiếp theo."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data.loader import load_air_quality
from src.inference.predictor import Predictor

DEFAULT_ARTIFACT_ROOT = Path("artifacts")


class Observation(BaseModel):
    """Một quan trắc đầu vào theo schema của sample CSV."""

    timestamp: datetime
    available_at: datetime | None = None
    station_id: str = Field(min_length=1, max_length=100)
    PM25: float = Field(alias="PM2.5", ge=0, le=1000)
    TSP: float | None = Field(default=None, ge=0)
    NO2: float | None = Field(default=None, ge=0)
    SO2: float | None = Field(default=None, ge=0)
    CO: float | None = Field(default=None, ge=0)
    O3: float | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=None, ge=-20, le=60)
    humidity: float | None = Field(default=None, ge=0, le=100)

    model_config = {"populate_by_name": True}


class PredictionRequest(BaseModel):
    """History một trạm dùng để tạo đặc trưng và dự báo."""

    observations: list[Observation] = Field(min_length=25, max_length=168)


class Interval(BaseModel):
    """Khoảng dự báo split-conformal."""

    method: str = "split_conformal_prediction_interval"
    coverage_target: float = Field(default=0.9, ge=0.0, le=1.0)
    coverage: float = Field(default=0.9, ge=0.0, le=1.0)
    lower: float
    upper: float
    width: float | None = None


class PredictionResponse(BaseModel):
    """Schema phản hồi của endpoint dự báo."""

    station: str
    station_id: str | None = None
    forecast_origin: str
    forecast_for: str
    current_pm25: float
    predicted_pm25: float
    level: str
    forecast_strategy: str
    best_cv_model: str | None = None
    dataset_scope: str = "sample"
    is_out_of_distribution: bool = False
    interval: Interval
    updated_at: str
    data_quality: dict[str, object] = Field(default_factory=dict)


def _resolve_artifact_root() -> Path:
    """Lấy thư mục artifact từ biến môi trường hoặc mặc định của repo."""
    return Path(os.environ.get("PM25_ARTIFACT_DIR", DEFAULT_ARTIFACT_ROOT))


@lru_cache
def get_predictor() -> Predictor:
    """Nạp model một lần và tái sử dụng giữa các request."""
    return Predictor.from_artifact(_resolve_artifact_root())


app = FastAPI(
    title="API dự báo PM2.5 TP.HCM",
    version="1.0.0",
    description="Dự báo nồng độ PM2.5 giờ tiếp theo từ history theo trạm.",
)


@app.get("/health")
def health() -> dict[str, object]:
    """Kiểm tra model artifact hiện tại đã được nạp."""
    try:
        predictor = get_predictor()
        if predictor.model is None:
            raise ValueError("Model is None")
        return {
            "status": "ready",
            "model_loaded": True,
            "best_cv_model": predictor.metadata.get("best_cv_model"),
            "forecast_strategy": predictor.metadata.get("forecast_strategy"),
            "dataset_scope": predictor.metadata.get("dataset_scope", "sample"),
        }
    except Exception as err:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng.") from err


@app.post("/predict", response_model=PredictionResponse)
@app.post("/v1/predict/raw", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Dự báo từ history raw gửi trong request."""
    try:
        records = [
            item.model_dump(by_alias=True, exclude_none=True) for item in request.observations
        ]
        return get_predictor().predict(pd.DataFrame(records))
    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={"code": "INVALID_INPUT", "message": str(error)},
        )
    except FileNotFoundError:
        return JSONResponse(
            status_code=503,
            content={"code": "MODEL_UNAVAILABLE", "message": "Model artifact chưa sẵn sàng."},
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"code": "INTERNAL_ERROR", "message": "Lỗi nội bộ hệ thống."},
        )


@app.get("/v1/stations/{station_id}/forecast", response_model=PredictionResponse)
def forecast_station(station_id: str):
    """Lấy 168 giờ gần nhất của một trạm từ CSV rồi dự báo."""
    try:
        predictor = get_predictor()
        config = predictor.config
        frame = load_air_quality(config)
        station_column = config["data"]["station_column"]
        history = (
            frame[frame[station_column].astype(str) == station_id]
            .sort_values(config["data"]["timestamp_column"], kind="stable")
            .tail(168)
        )
        if history.empty:
            raise ValueError(f"Không tìm thấy history cho station_id={station_id!r}.")
        return predictor.predict(history)
    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={"code": "INVALID_INPUT", "message": str(error)},
        )
    except FileNotFoundError as error:
        return JSONResponse(
            status_code=503,
            content={"code": "DATA_UNAVAILABLE", "message": str(error)},
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"code": "INTERNAL_ERROR", "message": "Lỗi nội bộ hệ thống."},
        )
