"""FastAPI serving layer for the PM2.5 forecast service.

P0 changes (2026-09-06):
* P0.4 — the predictor is built via ``Predictor.from_artifact`` which reads
  the model bundle from a versioned directory (default ``artifacts/``). The
  active version is resolved through ``production.json`` (or
  ``PM25_ARTIFACT_DIR`` env var). The legacy direct-load path is preserved
  via ``Predictor(load_config(...))`` for local development only.
* P0.6 — readiness fields are surfaced through ``/health`` and ``/predict``.
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data.loader import load_air_quality
from src.predict import Predictor

DEFAULT_ARTIFACT_ROOT = Path("artifacts")


class Observation(BaseModel):
    """Một quan trắc đầu vào của trạm với ràng buộc miền giá trị hợp lệ."""

    timestamp: datetime
    # station giữ lại để tương thích payload cũ; station_id là tên canonical.
    station: str | None = Field(default=None, min_length=1, max_length=100)
    station_id: str | None = Field(default=None, min_length=1, max_length=100)
    PM25: float = Field(alias="PM2.5", ge=0, le=1000)
    TSP: float | None = Field(default=None, ge=0)
    NO2: float | None = Field(default=None, ge=0)
    SO2: float | None = Field(default=None, ge=0)
    CO: float | None = Field(default=None, ge=0)
    O3: float | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=None, ge=-20, le=60)
    humidity: float | None = Field(default=None, ge=0, le=100)

    model_config = {"populate_by_name": True}

    def model_post_init(self, __context: object) -> None:
        if not self.station and not self.station_id:
            raise ValueError("Phải cung cấp station_id (hoặc station ở API legacy).")


class PredictionRequest(BaseModel):
    """Chuỗi quan trắc dùng để tạo lag và dự báo (tối thiểu 25, tối đa 168 giờ)."""

    observations: list[Observation] = Field(min_length=25, max_length=168)


class Interval(BaseModel):
    """Khoảng dự báo Conformal Interval."""

    method: str = "split_conformal_prediction_interval"
    coverage_target: float = Field(default=0.9, ge=0.0, le=1.0)
    coverage: float = Field(default=0.9, ge=0.0, le=1.0)
    lower: float
    upper: float
    width: float | None = None


class PredictionResponse(BaseModel):
    """Cấu trúc phản hồi chuẩn hóa của endpoint dự báo."""

    station: str
    station_id: str | None = None
    forecast_origin: str
    forecast_for: str
    current_pm25: float
    predicted_pm25: float
    level: str
    forecast_strategy: str = "ml_model"
    serving_champion: str | None = None
    is_out_of_distribution: bool = False
    interval: Interval
    model_version: str
    updated_at: str
    production_readiness: str = "unknown"
    calibration_gate: str = "unknown"


class ErrorResponse(BaseModel):
    """Cấu trúc phản hồi lỗi chuẩn hóa."""

    code: str
    message: str


def _resolve_artifact_root() -> Path:
    """Return the directory that holds the versioned artifact bundle.

    Resolution order:
    1. ``PM25_ARTIFACT_DIR`` environment variable;
    2. ``./artifacts`` (default working-tree location).

    The active version inside that directory is selected by
    ``Predictor.from_artifact`` via ``production.json``.
    """
    env_value = __import__("os").environ.get("PM25_ARTIFACT_DIR")
    if env_value:
        return Path(env_value)
    return DEFAULT_ARTIFACT_ROOT


@lru_cache
def get_predictor() -> Predictor:
    """Nạp predictor một lần và tái sử dụng giữa các request (P0.4).

    The predictor is built from a self-contained artifact bundle. The
    legacy ``Predictor(load_config(...))`` path is intentionally NOT used
    here because it would couple the API to the working tree's
    ``configs/config.yaml`` and reintroduce the training-serving skew that
    P0.4 is meant to eliminate.
    """
    artifact_root = _resolve_artifact_root()
    return Predictor.from_artifact(artifact_root)


app = FastAPI(
    title="API dự báo PM2.5 TP.HCM",
    version="1.1.0",
    description="Hệ thống dự báo nồng độ PM2.5 giờ tiếp theo không rò rỉ dữ liệu.",
)


@app.get("/health")
def health():
    """Kiểm tra mô hình đã được nạp và sẵn sàng phục vụ."""
    try:
        predictor = get_predictor()
        if predictor.model is None:
            raise ValueError("Model is None")
        metadata = predictor.metadata or {}
        return {
            "status": "ready",
            "model_loaded": True,
            "model_version": metadata.get("model_version"),
            "production_readiness": metadata.get("production_readiness", "unknown"),
            "calibration_gate": metadata.get("calibration_gate", "unknown"),
        }
    except Exception as err:
        raise HTTPException(
            status_code=503,
            detail="Mô hình chưa sẵn sàng.",
        ) from err


@app.post("/predict", response_model=PredictionResponse)
@app.post("/v1/predict/raw", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Endpoint debug nhận raw history; API chính nằm ở /v1/stations/{station_id}/forecast."""
    try:
        records = [item.model_dump(by_alias=True) for item in request.observations]
        # Chuẩn hóa alias theo config của artifact, tránh station/station_id
        # bị lệch giữa payload và model bundle.
        predictor = get_predictor()
        predictor_config = getattr(predictor, "config", {}) or {}
        station_column = predictor_config.get("data", {}).get("station_column", "station")
        for rec in records:
            station_value = rec.get("station_id") or rec.get("station")
            rec[station_column] = station_value
            rec["station_id"] = station_value
        # Chuyển timestamp datetime thành chuỗi ISO để pandas parser nhất quán
        for rec in records:
            if isinstance(rec.get("timestamp"), datetime):
                rec["timestamp"] = rec["timestamp"].isoformat()
        return predictor.predict(pd.DataFrame(records))
    except ValueError as error:
        return JSONResponse(
            status_code=400,
            content={"code": "INVALID_INPUT", "message": str(error)},
        )
    except FileNotFoundError:
        return JSONResponse(
            status_code=503,
            content={"code": "MODEL_UNAVAILABLE", "message": "Mô hình hoặc artifact chưa sẵn sàng."},
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"code": "INTERNAL_ERROR", "message": "Lỗi nội bộ hệ thống."},
        )


@app.get("/v1/stations/{station_id}/forecast", response_model=PredictionResponse)
def forecast_station(station_id: str):
    """API chính: backend tự lấy history gần nhất thay vì bắt client upload 25 dòng."""
    try:
        predictor = get_predictor()
        config = predictor.config
        frame = load_air_quality(config)
        station_column = config["data"]["station_column"]
        history = frame[frame[station_column].astype(str) == station_id].tail(168)
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
