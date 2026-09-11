import numpy as np
import pandas as pd
import pytest

from src.inference.predictor import Predictor


@pytest.fixture
def sample_config():
    return {
        "data": {
            "station_column": "station",
            "timestamp_column": "timestamp",
            "target_column": "PM2.5",
        },
        "features": {
            "lags": [1, 2, 3, 6, 12, 24],
            "rolling_windows": [3, 6, 24],
            "exogenous_columns": ["O3", "SO2"],
        },
        "thresholds": {
            "low_max": 12.0,
            "medium_max": 35.5,
            "labels": ["Thấp", "Trung bình", "Cao"],
        },
        "artifacts": {"directory": "artifacts", "model_file": "model.joblib"},
    }


def test_prediction_rejects_multiple_stations(sample_config, monkeypatch):
    class FakeModel:
        def predict(self, df):
            return np.array([20.0])

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)

    records = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=25, freq="h"),
            "station": ["A"] * 24 + ["B"],
            "PM2.5": [10.0] * 25,
        }
    )
    with pytest.raises(ValueError, match="một trạm duy nhất"):
        predictor.predict(records)


def test_prediction_rejects_insufficient_history(sample_config, monkeypatch):
    class FakeModel:
        def predict(self, df):
            return np.array([20.0])

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)

    records = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "station": ["A"] * 10,
            "PM2.5": [10.0] * 10,
        }
    )
    with pytest.raises(ValueError, match="Cần tối thiểu 25 quan trắc"):
        predictor.predict(records)


def test_prediction_rejects_duplicate_timestamps(sample_config, monkeypatch):
    class FakeModel:
        def predict(self, df):
            return np.array([20.0])

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)

    dates = list(pd.date_range("2024-01-01", periods=24, freq="h")) + [
        pd.Timestamp("2024-01-01 00:00")
    ]
    records = pd.DataFrame(
        {
            "timestamp": dates,
            "station": ["A"] * 25,
            "PM2.5": [10.0] * 25,
        }
    )
    with pytest.raises(ValueError, match="timestamp trùng lặp"):
        predictor.predict(records)


def test_prediction_rejects_negative_pm25(sample_config, monkeypatch):
    class FakeModel:
        def predict(self, df):
            return np.array([20.0])

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)

    records = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=25, freq="h"),
            "station": ["A"] * 25,
            "PM2.5": [10.0] * 24 + [-5.0],
        }
    )
    with pytest.raises(ValueError, match="không được âm"):
        predictor.predict(records)


def test_missing_o3_so2_uses_persistence_fallback(sample_config, monkeypatch):
    """Thiếu toàn bộ exogenous thì vẫn dự báo được bằng Persistence."""

    class FakeModel:
        def predict(self, df):
            return np.array([25.4])

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)
    predictor.metadata["forecast_strategy"] = "fake_model"

    records = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=25, freq="h"),
            "station": ["A"] * 25,
            "PM2.5": [15.0] * 25,
            "O3": [None] * 25,
            "SO2": [None] * 25,
        }
    )
    res = predictor.predict(records)
    assert res["predicted_pm25"] == 15.0
    assert np.isfinite(res["predicted_pm25"])
    assert res["level"] in ["Thấp", "Trung bình", "Cao"]
    assert res["forecast_strategy"] == "persistence"


def test_persistence_strategy_fallback(sample_config, monkeypatch):
    """Kiểm tra chiến lược Persistence không gọi model ML."""

    class FakeModel:
        def predict(self, df):
            return np.array([999.0])  # Giá trị bất kỳ không được gọi

    monkeypatch.setattr("joblib.load", lambda path: FakeModel())
    predictor = Predictor(sample_config)
    predictor.metadata = {
        "forecast_strategy": "persistence",
        "prediction_interval": {
            "method": "split_conformal",
            "coverage": 0.95,
            "residual_quantile": 3.5,
            "ml_residual_quantile": 10.0,
            "persistence_residual_quantile": 3.5,
        },
    }

    records = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=25, freq="h"),
            "station": ["A"] * 25,
            "PM2.5": [12.0] * 24 + [18.5],
        }
    )
    res = predictor.predict(records)
    assert res["predicted_pm25"] == 18.5  # Lấy quan trắc mới nhất thay vì 999.0 của FakeModel
    assert res["interval"]["coverage"] == 0.95
    assert res["interval"]["lower"] == 15.0  # 18.5 - 3.5
    assert res["interval"]["upper"] == 22.0  # 18.5 + 3.5
