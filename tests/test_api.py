import pandas as pd
from fastapi.testclient import TestClient

from app import api


def test_api_returns_503_when_model_missing(monkeypatch):
    class FakePredictorMissing:
        model = None

    monkeypatch.setattr(api, "get_predictor", lambda: FakePredictorMissing())
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 503
    assert "Mô hình chưa sẵn sàng" in response.json()["detail"]


def test_api_rejects_oversized_payload():
    client = TestClient(api.app)
    dates = pd.date_range("2024-01-01", periods=200, freq="h").astype(str)
    records = [{"timestamp": dates[i], "station": "A", "PM2.5": 10.0} for i in range(200)]
    response = client.post("/predict", json={"observations": records})
    assert response.status_code == 422


def test_api_prediction_response_schema(monkeypatch):
    expected = {
        "station": "Trạm A",
        "forecast_origin": "2024-01-01T10:00:00",
        "forecast_for": "2024-01-01T11:00:00",
        "current_pm25": 32.1,
        "predicted_pm25": 34.7,
        "level": "Trung bình",
        "forecast_strategy": "ml_model",
        "serving_champion": "random_forest",
        "interval": {
            "method": "split_conformal",
            "coverage_target": 0.9,
            "coverage": 0.9,
            "lower": 29.2,
            "upper": 40.1,
            "width": 10.9,
        },
        "model_version": "2026-09-01-001",
        "updated_at": "2024-01-01T10:00:00Z",
    }

    class FakePredictor:
        def predict(self, observations):
            return expected

    monkeypatch.setattr(api, "get_predictor", lambda: FakePredictor())
    client = TestClient(api.app)

    dates = pd.date_range("2024-01-01", periods=25, freq="h").astype(str)
    payload = [{"timestamp": dates[i], "station": "Trạm A", "PM2.5": 30.0} for i in range(25)]
    response = client.post("/predict", json={"observations": payload})
    assert response.status_code == 200, response.json()
    res = response.json()
    assert res["station"] == "Trạm A"
    assert res["predicted_pm25"] == 34.7
    assert res["forecast_strategy"] == "ml_model"
    assert res["interval"]["coverage"] == 0.9
    assert res["interval"]["method"] == "split_conformal"


def test_predict_returns_503_when_file_not_found(monkeypatch):
    def fake_get_predictor():
        raise FileNotFoundError("Model not found")

    monkeypatch.setattr(api, "get_predictor", fake_get_predictor)
    client = TestClient(api.app)
    dates = pd.date_range("2024-01-01", periods=25, freq="h").astype(str)
    payload = [{"timestamp": dates[i], "station": "Trạm A", "PM2.5": 30.0} for i in range(25)]
    response = client.post("/predict", json={"observations": payload})
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"

