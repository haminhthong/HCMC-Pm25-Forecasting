"""Kiểm thử hợp đồng dữ liệu, chống rò rỉ và dự báo."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration.conformal import conformal_quantile
from src.data.regularization import regularize_hourly_series
from src.evaluation.metrics import regression_and_classification_metrics
from src.features.lag import lookup_pm25_at_offset
from src.forecasting.models import build_model
from src.forecasting.trainer import make_pipeline, resolve_candidate_params
from src.inference.predictor import Predictor


class _MockModel:
    def predict(self, _x):
        return np.array([25.0])


def test_finite_sample_conformal_quantile_formula():
    """Kiểm tra finite-sample split-conformal quantile: rank = min(n, ceil((n+1)*coverage))."""
    residuals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # n = 10, coverage = 0.90 -> ceil((10+1)*0.90) = ceil(9.9) = 10 -> giá trị thứ 10 là 10.0
    q = conformal_quantile(residuals, coverage=0.90)
    assert q == 10.0

    # n = 4, coverage = 0.50 -> ceil((4+1)*0.50) = ceil(2.5) = 3 -> giá trị thứ 3
    q2 = conformal_quantile([10.0, 20.0, 30.0, 40.0], coverage=0.50)
    assert q2 == 30.0


def test_conformal_requires_calibration_set():
    """Kiểm tra conformal quantile ném lỗi nếu calibration set rỗng (chống fallback sang train/test)."""
    with pytest.raises(ValueError, match="calibration set không rỗng"):
        conformal_quantile([], coverage=0.90)


def test_clock_time_lag_with_missing_hour():
    """Kiểm tra exact clock-time lag trả NaN khi thiếu một mốc giờ (không shift mù quáng)."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 12:00"]),
            "station": ["Trạm A", "Trạm A"],
            "PM2.5": [25.0, 30.0],
        }
    )
    # Tại 12:00, lag 1h (t - 1h = 11:00) không tồn tại -> NaN
    lags = lookup_pm25_at_offset(df, "station", "timestamp", "PM2.5", offset_hours=-1)
    assert pd.isna(lags[1])


def test_training_inference_gap_policy_same():
    """Kiểm tra chính sách gap giống nhau giữa huấn luyện và dự báo."""
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 12:00"]),
            "station": ["Trạm A", "Trạm A"],
            "PM2.5": [20.0, 35.0],
        }
    )
    regularized = regularize_hourly_series(raw, "timestamp", group_columns=["station"])
    assert len(regularized) == 3
    assert regularized.iloc[1]["timestamp"] == pd.Timestamp("2024-01-01 11:00")
    assert pd.isna(regularized.iloc[1]["PM2.5"])


def test_regularization_reports_inserted_hours():
    """Kiểm tra regularization báo đúng số mốc giờ được chèn."""
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 12:00"]),
            "station": ["Trạm A", "Trạm A"],
            "PM2.5": [20.0, 35.0],
        }
    )
    regularized = regularize_hourly_series(raw, "timestamp", group_columns=["station"])

    assert regularized.attrs["regularize_hourly_inserted_rows"] == 1


def test_candidate_params_are_applied_consistently():
    """Kiểm tra hai model đang dùng nhận đúng siêu tham số từ cấu hình."""
    cfg = {
        "models": {
            "ridge": {"alpha": 2.5},
            "hist_gradient_boosting": {"max_iter": 30, "learning_rate": 0.1},
        }
    }
    for model_name in ["ridge", "hist_gradient_boosting"]:
        params = resolve_candidate_params(cfg, model_name)
        model = build_model(model_name, random_state=42, params=params)
        assert model is not None
        if model_name == "ridge":
            assert model.alpha == 2.5
        elif model_name == "hist_gradient_boosting":
            assert model.max_iter == 30
            assert model.learning_rate == 0.1


def test_no_standard_scaler_for_hist_gradient_boosting():
    """Kiểm tra Ridge mới dùng StandardScaler, còn HistGradientBoosting thì không."""
    cfg = {
        "project": {"random_state": 42},
        "data": {
            "station_column": "station",
            "timestamp_column": "timestamp",
            "target_column": "PM2.5",
        },
        "features": {
            "lags": [1],
            "rolling_windows": [3],
            "exogenous_columns": [],
        },
        "models": {
            "ridge": {},
            "hist_gradient_boosting": {},
        },
    }
    ridge_pipe, _ = make_pipeline(cfg, "ridge")
    tree_pipe, _ = make_pipeline(cfg, "hist_gradient_boosting")

    ridge_numeric_steps = dict(ridge_pipe.named_steps["preprocess"].transformers[0][1].steps)
    tree_numeric_steps = dict(tree_pipe.named_steps["preprocess"].transformers[0][1].steps)

    assert "scaler" in ridge_numeric_steps
    assert "scaler" not in tree_numeric_steps


def test_unknown_station_policy():
    """Kiểm tra Predictor nhận biết và gắn cờ is_out_of_distribution khi gặp trạm chưa từng học."""
    import tempfile

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        tmp_path = Path(tmpdir)
        config = {
            "data": {
                "station_column": "station",
                "timestamp_column": "timestamp",
                "target_column": "PM2.5",
            },
            "features": {
                "lags": [1, 2, 3],
                "rolling_windows": [3],
                "exogenous_columns": [],
            },
            "thresholds": {
                "low_max": 12.0,
                "medium_max": 35.5,
                "labels": ["Thấp", "Trung bình", "Cao"],
            },
            "artifacts": {"directory": str(tmp_path), "model_file": "model.joblib"},
        }
        import joblib

        joblib.dump(_MockModel(), tmp_path / "model.joblib")
        (tmp_path / "metadata.json").write_text(
            json.dumps(
                {
                    "model_name": "mock",
                    "trained_stations": ["Trạm A"],
                    "prediction_interval": {"residual_quantile": 4.0},
                }
            ),
            encoding="utf-8",
        )

        predictor = Predictor(config, artifact_dir=tmp_path)
        obs = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=25, freq="h"),
                "station": ["Trạm Lạ Chưa Từng Học"] * 25,
                "PM2.5": [20.0] * 25,
            }
        )
        pred = predictor.predict(obs)
        assert pred["is_out_of_distribution"] is True


def test_classification_metrics_handles_absent_classes():
    """Kiểm tra macro-F1 báo cáo cả 2 góc nhìn (all configured classes & observed classes) cùng support."""
    thresholds = {"low_max": 12.0, "medium_max": 35.5, "labels": ["Thấp", "Trung bình", "Cao"]}
    # Chỉ có mẫu Trung bình (20.0) và Cao (45.0), không có Thấp
    y_true = [20.0, 25.0, 45.0, 50.0]
    y_pred = [20.0, 25.0, 45.0, 50.0]
    metrics = regression_and_classification_metrics(y_true, y_pred, thresholds)

    assert metrics["support_by_class"]["Thấp"] == 0
    assert metrics["support_by_class"]["Trung bình"] == 2
    assert metrics["support_by_class"]["Cao"] == 2
    assert "Thấp" not in metrics["observed_classes"]

    # Trên các class quan sát được, dự đoán hoàn hảo -> 1.0
    assert metrics["macro_f1_observed_classes"] == 1.0
    # Trên cả 3 class cấu hình, do class Thấp không có mẫu -> F1 Thấp = 0 -> (1+1+0)/3 = 0.6667
    assert metrics["macro_f1_all_configured_classes"] == pytest.approx(0.6667, abs=1e-3)
