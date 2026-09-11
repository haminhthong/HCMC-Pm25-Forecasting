"""Bộ kiểm thử hợp đồng chống rò rỉ và quy trình đánh giá dự báo.

Bao gồm 12 ca kiểm thử bắt buộc:
1. test_exact_hour_lag_does_not_shift_over_gap
2. test_rolling_feature_excludes_current_observation
3. test_target_timestamp_never_crosses_split_boundary
4. test_backtest_train_before_validation
5. test_calibration_before_test
6. test_test_never_used_for_model_selection
7. test_seasonal_naive_same_hour_previous_day
8. test_persistence_baseline
9. test_model_selection_fallback_to_persistence
10. test_selected_strategy_test_metrics_match_policy
11. test_conformal_interval_uses_correct_strategy_residuals
12. test_train_and_inference_feature_columns_match
"""

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, model_feature_columns
from src.forecasting.baselines import persistence_predictions, seasonal_naive_predictions
from src.forecasting.selection import select_forecast_strategy
from src.forecasting.trainer import make_pipeline
from src.pipeline import run_train_pipeline
from src.validation.backtest import expanding_time_folds
from src.validation.split import split_by_time

CONFIG = {
    "project": {"random_state": 42},
    "data": {
        "station_column": "station",
        "timestamp_column": "timestamp",
        "target_column": "PM2.5",
        "required_columns": ["timestamp", "station", "PM2.5"],
        "optional_columns": ["O3", "SO2"],
        "path": "data/sample/air_quality_sample.csv",
    },
    "features": {
        "lags": [1, 2, 3, 6, 12, 24],
        "rolling_windows": [3, 6, 24],
        "delta_lags": [1, 3],
        "include_rolling_std": True,
        "exogenous_columns": ["O3", "SO2"],
    },
    "split": {
        "test_fraction": 0.1,
        "calibration_fraction": 0.1,
        "backtest_folds": 3,
        "minimum_train_periods": 12,
    },
    "model_comparison": {"candidates": ["ridge", "hist_gradient_boosting"]},
    "thresholds": {
        "low_max": 12.0,
        "medium_max": 35.5,
        "labels": ["Thấp", "Trung bình", "Cao"],
    },
    "model_selection": {
        "minimum_mae_improvement": 0.05,
        "maximum_cv_mae_std": 1.0,
    },
    "artifacts": {
        "directory": "artifacts",
        "model_file": "model.joblib",
        "metadata_file": "metadata.json",
        "evaluation_file": "evaluation.json",
    },
}


def test_exact_hour_lag_does_not_shift_over_gap():
    """1. Kiểm tra lag theo mốc thời gian thực: khi có khoảng trống thời gian, lag 1h là NaN thay vì lấy dòng trước."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"]),
            "station": ["Trạm A", "Trạm A"],
            "PM2.5": [15.0, 30.0],
            "O3": [1.0, 1.0],
            "SO2": [1.0, 1.0],
        }
    )
    featured = build_features(df, CONFIG)
    assert pd.isna(featured.iloc[1]["PM2.5_lag_1"]), (
        "Khoảng trống 2 giờ nhưng lag 1h không phải là NaN (bị lỗi row shift)!"
    )


def test_rolling_feature_excludes_current_observation():
    """2. Kiểm tra rolling window closed='left': không tính quan trắc hiện tại t vào lịch sử rolling."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 00:00", periods=5, freq="h"),
            "station": ["Trạm A"] * 5,
            "PM2.5": [10.0, 20.0, 30.0, 40.0, 50.0],
            "O3": [1.0] * 5,
            "SO2": [1.0] * 5,
        }
    )
    featured = build_features(df, CONFIG)
    # Tại t = 01:00 (PM2.5 = 20), rolling mean 3h với closed='left' chỉ tính giá trị trước đó (10.0)
    assert featured.iloc[1]["PM2.5_rolling_mean_3"] == 10.0
    # Tại t = 02:00 (PM2.5 = 30), rolling mean chỉ tính 10.0 và 20.0 -> trung bình là 15.0 (không gồm 30.0)
    assert featured.iloc[2]["PM2.5_rolling_mean_3"] == 15.0


def test_target_timestamp_never_crosses_split_boundary():
    """3. Chống rò rỉ biên split: target_timestamp (t+1) của train không bao giờ chạm vào calibration hay test."""
    periods = pd.date_range("2024-01-01", periods=40, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": np.repeat(periods, 2),
            "station": ["Trạm A", "Trạm B"] * 40,
            "target_next_hour": [20.0] * 80,
        }
    )
    df["target_timestamp"] = df["timestamp"] + pd.to_timedelta(1, unit="h")

    train_df, cal_df, test_df = split_by_time(
        df, test_fraction=0.1, calibration_fraction=0.1, timestamp_column="timestamp"
    )

    assert train_df["target_timestamp"].max() < cal_df["timestamp"].min()
    assert cal_df["target_timestamp"].max() < test_df["timestamp"].min()


def test_backtest_train_before_validation():
    """4. Kiểm tra expanding-window folds trong tập train tuân thủ quan hệ nhân quả nghiêm ngặt."""
    periods = pd.date_range("2024-01-01", periods=60, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": np.repeat(periods, 2),
            "station": ["Trạm A", "Trạm B"] * 60,
        }
    )
    df["target_timestamp"] = df["timestamp"] + pd.to_timedelta(1, unit="h")

    folds = expanding_time_folds(df, "timestamp", folds=3, minimum_train_periods=20)
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        train_targets = df.iloc[train_idx]["target_timestamp"]
        val_starts = df.iloc[val_idx]["timestamp"]
        assert train_targets.max() < val_starts.min(), (
            "target_timestamp của train trong fold vi phạm biên validation!"
        )


def test_calibration_before_test():
    """5. Kiểm tra tập Calibration nằm hoàn toàn trước tập Test cuối."""
    periods = pd.date_range("2024-01-01", periods=30, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": periods,
            "station": ["Trạm A"] * 30,
        }
    )
    train_df, cal_df, test_df = split_by_time(
        df, test_fraction=0.2, calibration_fraction=0.2, timestamp_column="timestamp"
    )
    assert cal_df["timestamp"].max() < test_df["timestamp"].min()


def test_test_never_used_for_model_selection():
    """6. Xác nhận tập Test hoàn toàn độc lập và không được sử dụng để chọn candidate model."""
    result = run_train_pipeline("configs/config.yaml", persist_artifacts=False)
    # Lựa chọn candidate model dựa trên backtest trên tập train
    backtest = result["evaluation"]["backtest"]
    selected_name = min(backtest, key=lambda n: backtest[n]["mae_mean"])
    assert result["best_cv_model"] == selected_name


def test_seasonal_naive_same_hour_previous_day():
    r"""7. Kiểm định công thức Seasonal Naive 24h: \hat{y}_{t+1} = y_{t-23} (cùng giờ ngày hôm trước)."""
    dates = pd.date_range("2024-01-01 00:00", periods=48, freq="h")
    values = np.arange(48, dtype=float)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "station": ["Trạm A"] * 48,
            "PM2.5": values,
            "O3": [1.0] * 48,
            "SO2": [1.0] * 48,
        }
    )
    featured = build_features(df, CONFIG)
    # Tại mốc t = 2024-01-02 00:00 (index 24), dự báo t+1h (01:00 ngày 2/1)
    # bằng cùng giờ ngày hôm trước (01:00 ngày 1/1, index 1)
    # t - 23h = 24 - 23 = index 1 -> giá trị phải là 1.0!
    assert featured.iloc[24]["seasonal_naive_24h"] == 1.0
    preds = seasonal_naive_predictions(featured, "PM2.5")
    assert preds[24] == 1.0


def test_persistence_baseline():
    r"""8. Kiểm định công thức Persistence: \hat{y}_{t+1} = y_t."""

    df = pd.DataFrame(
        {
            "station": ["Trạm A"] * 3,
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "PM2.5": [12.5, 18.2, 22.0],
        }
    )
    preds = persistence_predictions(df, "PM2.5")
    np.testing.assert_allclose(preds, [12.5, 18.2, 22.0])


def test_model_selection_fallback_to_persistence():
    """9. Model không vượt Persistence thì chiến lược được chọn là Persistence."""
    cfg = {
        "model_selection": {
            "minimum_mae_improvement": 0.05,
            "maximum_cv_mae_std": 1.0,
        }
    }
    selection = select_forecast_strategy(
        "ridge",
        model_metrics={"mae": 10.0},
        persistence_metrics={"mae": 10.0},
        cv_mae_std=0.2,
        config=cfg,
    )
    assert selection["forecast_strategy"] == "persistence"
    assert selection["status"] == "persistence"


def test_selected_strategy_test_metrics_match_policy():
    """10. Kết quả selected_strategy_test phải khớp chiến lược đã chọn."""
    result = run_train_pipeline("configs/config.yaml", persist_artifacts=False)
    evaluation = result["evaluation"]
    strategy = result["forecast_strategy"]

    if strategy == "persistence":
        assert evaluation["selected_strategy_test"]["mae"] == pytest.approx(
            evaluation["persistence_test"]["mae"]
        )
    else:
        assert evaluation["selected_strategy_test"]["mae"] == pytest.approx(
            evaluation["model_test"]["mae"]
        )


def test_conformal_interval_uses_correct_strategy_residuals():
    """11. Interval dùng đúng residual quantile của chiến lược được chọn."""
    result = run_train_pipeline("configs/config.yaml", persist_artifacts=False)
    interval_info = result["prediction_interval"]
    strategy = result["forecast_strategy"]
    cal = result["evaluation"]["calibration"]

    if strategy == "persistence":
        assert interval_info["residual_quantile"] == pytest.approx(
            round(cal["persistence_residual_quantile"], 4)
        )
    else:
        assert interval_info["residual_quantile"] == pytest.approx(
            round(cal["model_residual_quantile"], 4)
        )


def test_train_and_inference_feature_columns_match():
    """12. Đảm bảo cột đặc trưng khớp giữa huấn luyện và lớp dự báo."""
    _, train_cols = make_pipeline(CONFIG, "hist_gradient_boosting")
    expected_cols = [*model_feature_columns(CONFIG), CONFIG["data"]["station_column"]]
    assert train_cols == expected_cols
