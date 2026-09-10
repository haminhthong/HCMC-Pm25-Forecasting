import numpy as np
import pandas as pd

from src.forecasting.selection import select_forecast_strategy
from src.validation.backtest import expanding_time_folds
from src.validation.split import split_by_time


def test_expanding_folds_never_train_on_future():
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(pd.date_range("2024-01-01", periods=50, freq="h"), 2),
            "station": ["A", "B"] * 50,
        }
    )
    folds = expanding_time_folds(frame, "timestamp", folds=4, minimum_train_periods=40)
    assert len(folds) == 4
    for train_indices, validation_indices in folds:
        train_periods = frame.iloc[train_indices]["timestamp"]
        validation_periods = frame.iloc[validation_indices]["timestamp"]
        assert train_periods.max() < validation_periods.min()
        assert set(train_periods).isdisjoint(set(validation_periods))
        assert len(set(train_indices) & set(validation_indices)) == 0


def test_expanding_folds_keep_target_inside_validation_window():
    timestamps = pd.date_range("2024-01-01", periods=12, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "target_timestamp": timestamps + pd.Timedelta(hours=1),
        }
    )
    _, validation_indices = expanding_time_folds(
        frame,
        "timestamp",
        folds=2,
        minimum_train_periods=6,
    )[0]
    validation = frame.iloc[validation_indices]
    assert (validation["target_timestamp"] <= timestamps[8]).all()


def test_time_split_preserves_order():
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(pd.date_range("2024-01-01", periods=10, freq="h"), 2),
            "station": ["A", "B"] * 10,
        }
    )
    train_frame, cal_frame, test_frame = split_by_time(
        frame, test_fraction=0.2, calibration_fraction=0.2
    )
    assert train_frame["timestamp"].max() < cal_frame["timestamp"].min()
    assert cal_frame["timestamp"].max() < test_frame["timestamp"].min()
    assert len(test_frame) == 4
    assert test_frame.groupby("timestamp")["station"].nunique().eq(2).all()


def test_model_selection_requires_model_to_beat_persistence():
    cfg = {
        "model_selection": {
            "minimum_mae_improvement": 0.05,
            "maximum_cv_mae_std": 1.0,
        }
    }
    passed = select_forecast_strategy("ridge", {"mae": 1.0}, {"mae": 2.0}, 0.5, cfg)
    failed = select_forecast_strategy("ridge", {"mae": 1.98}, {"mae": 2.0}, 0.5, cfg)
    assert passed["forecast_strategy"] == "ridge"
    assert failed["forecast_strategy"] == "persistence"
