import numpy as np
import pandas as pd

from src.validation.backtest import expanding_time_folds
from src.validation.split import split_by_time


def test_train_target_never_crosses_test_boundary():
    """Kiểm tra mốc target_timestamp (t+1h) của tập train không vượt mốc bắt đầu của tập test."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
            "station": "Trạm A",
        }
    )
    frame["target_timestamp"] = frame["timestamp"] + pd.to_timedelta(1, unit="h")

    train_frame, cal_frame, test_frame = split_by_time(
        frame, test_fraction=0.2, calibration_fraction=0.1
    )

    assert train_frame["target_timestamp"].max() < cal_frame["timestamp"].min()
    assert cal_frame["target_timestamp"].max() < test_frame["timestamp"].min()


def test_target_does_not_cross_validation_boundary():
    """Kiểm tra mốc target_timestamp của train trong từng fold không đè vào mốc đầu của validation fold."""
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(pd.date_range("2024-01-01", periods=50, freq="h"), 2),
            "station": ["A", "B"] * 50,
        }
    )
    frame["target_timestamp"] = frame["timestamp"] + pd.to_timedelta(1, unit="h")

    folds = expanding_time_folds(frame, "timestamp", folds=4, minimum_train_periods=40)
    assert len(folds) == 4
    for train_indices, validation_indices in folds:
        train_targets = frame.iloc[train_indices]["target_timestamp"]
        val_starts = frame.iloc[validation_indices]["timestamp"]
        assert train_targets.max() < val_starts.min()
