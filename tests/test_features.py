import pandas as pd
import pytest

from src.data import audit_air_quality, validate_config, validate_schema
from src.features import build_features

CONFIG = {
    "data": {
        "station_column": "station",
        "timestamp_column": "timestamp",
        "target_column": "PM2.5",
    },
    "features": {"lags": [1], "rolling_windows": [2], "exogenous_columns": ["O3", "SO2"]},
}


def test_schema_csv():
    with pytest.raises(ValueError, match="station"):
        validate_schema(
            pd.DataFrame({"timestamp": [], "PM2.5": []}), ["timestamp", "station", "PM2.5"]
        )


def test_sort_and_lag_do_not_use_future():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 02:00", "2024-01-01 00:00", "2024-01-01 01:00"]
            ),
            "station": ["A", "A", "A"],
            "PM2.5": [30.0, 10.0, 20.0],
            "O3": [1, 1, 1],
            "SO2": [1, 1, 1],
        }
    )
    result = build_features(frame, CONFIG)
    assert result["timestamp"].is_monotonic_increasing
    assert pd.isna(result.iloc[0]["PM2.5_lag_1"])
    assert result.iloc[1]["PM2.5_lag_1"] == 10.0


def test_data_audit_detects_duplicates_and_gaps():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 03:00"]
            ),
            "station": ["A", "A", "A"],
            "PM2.5": [10.0, 10.0, 12.0],
        }
    )
    report = audit_air_quality(frame, CONFIG)
    assert report["duplicate_station_timestamps"] == 2
    assert report["irregular_hourly_gaps"] >= 1


def test_config_rejects_invalid_test_fraction():
    config = {
        "project": {},
        "data": {},
        "features": {"lags": [1], "rolling_windows": [3]},
        "split": {"test_fraction": 1.0, "backtest_folds": 3, "minimum_train_periods": 24},
        "model_comparison": {"candidates": ["ridge"]},
        "models": {"ridge": {}},
        "thresholds": {"good_max": 12, "moderate_max": 35.5},
        "artifacts": {},
    }
    with pytest.raises(ValueError, match="test_fraction"):
        validate_config(config)


def test_lag_uses_exact_hour_instead_of_previous_row():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"]),
            "station": ["A", "A"],
            "PM2.5": [10.0, 30.0],
            "O3": [1.0, 1.0],
            "SO2": [1.0, 1.0],
        }
    )
    result = build_features(frame, CONFIG)
    assert pd.isna(result.iloc[1]["PM2.5_lag_1"])
    assert pd.isna(result.iloc[0]["target_next_hour"])
