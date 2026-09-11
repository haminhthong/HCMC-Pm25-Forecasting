import pandas as pd
import pytest

from src.data import audit_air_quality, validate_config, validate_schema
from src.data.loader import load_air_quality

CONFIG = {
    "project": {"random_state": 42},
    "data": {
        "station_column": "station",
        "timestamp_column": "timestamp",
        "target_column": "PM2.5",
    },
    "features": {"lags": [1], "rolling_windows": [2], "exogenous_columns": ["O3", "SO2"]},
    "split": {"test_fraction": 0.2, "backtest_folds": 2, "minimum_train_periods": 24},
    "model_comparison": {"candidates": ["ridge"]},
    "models": {"ridge": {}},
    "thresholds": {"low_max": 12.0, "medium_max": 35.5},
    "artifacts": {"directory": "artifacts"},
}


def test_schema_csv_missing_column():
    with pytest.raises(ValueError, match="station"):
        validate_schema(
            pd.DataFrame({"timestamp": [], "PM2.5": []}), ["timestamp", "station", "PM2.5"]
        )


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


def test_loader_rejects_duplicate_station_timestamps(tmp_path):
    path = tmp_path / "duplicate.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00", "2024-01-01 00:00"],
            "station_id": ["A", "A"],
            "PM2.5": [10.0, 11.0],
        }
    ).to_csv(path, index=False)
    config = {"data": {
        "path": str(path),
        "timestamp_column": "timestamp",
        "station_column": "station_id",
        "target_column": "PM2.5",
        "required_columns": ["timestamp", "station_id", "PM2.5"],
    }}
    with pytest.raises(ValueError, match="trùng"):
        load_air_quality(config)


def test_config_rejects_invalid_test_fraction():
    invalid_config = CONFIG.copy()
    invalid_config["split"] = {
        "test_fraction": 1.5,
        "backtest_folds": 2,
        "minimum_train_periods": 24,
    }
    with pytest.raises(ValueError, match="test_fraction"):
        validate_config(invalid_config)


def test_config_rejects_model_outside_project_scope():
    invalid_config = CONFIG | {
        "model_comparison": {"candidates": ["unknown_model"]},
        "models": {"unknown_model": {}},
    }
    with pytest.raises(ValueError, match="chưa được hỗ trợ"):
        validate_config(invalid_config)
