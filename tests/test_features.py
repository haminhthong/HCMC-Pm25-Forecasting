import pandas as pd

from src.features import build_features

CONFIG = {
    "data": {
        "station_column": "station",
        "timestamp_column": "timestamp",
        "target_column": "PM2.5",
    },
    "features": {"lags": [1], "rolling_windows": [2], "exogenous_columns": ["O3", "SO2"]},
}


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
