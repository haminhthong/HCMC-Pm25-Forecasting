import pandas as pd

from src.data.loader import load_air_quality
from src.data.regularization import audit_hourly_gaps
from src.data.schema import normalize_timestamp_series
from src.features.exogenous import lookup_feature_at_offset
from src.inference.input_validation import check_forecast_input


def test_exogenous_latency_is_exact_and_station_aware():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 01:00", "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 00:00"]
            ),
            "station": ["A", "A", "B", "B"],
            "PM2.5": [11.0, 10.0, 21.0, 20.0],
            "NO2": [101.0, 100.0, 201.0, 200.0],
        }
    )
    values = lookup_feature_at_offset(frame, "station", "timestamp", "NO2", -1)
    result = frame.assign(no2_lag=values).sort_values(["station", "timestamp"])
    assert pd.isna(result.iloc[0]["no2_lag"])
    assert result.iloc[1]["no2_lag"] == 100.0
    assert pd.isna(result.iloc[2]["no2_lag"])
    assert result.iloc[3]["no2_lag"] == 200.0


def test_available_at_blocks_future_telemetry():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
            "station": ["A", "A"],
            "PM2.5": [10.0, 11.0],
            "NO2": [100.0, 101.0],
            "available_at": pd.to_datetime(["2024-01-01 00:30", "2024-01-01 01:00"]),
        }
    )
    values = lookup_feature_at_offset(frame, "station", "timestamp", "NO2", 0)
    assert pd.isna(values.iloc[0])
    assert values.iloc[1] == 101.0


def test_input_validation_marks_large_gap_for_persistence():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 10:00"]
            ),
            "station": ["A"] * 3,
            "PM2.5": [10.0, 11.0, 12.0],
        }
    )
    audit = check_forecast_input(
        frame,
        timestamp_column="timestamp",
        target_column="PM2.5",
        required_history_hours=3,
        allowed_gap_hours=6,
    )
    assert audit["status"] == "warning"
    assert audit["use_persistence"] is True


def test_naive_timestamp_is_stored_as_utc():
    values = normalize_timestamp_series(pd.Series(["2024-01-01 07:00"]))
    assert str(values.iloc[0]) == "2024-01-01 00:00:00+00:00"


def test_csv_loader_normalizes_available_at_to_utc(tmp_path):
    csv_path = tmp_path / "observations.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01 07:00"],
            "station_id": ["A"],
            "PM2.5": [10.0],
            "available_at": ["2024-01-01 07:30"],
        }
    ).to_csv(csv_path, index=False)
    config = {
        "data": {
            "path": str(csv_path),
            "timestamp_column": "timestamp",
            "station_column": "station_id",
            "target_column": "PM2.5",
            "required_columns": ["timestamp", "station_id", "PM2.5"],
            "source_timezone": "Asia/Ho_Chi_Minh",
            "zero_as_missing": False,
        }
    }

    loaded = load_air_quality(config)

    assert str(loaded["timestamp"].iloc[0]) == "2024-01-01 00:00:00+00:00"
    assert str(loaded["available_at"].iloc[0]) == "2024-01-01 00:30:00+00:00"


def test_gap_audit_sorts_out_of_order_events_before_counting():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 02:00", "2024-01-01 00:00", "2024-01-01 01:00"]
            ),
            "station_id": ["A", "A", "A"],
        }
    )

    report = audit_hourly_gaps(frame, "timestamp", group_columns=["station_id"])

    assert report["irregular_hourly_gaps"] == 0
