"""Temporal data splitting and split manifest generator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def split_by_time(
    frame: pd.DataFrame,
    test_fraction: float,
    calibration_fraction: float = 0.0,
    timestamp_column: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia theo mốc thời gian sao cho target_timestamp của train nhỏ hơn mốc cal_start và test_start."""
    periods = np.sort(frame[timestamp_column].unique())
    total_periods = len(periods)
    test_period_count = max(1, int(np.ceil(total_periods * test_fraction)))
    cal_period_count = (
        int(np.ceil(total_periods * calibration_fraction)) if calibration_fraction > 0 else 0
    )

    if test_period_count + cal_period_count >= total_periods:
        raise ValueError("Dữ liệu quá ít để tạo các tập kiểm thử và hiệu chuẩn theo thời gian.")

    test_start = periods[-test_period_count]
    if cal_period_count > 0:
        cal_start = periods[-(test_period_count + cal_period_count)]
        if "target_timestamp" in frame.columns:
            train_frame = frame[frame["target_timestamp"] < cal_start].copy()
        else:
            train_frame = frame[frame[timestamp_column] < cal_start].copy()

        cal_mask = (frame[timestamp_column] >= cal_start) & (frame[timestamp_column] < test_start)
        if "target_timestamp" in frame.columns:
            cal_mask = cal_mask & (frame["target_timestamp"] < test_start)
        cal_frame = frame[cal_mask].copy()
    else:
        if "target_timestamp" in frame.columns:
            train_frame = frame[frame["target_timestamp"] < test_start].copy()
        else:
            train_frame = frame[frame[timestamp_column] < test_start].copy()
        cal_frame = pd.DataFrame(columns=frame.columns)

    test_frame = frame[frame[timestamp_column] >= test_start].copy()
    return train_frame, cal_frame, test_frame


def generate_split_manifest(
    train_frame: pd.DataFrame,
    cal_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
    station_column: str = "station",
) -> dict[str, Any]:
    """Tạo split manifest ghi lại ranh giới thời gian, trạm và số lượng dòng (Point 16)."""
    stations = sorted(
        set(train_frame[station_column].unique())
        | set(cal_frame[station_column].unique() if not cal_frame.empty else [])
        | set(test_frame[station_column].unique())
    )

    def _period_range(df: pd.DataFrame) -> dict[str, str | None]:
        if df.empty:
            return {"start": None, "end": None}
        return {
            "start": str(df[timestamp_column].min()),
            "end": str(df[timestamp_column].max()),
        }

    return {
        "train": _period_range(train_frame),
        "calibration": _period_range(cal_frame),
        "test": _period_range(test_frame),
        "stations": [str(s) for s in stations],
        "rows": {
            "train": int(len(train_frame)),
            "calibration": int(len(cal_frame)),
            "test": int(len(test_frame)),
        },
    }
