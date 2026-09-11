"""Chia dữ liệu theo thời gian và tạo tóm tắt split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def split_by_time(
    frame: pd.DataFrame,
    test_fraction: float | None = None,
    calibration_fraction: float = 0.0,
    timestamp_column: str = "timestamp",
    *,
    train_end: str | pd.Timestamp | None = None,
    calibration_end: str | pd.Timestamp | None = None,
    test_end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia dữ liệu theo lịch cố định hoặc tỷ lệ dự phòng.

    Khi có ``train_end``, ``calibration_end`` và ``test_end``, các mốc này được
    ưu tiên để ranh giới không dịch chuyển khi dữ liệu mới được append.
    """
    explicit_boundaries = [train_end, calibration_end, test_end]
    if any(value is not None for value in explicit_boundaries):
        if not all(value is not None for value in explicit_boundaries):
            raise ValueError("Phải cung cấp đủ train_end, calibration_end và test_end.")

        boundaries = [pd.Timestamp(value) for value in explicit_boundaries]
        frame_tz = getattr(frame[timestamp_column].dtype, "tz", None)
        normalized_boundaries = []
        for boundary in boundaries:
            if boundary.tzinfo is None and frame_tz is not None:
                boundary = boundary.tz_localize(frame_tz)
            elif boundary.tzinfo is not None and frame_tz is not None:
                boundary = boundary.tz_convert(frame_tz)
            normalized_boundaries.append(boundary)
        train_end_ts, calibration_end_ts, test_end_ts = normalized_boundaries
        if not train_end_ts < calibration_end_ts < test_end_ts:
            raise ValueError("Các mốc split calendar phải tăng dần theo thời gian.")

        target_column = "target_timestamp" if "target_timestamp" in frame.columns else timestamp_column
        train_frame = frame[frame[target_column] < train_end_ts].copy()
        cal_mask = (frame[timestamp_column] >= train_end_ts) & (
            frame[timestamp_column] < calibration_end_ts
        )
        cal_mask &= frame[target_column] < calibration_end_ts
        cal_frame = frame[cal_mask].copy()
        test_mask = (frame[timestamp_column] >= calibration_end_ts) & (
            frame[timestamp_column] < test_end_ts
        )
        test_mask &= frame[target_column] < test_end_ts
        test_frame = frame[test_mask].copy()
        return train_frame, cal_frame, test_frame

    if test_fraction is None:
        raise ValueError("Cần test_fraction hoặc bộ mốc split calendar.")

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
    """Tạo tóm tắt ranh giới thời gian, trạm và số lượng dòng."""
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
