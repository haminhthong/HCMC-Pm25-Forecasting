"""Quality gate cho dữ liệu tại thời điểm serving."""

from __future__ import annotations

from typing import Any

import pandas as pd


def audit_runtime_history(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    target_column: str,
    required_history_hours: int,
    allowed_gap_hours: int,
    exogenous_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Đánh giá history hiện tại trước khi chọn ML hay persistence.

    ``GOOD`` cho phép dùng model. ``DEGRADED`` vẫn có thể dự báo nhưng kèm
    cảnh báo; nếu khoảng trống vượt policy thì predictor sẽ dùng persistence.
    ``UNUSABLE`` không có đủ dữ liệu hoặc thiếu PM2.5 hiện tại.
    """
    exogenous_columns = exogenous_columns or []
    if frame.empty:
        return {
            "status": "UNUSABLE",
            "history_completeness": 0.0,
            "missing_lag_count": required_history_hours,
            "missing_exogenous_count": 0,
            "largest_gap_hours": None,
            "fallback_required": True,
            "warnings": ["history_empty"],
        }

    ordered = frame.sort_values(timestamp_column, kind="stable")
    timestamps = pd.to_datetime(ordered[timestamp_column], errors="coerce")
    target = pd.to_numeric(ordered[target_column], errors="coerce")
    recent = ordered.tail(required_history_hours)
    recent_target = pd.to_numeric(recent[target_column], errors="coerce")
    missing_target = int(recent_target.isna().sum())
    completeness = float(1.0 - missing_target / max(required_history_hours, 1))

    observed_times = timestamps[target.notna()].sort_values()
    if len(observed_times) >= 2:
        gaps = observed_times.diff().dropna().dt.total_seconds().div(3600)
        largest_gap = float(gaps.max())
    else:
        largest_gap = None

    existing_exogenous = [column for column in exogenous_columns if column in recent.columns]
    missing_exogenous = (
        int(recent[existing_exogenous].isna().sum().sum()) if existing_exogenous else 0
    )
    missing_exogenous += len(exogenous_columns) - len(existing_exogenous)
    current_available = bool(pd.notna(target.iloc[-1]))
    warnings: list[str] = []
    if missing_target:
        warnings.append("pm25_history_missing")
    if missing_exogenous:
        warnings.append("exogenous_history_missing")
    if largest_gap is not None and largest_gap > allowed_gap_hours:
        warnings.append("gap_exceeds_allowed_policy")

    unusable = len(frame) < required_history_hours or not current_available
    degraded = (
        not unusable
        and (missing_target > 0 or missing_exogenous > 0 or bool(warnings))
    )
    status = "UNUSABLE" if unusable else "DEGRADED" if degraded else "GOOD"
    return {
        "status": status,
        "history_completeness": round(completeness, 4),
        "missing_lag_count": missing_target,
        "missing_exogenous_count": missing_exogenous,
        "largest_gap_hours": largest_gap,
        "fallback_required": bool(
            unusable or (largest_gap is not None and largest_gap > allowed_gap_hours)
        ),
        "warnings": warnings,
    }
