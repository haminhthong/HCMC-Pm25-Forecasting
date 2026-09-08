"""Drift and data distribution monitoring module (Point 31)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_column_drift(
    reference: pd.Series,
    current: pd.Series,
) -> dict[str, float]:
    """Tính độ lệch phân phối thống kê cơ bản giữa tập tham chiếu và quan trắc hiện tại."""
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    if ref_clean.empty or cur_clean.empty:
        return {"mean_diff": 0.0, "std_diff": 0.0, "missing_rate_diff": 0.0}

    ref_missing = float(reference.isna().mean())
    cur_missing = float(current.isna().mean())

    mean_diff = float(cur_clean.mean() - ref_clean.mean())
    # ddof=0 để chuỗi chỉ có một mẫu không sinh NaN trong JSON report.
    std_diff = float(cur_clean.std(ddof=0) - ref_clean.std(ddof=0))

    return {
        "reference_mean": float(ref_clean.mean()),
        "current_mean": float(cur_clean.mean()),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "reference_missing_rate": ref_missing,
        "current_missing_rate": cur_missing,
        "missing_rate_diff": cur_missing - ref_missing,
    }


def monitor_drift(
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    columns_to_monitor: list[str],
) -> dict[str, Any]:
    """Giám sát độ trôi dạt phân phối và tỷ lệ khuyết dữ liệu trên các biến quan trọng."""
    report: dict[str, Any] = {}
    for col in columns_to_monitor:
        if col in reference_frame.columns and col in current_frame.columns:
            report[col] = compute_column_drift(reference_frame[col], current_frame[col])
    return report
