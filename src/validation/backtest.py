"""Expanding-window rolling backtest engine without future leakage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.calibration.conformal import conformal_quantile
from src.evaluate import regression_and_classification_metrics
from src.forecasting.trainer import make_pipeline


def expanding_time_folds(
    frame: pd.DataFrame,
    timestamp_column: str,
    folds: int,
    minimum_train_periods: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sinh expanding folds mà không làm rò rỉ target_timestamp sang validation."""
    periods = np.sort(frame[timestamp_column].unique())
    validation_periods = periods[minimum_train_periods:]
    if len(validation_periods) < folds:
        raise ValueError("Dữ liệu quá ít mốc thời gian để tạo đủ backtest folds.")

    result = []
    for period_group in np.array_split(validation_periods, folds):
        validation_start = period_group[0]
        validation_end = period_group[-1]
        if "target_timestamp" in frame.columns:
            train_mask = frame["target_timestamp"] < validation_start
        else:
            train_mask = frame[timestamp_column] < validation_start
        validation_mask = frame[timestamp_column].between(validation_start, validation_end)
        result.append((np.flatnonzero(train_mask), np.flatnonzero(validation_mask)))
    return result


def evaluate_candidate(
    name: str,
    train_frame: pd.DataFrame,
    config: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    """Đánh giá một model trên toàn bộ expanding-window folds và thu thập validation residuals."""
    fold_metrics = []
    validation_residuals = []
    split = config["split"]
    timestamp = config["data"]["timestamp_column"]
    for train_indices, validation_indices in expanding_time_folds(
        train_frame,
        timestamp,
        split["backtest_folds"],
        split["minimum_train_periods"],
    ):
        pipeline, _ = make_pipeline(config, name)
        train_fold = train_frame.iloc[train_indices]
        validation_fold = train_frame.iloc[validation_indices]
        pipeline.fit(train_fold[columns], train_fold["target_next_hour"])
        prediction = pipeline.predict(validation_fold[columns])
        abs_err = np.abs(validation_fold["target_next_hour"].to_numpy() - prediction)
        validation_residuals.extend(abs_err)
        fold_metrics.append(
            regression_and_classification_metrics(
                validation_fold["target_next_hour"],
                prediction,
                config["thresholds"],
            )
        )
    if not fold_metrics:
        raise ValueError("Dữ liệu quá ít cho rolling backtest với cấu hình hiện tại.")

    coverage_target = float(config.get("split", {}).get("coverage", 0.9))
    residual_q90 = (
        conformal_quantile(validation_residuals, coverage_target)
        if validation_residuals
        else 5.0
    )
    return {
        "folds": fold_metrics,
        "mae_mean": float(np.mean([item["mae"] for item in fold_metrics])),
        "mae_std": float(np.std([item["mae"] for item in fold_metrics])),
        "rmse_mean": float(np.mean([item["rmse"] for item in fold_metrics])),
        "residual_quantile_90": residual_q90,
    }
