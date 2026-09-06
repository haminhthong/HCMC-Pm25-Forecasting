"""Metrics for regression, classification, and conformal intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

VALID_LABELS = ("Thấp", "Trung bình", "Cao")


def get_threshold_params(thresholds: dict[str, Any]) -> tuple[float, float, list[str]]:
    """Trích xuất ngưỡng và nhãn từ cấu hình thresholds."""
    low_max = float(thresholds.get("low_max", thresholds.get("good_max", 12.0)))
    medium_max = float(thresholds.get("medium_max", thresholds.get("moderate_max", 35.5)))
    labels = list(thresholds.get("labels", VALID_LABELS))
    return low_max, medium_max, labels


def classify_pm25(
    values: Any,
    low_max: float = 12.0,
    medium_max: float = 35.5,
    labels: list[str] | tuple[str, ...] = VALID_LABELS,
) -> np.ndarray:
    """Chuyển nồng độ PM2.5 thành ba mức phân tích nội bộ (Thấp, Trung bình, Cao)."""
    array = np.asarray(values, dtype=float)
    label_list = list(labels)
    return np.select([array <= low_max, array < medium_max], label_list[:2], default=label_list[2])


def regression_and_classification_metrics(
    y_true: Any,
    y_pred: Any,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Tính đồng thời metric hồi quy, phân lớp và phân tích class imbalance (Point 22)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    low_max, medium_max, labels = get_threshold_params(thresholds)

    true_labels = classify_pm25(y_true_arr, low_max, medium_max, labels)
    predicted_labels = classify_pm25(y_pred_arr, low_max, medium_max, labels)
    high_mask = y_true_arr >= medium_max

    # Support by class
    support_by_class = {
        label: int((true_labels == label).sum()) for label in labels
    }
    observed_labels = [label for label in labels if (true_labels == label).any()]

    # Macro F1: both all configured and observed only (Point 22)
    macro_f1_all = float(
        f1_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)
    )
    macro_f1_observed = (
        float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=observed_labels,
                average="macro",
                zero_division=0,
            )
        )
        if observed_labels
        else 0.0
    )

    if len(set(true_labels) | set(predicted_labels)) < 2:
        qwk = None
    else:
        qwk_value = float(
            cohen_kappa_score(
                true_labels,
                predicted_labels,
                labels=labels,
                weights="quadratic",
            )
        )
        qwk = qwk_value if np.isfinite(qwk_value) else None

    abs_errors = np.abs(y_true_arr - y_pred_arr)
    report = classification_report(
        true_labels, predicted_labels, labels=labels, output_dict=True, zero_division=0
    )

    high_label = labels[2] if len(labels) > 2 else "Cao"
    high_recall = float(report[high_label]["recall"]) if high_label in report else 0.0

    result = {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(mean_squared_error(y_true_arr, y_pred_arr) ** 0.5),
        "bias": float(np.mean(y_pred_arr - y_true_arr)),
        "p90_absolute_error": float(np.quantile(abs_errors, 0.9)),
        "macro_f1": macro_f1_all,
        "macro_f1_all_configured_classes": macro_f1_all,
        "macro_f1_observed_classes": macro_f1_observed,
        "support_by_class": support_by_class,
        "observed_classes": observed_labels,
        "qwk": qwk,
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=labels).tolist(),
        "classification_report": report,
    }
    result["high_pm25_mae"] = (
        float(mean_absolute_error(y_true_arr[high_mask], y_pred_arr[high_mask]))
        if high_mask.any()
        else None
    )
    result["high_pm25_recall"] = high_recall
    return result


def compute_mase(y_true: Any, y_pred: Any, y_naive: Any) -> float | None:
    """Tính MASE (Mean Absolute Scaled Error) so với naive baseline: MAE(model) / MAE(naive)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_naive_arr = np.asarray(y_naive, dtype=float)

    mae_model = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mae_naive = float(mean_absolute_error(y_true_arr, y_naive_arr))

    if mae_naive == 0:
        return 1.0 if mae_model == 0 else None
    return float(mae_model / mae_naive)


def compute_skill_score(y_true: Any, y_pred: Any, y_naive: Any) -> float:
    """Tính Skill Score so với Baseline: 1 - (MAE(model) / MAE(baseline))."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_naive_arr = np.asarray(y_naive, dtype=float)

    mae_model = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mae_naive = float(mean_absolute_error(y_true_arr, y_naive_arr))

    if mae_naive == 0:
        return 0.0
    return float(1.0 - (mae_model / mae_naive))


def conformal_interval_metrics(y_true: Any, y_lower: Any, y_upper: Any) -> dict[str, float]:
    """Tính các metric đánh giá chất lượng Conformal Prediction Interval."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_lower_arr = np.asarray(y_lower, dtype=float)
    y_upper_arr = np.asarray(y_upper, dtype=float)

    covered = (y_true_arr >= y_lower_arr) & (y_true_arr <= y_upper_arr)
    picp = float(np.mean(covered))
    widths = y_upper_arr - y_lower_arr
    mpiw = float(np.mean(widths))

    return {
        "picp": picp,
        "mpiw": mpiw,
        "min_width": float(np.min(widths)),
        "max_width": float(np.max(widths)),
        "coverage_error_vs_90pct": float(picp - 0.90),
    }
