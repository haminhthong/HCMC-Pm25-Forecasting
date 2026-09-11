"""Chọn chiến lược dự báo dựa trên backtest và baseline Persistence."""

from __future__ import annotations

from typing import Any


def select_forecast_strategy(
    model_name: str,
    model_metrics: dict[str, Any],
    persistence_metrics: dict[str, Any],
    cv_mae_std: float,
    config: dict[str, Any],
    conformal_picp: float | None = None,
    coverage_target: float = 0.90,
) -> dict[str, Any]:
    """Quyết định dùng model ứng viên hay Persistence.

    Model được chọn bằng expanding-window backtest. Quyết định chiến lược
    cuối cùng chỉ dùng các metric của calibration và độ ổn định CV; final test
    được giữ nguyên để báo cáo sau cùng.
    """
    selection_cfg = config.get("model_selection", {})
    minimum_improvement = float(selection_cfg.get("minimum_mae_improvement", 0.05))
    maximum_cv_mae_std = float(selection_cfg.get("maximum_cv_mae_std", 1.0))
    maximum_picp_gap = float(selection_cfg.get("maximum_picp_gap", 0.15))

    model_mae = float(model_metrics["mae"])
    persistence_mae = float(persistence_metrics["mae"])
    improvement = (persistence_mae - model_mae) / persistence_mae if persistence_mae > 0 else 0.0
    passes_mae = model_mae < persistence_mae and improvement >= minimum_improvement
    passes_cv_stability = float(cv_mae_std) <= maximum_cv_mae_std

    if conformal_picp is None:
        picp_gap = None
        passes_picp = True
    else:
        picp_gap = abs(float(conformal_picp) - float(coverage_target))
        passes_picp = picp_gap <= maximum_picp_gap

    use_model = bool(passes_mae and passes_cv_stability and passes_picp)
    return {
        "best_cv_model": model_name,
        "forecast_strategy": model_name if use_model else "persistence",
        "model_beats_persistence": bool(passes_mae),
        "meets_selection_criteria": use_model,
        "mae_improvement_vs_persistence": float(improvement),
        "cv_mae_std": float(cv_mae_std),
        "picp_gap": None if picp_gap is None else float(picp_gap),
        "checks": {
            "mae_improvement_meets_threshold": bool(passes_mae),
            "cv_mae_std_within_threshold": bool(passes_cv_stability),
            "picp_gap_acceptable": bool(passes_picp),
        },
        "thresholds": {
            "minimum_mae_improvement": minimum_improvement,
            "maximum_cv_mae_std": maximum_cv_mae_std,
            "maximum_picp_gap": maximum_picp_gap,
            "coverage_target": float(coverage_target),
        },
        "status": "model" if use_model else "persistence",
    }
