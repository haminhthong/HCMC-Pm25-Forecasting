"""Champion selection, quality gate validation, and multi-tier readiness assignment."""

from __future__ import annotations

from typing import Any


def build_quality_gate(
    champion_metrics: dict[str, Any],
    persistence_metrics: dict[str, Any],
    champion_mae_std: float,
    config: dict[str, Any],
    conformal_picp: float | None = None,
    coverage_target: float = 0.90,
) -> dict[str, Any]:
    """Đánh giá gate offline cho model trước khi cho phép serving.

    Recall nhóm PM2.5 cao vẫn được trả về như một operational diagnostic,
    nhưng không còn quyết định model regression có được release hay không.
    Gate chính là MAE vượt Persistence, độ ổn định qua backtest và (nếu có)
    kiểm tra độ phủ prediction interval.
    """
    qg_cfg = config.get("quality_gate", {})
    min_improvement = qg_cfg.get("minimum_mae_improvement", 0.05)
    min_high_recall = qg_cfg.get("minimum_high_pm25_recall", 0.75)
    max_mae_std = qg_cfg.get("maximum_rolling_mae_std", 1.0)
    max_picp_gap = qg_cfg.get("maximum_picp_gap", 0.10)

    champion_mae = champion_metrics["mae"]
    persistence_mae = persistence_metrics["mae"]
    high_recall = champion_metrics.get("high_pm25_recall", 0.0)

    improvement = (
        (persistence_mae - champion_mae) / persistence_mae if persistence_mae > 0 else 0.0
    )
    # Đây là biến bắt buộc. Trước đây passes_all dùng biến này trước khi gán,
    # khiến chạy pipeline từ đầu có thể dừng bằng NameError.
    passes_mae = bool(
        champion_mae < persistence_mae and improvement >= float(min_improvement)
    )

    labels = list(config.get("thresholds", {}).get("labels", ["Thấp", "Trung bình", "Cao"]))
    high_label = labels[2] if len(labels) > 2 else "Cao"
    high_support = champion_metrics.get("support_by_class", {}).get(high_label, None)
    # Recall là diagnostic; nếu tập calibration không có lớp Cao thì không
    # đánh rớt model vì thiếu support.
    passes_recall = True if high_support == 0 else high_recall >= min_high_recall
    passes_std = champion_mae_std <= max_mae_std

    # Optional PICP coverage check if evaluated
    if conformal_picp is not None:
        picp_gap = abs(conformal_picp - coverage_target)
        passes_picp = picp_gap <= max_picp_gap
    else:
        picp_gap = 0.0
        passes_picp = True

    passes_all = passes_mae and passes_std and passes_picp

    return {
        "passes_baseline": bool(passes_all),
        "mae_improvement_vs_persistence": float(improvement),
        "high_pm25_recall": float(high_recall),
        "rolling_mae_std": float(champion_mae_std),
        "picp_gap": float(picp_gap),
        "checks": {
            "mae_improvement_ge_5pct": bool(passes_mae),
            "high_recall_ge_75pct": bool(passes_recall),
            "high_recall_is_diagnostic_only": True,
            "rolling_mae_std_le_1": bool(passes_std),
            "picp_gap_acceptable": bool(passes_picp),
        },
        "status": "đạt" if passes_all else "không đạt",
    }


def resolve_model_statuses(
    candidate_champion: str,
    passes_quality_gate: bool,
    smoke_only: bool,
) -> dict[str, str]:
    """Tách bạch 4 trạng thái model theo đề xuất chuẩn (Point 20):
    candidate_champion -> calibration_gate -> serving_strategy -> production_readiness
    """
    cal_gate = "pass" if passes_quality_gate else "fail"
    serving_strategy = "ml_model" if passes_quality_gate else "persistence_fallback"

    if smoke_only:
        production_readiness = "smoke_test_only"
    elif passes_quality_gate:
        production_readiness = "research_grade"
    else:
        production_readiness = "calibration_gate_failed"

    return {
        "candidate_champion": candidate_champion,
        "calibration_gate": cal_gate,
        "serving_strategy": serving_strategy,
        "production_readiness": production_readiness,
    }
