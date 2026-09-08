"""Tạo báo cáo Markdown ngắn từ evaluation artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def format_number(value: float | None) -> str:
    """Định dạng metric nhất quán; giữ rõ trường hợp chưa xác định."""
    return "N/A" if value is None else f"{value:.3f}"


def format_percent(value: float | None) -> str:
    """Định dạng phần trăm nhất quán."""
    return "N/A" if value is None else f"{value * 100:.1f}%"


def build_markdown(evaluation: dict) -> str:
    """Chuyển kết quả machine-readable thành báo cáo Markdown chuyên sâu."""
    # 1. Bảng so sánh Backtest trên tập Train
    backtest_rows = [
        "| Mô hình ứng viên | MAE CV trung bình | Độ lệch chuẩn (Std) | RMSE CV trung bình |",
        "|---|---:|---:|---:|",
    ]
    for name, report in evaluation.get("backtest", {}).items():
        backtest_rows.append(
            f"| `{name}` | {format_number(report['mae_mean'])} | "
            f"{format_number(report['mae_std'])} | {format_number(report['rmse_mean'])} |"
        )

    # 2. Bảng so sánh Final Test
    test_rows = [
        "| Mô hình / Chiến lược | MAE Test | RMSE Test | MASE | Skill vs Pers | Macro-F1 | Recall PM2.5 Cao |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    candidate_ml = evaluation.get("candidate_ml_test", evaluation.get("champion_test", {}))
    candidate_name = evaluation.get(
        "candidate_champion",
        evaluation.get("model_name", "ML candidate"),
    )
    pers = evaluation.get("persistence_test", evaluation.get("baselines", {}).get("persistence", {}))
    seasonal = evaluation.get(
        "seasonal_naive_test", evaluation.get("baselines", {}).get("seasonal_naive_24h", {})
    )
    serving = evaluation.get("serving_champion_test", evaluation.get("champion_test", {}))

    test_rows.append(
        f"| **Ứng viên ML ({candidate_name})** | "
        f"{format_number(candidate_ml.get('mae'))} | {format_number(candidate_ml.get('rmse'))} | "
        f"{format_number(candidate_ml.get('mase'))} | {format_percent(candidate_ml.get('skill_score_vs_persistence'))} | "
        f"{format_number(candidate_ml.get('macro_f1'))} | {format_percent(candidate_ml.get('high_pm25_recall'))} |"
    )
    test_rows.append(
        f"| **Persistence Baseline (t+1 = t)** | "
        f"{format_number(pers.get('mae'))} | {format_number(pers.get('rmse'))} | "
        f"{format_number(pers.get('mase', 1.0))} | 0.0% | "
        f"{format_number(pers.get('macro_f1'))} | {format_percent(pers.get('high_pm25_recall'))} |"
    )
    test_rows.append(
        f"| **Seasonal Naive 24h (t+1 = t-23h)** | "
        f"{format_number(seasonal.get('mae'))} | {format_number(seasonal.get('rmse'))} | "
        f"{format_number(seasonal.get('mase'))} | {format_percent(seasonal.get('skill_score_vs_persistence'))} | "
        f"{format_number(seasonal.get('macro_f1'))} | {format_percent(seasonal.get('high_pm25_recall'))} |"
    )
    test_rows.append(
        f"| 🏆 **Actual Serving Champion ({evaluation.get('serving_champion', 'N/A')})** | "
        f"{format_number(serving.get('mae'))} | {format_number(serving.get('rmse'))} | "
        f"{format_number(serving.get('mase'))} | {format_percent(serving.get('skill_score_vs_persistence'))} | "
        f"{format_number(serving.get('macro_f1'))} | {format_percent(serving.get('high_pm25_recall'))} |"
    )

    gate = evaluation.get("quality_gate", {})
    conformal = evaluation.get("conformal_test_evaluation", {})

    station_rows = [
        "| Trạm quan trắc | MAE Test | RMSE Test | PICP Conformal | Độ rộng khoảng (MPIW) |",
        "|---|---:|---:|---:|---:|",
    ]
    for station_name, st_metrics in evaluation.get("metrics_by_station", {}).items():
        # metrics_by_station lưu field phẳng; vẫn đọc nested schema cũ để
        # báo cáo artifact trước đây không bị hỏng.
        picp = st_metrics.get("conformal_picp")
        interval_width = st_metrics.get("conformal_mpiw")
        if picp is None or interval_width is None:
            conf_st = st_metrics.get("conformal_interval", {})
            picp = conf_st.get("picp")
            interval_width = conf_st.get("mean_interval_width")
        station_rows.append(
            f"| `{station_name}` | {format_number(st_metrics.get('mae'))} | "
            f"{format_number(st_metrics.get('rmse'))} | "
            f"{format_percent(picp)} | "
            f"±{format_number((interval_width or 0) / 2)} µg/m³ |"
        )

    return "\n".join(
        [
            "# 📋 Báo Cáo Đánh Giá Hệ Thống Dự Báo PM2.5 Giờ Tiếp Theo",
            "",
            "## 1. Kết Quả Expanding-Window Backtest (Tập Train)",
            "",
            *backtest_rows,
            "",
            "## 2. Đánh Giá Quality Gate & Quyết Định Serving Champion",
            "",
            f"- Quality gate: **{gate.get('status', 'N/A')}**",
            f"- Cải thiện MAE vs Persistence: **{format_percent(gate.get('mae_improvement_vs_persistence'))}** (đối chiếu theo cấu hình quality gate)",
            f"- Recall nhóm PM2.5 cao: **{format_percent(gate.get('high_pm25_recall'))}** (diagnostic, không phải điều kiện release)",
            f"- Độ lệch chuẩn Rolling MAE: **{format_number(gate.get('rolling_mae_std'))}** (đối chiếu theo cấu hình quality gate)",
            f"- **Chính sách phục vụ suy luận (Serving Champion):** `{evaluation.get('serving_champion', 'N/A')}`",

            "",
            "## 3. Đánh Giá Tập Test Cuối (Freeze-Policy Final Test)",
            "",
            *test_rows,
            "",
            "## 4. Kiểm Định Khoảng Tin Cậy Conformal (90% Target Coverage)",
            "",
            f"- **Độ phủ thực tế trên tập Test (PICP):** {format_percent(conformal.get('picp'))}",
            f"- **Độ rộng khoảng trung bình (MPIW):** {format_number(conformal.get('mean_interval_width'))} µg/m³ (±{format_number(conformal.get('mean_interval_width', 0) / 2)} µg/m³)",
            f"- **Độ rộng khoảng trung vị:** {format_number(conformal.get('median_interval_width'))} µg/m³",
            "",
            "## 5. Đánh Giá Phân Rã Theo Trạm Quan Trắc",
            "",
            *station_rows,
            "",
            "> ⚠️ **Lưu ý:** Báo cáo này áp dụng quy trình đánh giá chuẩn mực chống rò rỉ dữ liệu (Nested Temporal Evaluation). Khi áp dụng dữ liệu thực tế tại TP.HCM, cần kiểm tra nguồn và giấy phép cung cấp dữ liệu.",
        ]
    )

def main() -> None:
    """Đọc evaluation JSON và ghi báo cáo Markdown."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Tạo báo cáo Markdown từ evaluation artifact")
    parser.add_argument("--input", default="artifacts/evaluation.json")
    parser.add_argument("--output", default="reports/evaluation_summary.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.input == "artifacts/evaluation.json":
        pointer_path = Path("artifacts/active_release.json")
        if pointer_path.is_file():
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            active_version = pointer.get("active_version")
            if active_version:
                active_input = pointer_path.parent / "models" / active_version / "evaluation.json"
                if active_input.is_file():
                    input_path = active_input
    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy evaluation artifact tại {input_path}")
    evaluation = json.loads(input_path.read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(evaluation), encoding="utf-8")
    print(f"Đã tạo báo cáo: {output}")


if __name__ == "__main__":
    main()
