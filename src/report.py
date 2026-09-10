"""Tạo báo cáo Markdown ngắn từ evaluation artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def format_number(value: float | None) -> str:
    """Định dạng metric và giữ rõ giá trị chưa có."""
    return "N/A" if value is None else f"{value:.3f}"


def format_percent(value: float | None) -> str:
    """Định dạng metric tỷ lệ thành phần trăm."""
    return "N/A" if value is None else f"{value * 100:.1f}%"


def build_markdown(evaluation: dict) -> str:
    """Chuyển evaluation JSON thành báo cáo dễ đọc."""
    backtest_rows = [
        "| Mô hình | MAE CV trung bình | Độ lệch chuẩn | RMSE CV trung bình |",
        "|---|---:|---:|---:|",
    ]
    for name, report in evaluation.get("backtest", {}).items():
        backtest_rows.append(
            f"| `{name}` | {format_number(report.get('mae_mean'))} | "
            f"{format_number(report.get('mae_std'))} | "
            f"{format_number(report.get('rmse_mean'))} |"
        )

    model_test = evaluation.get("model_test", {})
    persistence = evaluation.get("persistence_test", {})
    seasonal = evaluation.get("seasonal_naive_test", {})
    selected = evaluation.get("selected_strategy_test", {})
    model_name = evaluation.get("model_selection", {}).get("best_cv_model", "N/A")
    strategy = evaluation.get("model_selection", {}).get("forecast_strategy", "N/A")
    test_rows = [
        "| Mô hình / chiến lược | MAE | RMSE | MASE | Skill vs Persistence |",
        "|---|---:|---:|---:|---:|",
        f"| Ứng viên ML (`{model_name}`) | {format_number(model_test.get('mae'))} | "
        f"{format_number(model_test.get('rmse'))} | {format_number(model_test.get('mase'))} | "
        f"{format_percent(model_test.get('skill_score_vs_persistence'))} |",
        f"| Persistence | {format_number(persistence.get('mae'))} | "
        f"{format_number(persistence.get('rmse'))} | {format_number(persistence.get('mase', 1.0))} | 0.0% |",
        f"| Seasonal Naive 24h | {format_number(seasonal.get('mae'))} | "
        f"{format_number(seasonal.get('rmse'))} | {format_number(seasonal.get('mase'))} | "
        f"{format_percent(seasonal.get('skill_score_vs_persistence'))} |",
        f"| Chiến lược được chọn (`{strategy}`) | {format_number(selected.get('mae'))} | "
        f"{format_number(selected.get('rmse'))} | {format_number(selected.get('mase'))} | "
        f"{format_percent(selected.get('skill_score_vs_persistence'))} |",
    ]

    selection = evaluation.get("model_selection", {})
    conformal = evaluation.get("conformal_test_evaluation", {})
    station_rows = [
        "| Trạm | MAE | RMSE | PICP | Độ rộng khoảng |",
        "|---|---:|---:|---:|---:|",
    ]
    for station_name, metrics in evaluation.get("metrics_by_station", {}).items():
        picp = metrics.get("conformal_picp")
        width = metrics.get("conformal_mpiw")
        station_rows.append(
            f"| `{station_name}` | {format_number(metrics.get('mae'))} | "
            f"{format_number(metrics.get('rmse'))} | {format_percent(picp)} | "
            f"{format_number(width)} µg/m³ |"
        )

    return "\n".join(
        [
            "# Báo cáo đánh giá dự báo PM2.5 giờ tiếp theo",
            "",
            "## Expanding-window backtest",
            "",
            *backtest_rows,
            "",
            "## So sánh và chọn chiến lược",
            "",
            f"- Best CV model: `{selection.get('best_cv_model', 'N/A')}`",
            f"- Forecast strategy: `{selection.get('forecast_strategy', 'N/A')}`",
            f"- Cải thiện MAE so với Persistence: "
            f"**{format_percent(selection.get('mae_improvement_vs_persistence'))}**",
            f"- Trạng thái tiêu chí chọn model: **{selection.get('status', 'N/A')}**",
            "",
            *test_rows,
            "",
            "## Khoảng dự báo Conformal",
            "",
            f"- PICP trên final test: **{format_percent(conformal.get('picp'))}**",
            f"- Độ rộng khoảng trung bình: **{format_number(conformal.get('mean_interval_width'))} µg/m³**",
            "",
            "## Phân rã theo trạm",
            "",
            *station_rows,
            "",
            "> Kết quả chỉ phản ánh sample data dùng để kiểm thử hệ thống, không phải benchmark đại diện cho toàn TP.HCM.",
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
    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy evaluation artifact tại {input_path}")
    evaluation = json.loads(input_path.read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(evaluation), encoding="utf-8")
    print(f"Đã tạo báo cáo: {output}")


if __name__ == "__main__":
    main()
