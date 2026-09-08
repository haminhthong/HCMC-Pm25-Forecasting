"""Backward-compatible facade for training pipeline.

Delegates core operations to the modular architecture under src/:
- src.pipeline.run_train_pipeline
- src.validation.split.split_by_time
- src.validation.backtest.expanding_time_folds, evaluate_candidate
- src.forecasting.trainer.make_pipeline, resolve_candidate_params
- src.forecasting.selector.build_quality_gate
- src.artifacts.writer.save_artifacts
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from src.artifacts.writer import save_artifacts
from src.evaluation.evaluator import evaluate_baselines
from src.forecasting.selector import build_quality_gate
from src.forecasting.trainer import make_pipeline, resolve_candidate_params
from src.pipeline import generate_model_version, run_train_pipeline
from src.validation.backtest import evaluate_candidate, expanding_time_folds
from src.validation.split import split_by_time

__all__ = [
    "build_quality_gate",
    "evaluate_baselines",
    "evaluate_candidate",
    "expanding_time_folds",
    "generate_model_version",
    "make_pipeline",
    "resolve_candidate_params",
    "save_artifacts",
    "split_by_time",
    "train",
]


def train(config_path: str, persist_artifacts: bool = True) -> dict[str, Any]:
    """Chạy pipeline huấn luyện đầy đủ (backward compatible)."""
    return run_train_pipeline(config_path=config_path, persist_artifacts=persist_artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình dự báo PM2.5 giờ tiếp theo")
    parser.add_argument("--config", default="configs/config.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--no-artifacts", action="store_true", help="Không ghi đè artifact lên đĩa")
    args = parser.parse_args()
    result = train(config_path=args.config, persist_artifacts=not args.no_artifacts)
    champion = result["serving_champion"]
    gate_status = result["metadata"]["calibration_gate"]
    # Windows có thể dùng code page cp1252; CLI của dự án có thông báo tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"Hoàn thành huấn luyện. Serving champion: {champion} (Calibration gate: {gate_status})")


if __name__ == "__main__":
    main()
