"""Đọc và kiểm tra cấu hình dùng chung cho toàn bộ pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc và kiểm định cấu hình YAML từ đường dẫn được cung cấp."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại {config_path.resolve()}")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Kiểm tra tính hợp lệ của cấu hình theo contract chuẩn."""
    required_sections = {
        "project",
        "data",
        "features",
        "split",
        "model",
        "thresholds",
        "artifacts",
    }
    missing_sections = sorted(required_sections - set(config or {}))
    if missing_sections:
        raise ValueError(f"Cấu hình thiếu section: {', '.join(missing_sections)}")

    # Split checks
    split_cfg = config["split"]
    test_fraction = split_cfg.get("test_fraction")
    calibration_fraction = split_cfg.get("calibration_fraction", 0.1)
    coverage = split_cfg.get("coverage", 0.9)

    has_calendar_boundaries = all(
        split_cfg.get(key) for key in ("train_end", "calibration_end", "test_end")
    )
    if has_calendar_boundaries:
        boundaries = [
            split_cfg["train_end"],
            split_cfg["calibration_end"],
            split_cfg["test_end"],
        ]
        parsed_boundaries = [yaml.safe_load(f"value: {value}")["value"] for value in boundaries]
        if parsed_boundaries != sorted(parsed_boundaries):
            raise ValueError(
                "split.train_end, calibration_end và test_end phải tăng dần theo thời gian."
            )

    if not has_calendar_boundaries and (
        not isinstance(test_fraction, int | float) or not 0 < test_fraction < 1
    ):
        raise ValueError("split.test_fraction phải nằm trong khoảng (0, 1).")
    if not has_calendar_boundaries and (
        not isinstance(calibration_fraction, int | float) or not 0 < calibration_fraction < 1
    ):
        raise ValueError(
            "split.calibration_fraction phải nằm trong khoảng (0, 1) để đảm bảo "
            "tập hiệu chuẩn độc lập cho Conformal Prediction (P0.1)."
        )
    if (
        not has_calendar_boundaries
        and isinstance(test_fraction, int | float)
        and isinstance(calibration_fraction, int | float)
        and test_fraction + calibration_fraction >= 1
    ):
        raise ValueError("Tổng test_fraction và calibration_fraction phải nhỏ hơn 1.")
    if not isinstance(coverage, int | float) or not 0 < coverage < 1:
        raise ValueError("split.coverage phải nằm trong khoảng (0, 1).")

    # Model comparison checks
    models_section = config.get("models")
    candidates = config.get("model_comparison", {}).get("candidates", [])
    if candidates:
        if not isinstance(models_section, dict):
            raise ValueError(
                "Cấu hình phải có section `models:` map từng candidate name sang "
                "hyperparameters tương ứng (P0.5)."
            )
        missing_models = [name for name in candidates if name not in models_section]
        if missing_models:
            raise ValueError(
                "Các candidate sau thiếu trong `models:` section: "
                + ", ".join(sorted(missing_models))
            )

    # Backtest checks
    folds = split_cfg.get("backtest_folds")
    minimum_periods = split_cfg.get("minimum_train_periods")
    if not isinstance(folds, int) or folds < 2:
        raise ValueError("split.backtest_folds phải là số nguyên từ 2 trở lên.")
    if not isinstance(minimum_periods, int) or minimum_periods < 1:
        raise ValueError("split.minimum_train_periods phải là số nguyên dương.")

    # Features checks
    lags = config["features"].get("lags", [])
    windows = config["features"].get("rolling_windows", [])
    if not lags or any(not isinstance(val, int) or val < 1 for val in lags):
        raise ValueError("features.lags phải chứa các số nguyên dương.")
    if not windows or any(not isinstance(val, int) or val < 1 for val in windows):
        raise ValueError("features.rolling_windows phải chứa các số nguyên dương.")

    availability = config.get("feature_availability", {})
    invalid_availability = [
        name for name, delay in availability.items()
        if not isinstance(delay, int) or delay < 0
    ]
    if invalid_availability:
        raise ValueError(
            "feature_availability phải là số giờ nguyên không âm: "
            + ", ".join(sorted(invalid_availability))
        )

    # Threshold checks
    low_max = config["thresholds"].get("low_max", config["thresholds"].get("good_max"))
    medium_max = config["thresholds"].get("medium_max", config["thresholds"].get("moderate_max"))
    if low_max is None or medium_max is None or low_max >= medium_max:
        raise ValueError("thresholds.low_max phải nhỏ hơn thresholds.medium_max.")
