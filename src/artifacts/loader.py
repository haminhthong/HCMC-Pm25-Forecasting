"""Nạp bộ artifact hiện tại cho inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def load_artifact_bundle(artifact_dir: str | Path = "artifacts") -> dict[str, Any]:
    """Nạp model và các metadata tùy chọn từ một thư mục artifact."""
    root = Path(artifact_dir).resolve()
    model_path = root / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model.joblib tại {root}")

    return {
        "pipeline": joblib.load(model_path),
        "metadata": _read_json(root / "metadata.json"),
        "evaluation": _read_json(root / "evaluation.json"),
        "feature_schema": _read_json(root / "feature_schema.json"),
        "artifact_dir": root,
    }


def _read_json(path: Path) -> dict[str, Any]:
    """Đọc JSON nếu có; metadata phụ không làm hỏng việc nạp model."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
