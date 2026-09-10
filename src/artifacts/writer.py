"""Ghi bộ artifact hiện tại của prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_artifacts(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Ghi model, metadata, evaluation và schema vào một thư mục cố định.

    Prototype chỉ giữ bộ artifact hiện tại. Provenance và kết quả split nằm
    trong JSON; cấu hình gốc vẫn được đọc từ ``configs/config.yaml``.
    """
    artifact_cfg = config["artifacts"]
    artifact_root = Path(artifact_cfg["directory"])
    artifact_root.mkdir(parents=True, exist_ok=True)

    model_path = artifact_root / artifact_cfg.get("model_file", "model.joblib")
    metadata_path = artifact_root / artifact_cfg.get("metadata_file", "metadata.json")
    evaluation_path = artifact_root / artifact_cfg.get("evaluation_file", "evaluation.json")
    schema_path = artifact_root / artifact_cfg.get("feature_schema_file", "feature_schema.json")

    joblib.dump(pipeline, model_path)
    _write_json(metadata_path, metadata)
    _write_json(evaluation_path, evaluation)

    feature_columns = list(metadata.get("features", []))
    feature_schema = {
        "timestamp_column": config["data"]["timestamp_column"],
        "station_column": config["data"]["station_column"],
        "target_column": config["data"]["target_column"],
        "model_feature_columns": feature_columns,
        "feature_count": len(feature_columns),
    }
    _write_json(schema_path, feature_schema)
    return artifact_root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ghi JSON UTF-8 và từ chối NaN để artifact luôn đọc được."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
