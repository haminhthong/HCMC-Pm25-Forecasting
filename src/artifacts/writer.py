"""Versioned artifact writing and production pointer management."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.pipeline import Pipeline

from src.artifacts.schema import ForecastContext


def save_artifacts(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    config: dict[str, Any],
    split_manifest: dict[str, Any] | None = None,
    forecast_context: ForecastContext | None = None,
) -> Path:
    """Lưu model, metadata, schema, config snapshot, split manifest và evaluation."""
    artifact_root = Path(config["artifacts"]["directory"])
    artifact_root.mkdir(parents=True, exist_ok=True)

    versioned = config["artifacts"].get("versioned", True)
    model_version = metadata.get("model_version") or "latest"
    version_dir = artifact_root / "models" / model_version if versioned else artifact_root
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / config["artifacts"]["model_file"]
    evaluation_path = version_dir / config["artifacts"]["evaluation_file"]
    metadata_path = version_dir / config["artifacts"]["metadata_file"]
    schema_file = config["artifacts"].get("feature_schema_file", "feature_schema.json")
    schema_path = version_dir / schema_file
    config_snapshot_file = config["artifacts"].get("config_snapshot_file", "config_snapshot.yaml")
    config_snapshot_path = version_dir / config_snapshot_file
    split_manifest_path = version_dir / "split_manifest.json"

    # Embed ForecastContext in metadata if provided
    if forecast_context is not None:
        metadata["forecast_context"] = forecast_context.to_dict()

    joblib.dump(pipeline, model_path)
    evaluation_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    feature_schema = {
        "target_column": config["data"]["target_column"],
        "station_column": config["data"]["station_column"],
        "timestamp_column": config["data"]["timestamp_column"],
        "model_feature_columns": metadata.get("features", []),
        "feature_count": len(metadata.get("features", [])),
        "schema_version": 2,
    }
    schema_path.write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with config_snapshot_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True)

    if split_manifest is not None:
        split_manifest_path.write_text(
            json.dumps(split_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Pointer to active version in production.json
    production_pointer = artifact_root / "production.json"
    production_payload = {
        "active_version": model_version if versioned else ".",
        "updated_at": datetime.now(UTC).isoformat(),
        "production_readiness": metadata.get("production_readiness", "unknown"),
        "calibration_gate": metadata.get("calibration_gate", "unknown"),
        "artifact_path": model_path.as_posix(),
        "version_dir": version_dir.as_posix(),
    }
    production_pointer.write_text(
        json.dumps(production_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata["artifact_version_dir"] = version_dir.resolve().as_posix()
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # Legacy flat layout mirrors for backward compatibility if configured
    if versioned and config["artifacts"].get("mirror_flat_legacy", True):
        try:
            joblib.dump(pipeline, artifact_root / config["artifacts"]["model_file"])
            (artifact_root / config["artifacts"]["metadata_file"]).write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            (artifact_root / schema_file).write_text(
                json.dumps(feature_schema, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with (artifact_root / config_snapshot_file).open("w", encoding="utf-8") as file:
                yaml.safe_dump(config, file, allow_unicode=True)
        except Exception:
            pass

    return version_dir
