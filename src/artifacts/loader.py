"""Self-contained artifact bundle loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.pipeline import Pipeline

from src.artifacts.schema import ForecastContext


def resolve_artifact_dir(
    artifact_path: str | Path = "artifacts",
    production_pointer: str | Path | None = None,
) -> Path:
    """Xác định đường dẫn thư mục artifact phiên bản đang kích hoạt (active version)."""
    dir_path = Path(artifact_path)

    # Pointer release mới được ưu tiên hơn flat mirror để không phục vụ nhầm
    # model cũ khi artifact root có cả active_release và model.joblib.
    pointer_path = Path(production_pointer) if production_pointer else None
    if pointer_path is None:
        pointer_path = next(
            (
                candidate
                for candidate in (dir_path / "active_release.json", dir_path / "production.json")
                if candidate.is_file()
            ),
            dir_path / "active_release.json",
        )
    if pointer_path.is_file():
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        active = payload.get("active_version")
        if active:
            resolved = (pointer_path.parent / "models" / active).resolve()
            if resolved.is_dir():
                return resolved
            flat_resolved = (pointer_path.parent / active).resolve()
            if flat_resolved.is_dir():
                return flat_resolved

    # Nếu người dùng truyền thẳng một bundle versioned không có pointer,
    # cho phép nạp trực tiếp model.joblib tại thư mục đó.
    if (dir_path / "model.joblib").is_file():
        return dir_path.resolve()

    return dir_path.resolve()


def load_artifact_bundle(artifact_dir: str | Path) -> dict[str, Any]:
    """Tải toàn bộ artifact bundle tự chứa (self-contained)."""
    resolved_dir = resolve_artifact_dir(artifact_dir)

    model_path = resolved_dir / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model.joblib tại {resolved_dir}")

    pipeline: Pipeline = joblib.load(model_path)

    metadata_path = resolved_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )

    schema_path = resolved_dir / "feature_schema.json"
    feature_schema = (
        json.loads(schema_path.read_text(encoding="utf-8"))
        if schema_path.is_file()
        else {}
    )

    snapshot_path = resolved_dir / "config_snapshot.yaml"
    config_snapshot = (
        yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
        if snapshot_path.is_file()
        else {}
    )

    split_manifest_path = resolved_dir / "split_manifest.json"
    split_manifest = (
        json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if split_manifest_path.is_file()
        else {}
    )

    raw_ctx = metadata.get("forecast_context")
    forecast_context = ForecastContext.from_dict(raw_ctx) if raw_ctx else ForecastContext()

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "feature_schema": feature_schema,
        "config": config_snapshot,
        "split_manifest": split_manifest,
        "forecast_context": forecast_context,
        "artifact_dir": resolved_dir,
    }
