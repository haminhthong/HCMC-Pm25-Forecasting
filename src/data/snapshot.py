"""Dataset snapshot management and versioning."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.schema import (
    CANONICAL_STORAGE_TIMEZONE,
    DEFAULT_SOURCE_TIMEZONE,
    AirQualityDataset,
    normalize_timestamp_series,
)
from src.utils import sha256_file


def save_snapshot(dataset: AirQualityDataset, base_dir: str | Path = "data/raw") -> Path:
    """Lưu snapshot bất biến; không ghi đè snapshot cùng ``snapshot_id``."""
    target_dir = Path(base_dir) / dataset.snapshot_id
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(
            f"Snapshot {dataset.snapshot_id!r} đã tồn tại; hãy tạo snapshot_id mới để giữ raw immutable."
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = target_dir / "observations.csv"
    manifest_path = target_dir / "manifest.json"

    dataset.frame.to_csv(csv_path, index=False)
    manifest = dataset.to_manifest()
    manifest["sha256"] = sha256_file(csv_path)
    manifest["storage_timezone"] = "UTC"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return target_dir


def load_snapshot(snapshot_dir: str | Path) -> AirQualityDataset:
    """Tải snapshot dữ liệu canonical từ thư mục snapshot."""
    dir_path = Path(snapshot_dir)
    manifest_path = dir_path / "manifest.json"
    csv_path = dir_path / "observations.csv"

    if not manifest_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(f"Thư mục snapshot không hợp lệ: {dir_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("sha256")
    if expected_hash and sha256_file(csv_path) != expected_hash:
        raise ValueError(f"Snapshot bị thay đổi hoặc hỏng checksum: {csv_path}")
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = normalize_timestamp_series(
            df["timestamp"],
            source_timezone=manifest.get("source_timezone", DEFAULT_SOURCE_TIMEZONE),
        )
    if "available_at" in df.columns:
        df["available_at"] = normalize_timestamp_series(
            df["available_at"],
            source_timezone=manifest.get("source_timezone", DEFAULT_SOURCE_TIMEZONE),
        )

    return AirQualityDataset(
        frame=df,
        source=manifest.get("source", "unknown"),
        snapshot_id=manifest["snapshot_id"],
        frequency=manifest.get("frequency", "1h"),
        timezone=manifest.get("timezone", CANONICAL_STORAGE_TIMEZONE),
        station_ids=manifest.get("station_ids", []),
        metadata=manifest.get("metadata", {}),
    )
