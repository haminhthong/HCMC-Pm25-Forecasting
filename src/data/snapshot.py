"""Dataset snapshot management and versioning."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.schema import AirQualityDataset


def save_snapshot(dataset: AirQualityDataset, base_dir: str | Path = "data/processed/snapshots") -> Path:
    """Lưu snapshot dữ liệu quan trắc canonical cùng manifest.json."""
    target_dir = Path(base_dir) / dataset.snapshot_id
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = target_dir / "observations.csv"
    manifest_path = target_dir / "manifest.json"

    dataset.frame.to_csv(csv_path, index=False)
    manifest = dataset.to_manifest()
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
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return AirQualityDataset(
        frame=df,
        source=manifest.get("source", "unknown"),
        snapshot_id=manifest["snapshot_id"],
        frequency=manifest.get("frequency", "1h"),
        timezone=manifest.get("timezone", "Asia/Ho_Chi_Minh"),
        station_ids=manifest.get("station_ids", []),
        metadata=manifest.get("metadata", {}),
    )
