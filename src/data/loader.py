"""Data loader functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.quality import validate_schema
from src.data.schema import DEFAULT_SOURCE_TIMEZONE, normalize_timestamp_series


def resolve_data_path(configured_path: str | Path) -> Path:
    """Tìm CSV trong đường dẫn cấu hình, thư mục dự án hoặc /content (Colab)."""
    candidate = Path(configured_path).expanduser()
    # Support both running from repo root or parent workspace
    candidates = [
        candidate,
        Path.cwd() / candidate,
        Path(__file__).resolve().parents[2] / candidate,
        Path(__file__).resolve().parents[1] / candidate,
        Path("/content") / candidate.name,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    searched = "\n- ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu. Đã kiểm tra:\n- {searched}\n"
        "Hãy sửa data.path trong config hoặc tải CSV lên /content khi dùng Colab."
    )


def load_air_quality(config: dict[str, Any]) -> pd.DataFrame:
    """Đọc dữ liệu, chuẩn hóa timestamp về UTC và kiểm tra schema cơ bản."""
    data_config = config["data"]
    path = resolve_data_path(data_config["path"])
    frame = pd.read_csv(path)
    # Canonical v2 dùng station_id nhưng vẫn đọc được file legacy có cột station.
    station = data_config["station_column"]
    if station not in frame.columns:
        legacy_alias = "station" if station == "station_id" else "station_id"
        if legacy_alias in frame.columns:
            frame = frame.rename(columns={legacy_alias: station})
    validate_schema(frame, data_config["required_columns"])

    timestamp = data_config["timestamp_column"]
    target = data_config["target_column"]

    frame[timestamp] = normalize_timestamp_series(
        frame[timestamp],
        source_timezone=data_config.get("source_timezone", DEFAULT_SOURCE_TIMEZONE),
    )
    if frame[timestamp].isna().any():
        raise ValueError("Cột timestamp chứa giá trị không hợp lệ hoặc không parse được.")

    # Chuẩn hóa cả thời điểm phát hành để hợp đồng availability luôn so sánh
    # giữa hai chuỗi datetime cùng timezone, tránh lỗi mixed aware/naive.
    if "available_at" in frame.columns:
        frame["available_at"] = normalize_timestamp_series(
            frame["available_at"],
            source_timezone=data_config.get("source_timezone", DEFAULT_SOURCE_TIMEZONE),
        )

    numeric_columns = [
        data_config["target_column"],
        *data_config.get("optional_columns", []),
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=[station]).copy()

    if data_config.get("zero_as_missing", False):
        frame.loc[frame[target] == 0, target] = np.nan

    return frame
