"""Các hàm đọc và chuẩn hóa dữ liệu."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.quality import validate_schema
from src.data.schema import (
    DEFAULT_SOURCE_TIMEZONE,
    PHYSICAL_RANGES,
    normalize_timestamp_series,
)


def resolve_data_path(configured_path: str | Path) -> Path:
    """Tìm CSV trong đường dẫn cấu hình, thư mục dự án hoặc /content (Colab)."""
    candidate = Path(configured_path).expanduser()
    # Hỗ trợ chạy từ thư mục repository hoặc thư mục cha của workspace.
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
    station = data_config["station_column"]
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
        original_available = frame["available_at"].notna()
        frame["available_at"] = normalize_timestamp_series(
            frame["available_at"],
            source_timezone=data_config.get("source_timezone", DEFAULT_SOURCE_TIMEZONE),
        )
        if frame.loc[original_available, "available_at"].isna().any():
            raise ValueError("Cột available_at chứa giá trị không hợp lệ.")

    numeric_columns = [
        data_config["target_column"],
        *data_config.get("optional_columns", []),
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=[station]).copy()
    frame[station] = frame[station].astype(str).str.strip()
    if frame[station].eq("").any():
        raise ValueError(f"Cột {station} không được chứa station_id rỗng.")
    if frame.duplicated([station, timestamp]).any():
        raise ValueError(
            f"Dữ liệu chứa timestamp trùng sau khi chuẩn hóa station_id ({station}, {timestamp})."
        )

    if data_config.get("zero_as_missing", False):
        frame.loc[frame[target] == 0, target] = np.nan

    if target in PHYSICAL_RANGES:
        minimum, maximum = PHYSICAL_RANGES[target]
        valid_target = frame[target].isna() | frame[target].between(minimum, maximum)
        if not valid_target.all():
            invalid_count = int((~valid_target).sum())
            raise ValueError(
                f"Cột {target} có {invalid_count} giá trị ngoài miền [{minimum}, {maximum}]."
            )

    return frame
