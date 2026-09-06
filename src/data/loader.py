"""Data loader functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.quality import validate_schema


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
    """Đọc dữ liệu bảng, chuẩn hóa timestamp và kiểm tra tính hợp lệ cơ bản."""
    data_config = config["data"]
    path = resolve_data_path(data_config["path"])
    frame = pd.read_csv(path)
    validate_schema(frame, data_config["required_columns"])

    timestamp = data_config["timestamp_column"]
    station = data_config["station_column"]
    target = data_config["target_column"]

    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
    if frame[timestamp].isna().any():
        raise ValueError("Cột timestamp chứa giá trị không hợp lệ hoặc không parse được.")

    frame = frame.dropna(subset=[station]).copy()

    if data_config.get("zero_as_missing", False):
        frame.loc[frame[target] == 0, target] = np.nan

    return frame
