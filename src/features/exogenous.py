"""Feature lookup theo thời gian thực và hợp đồng availability."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.schema import DEFAULT_SOURCE_TIMEZONE, normalize_timestamp_series


def get_feature_availability(config: dict[str, Any]) -> dict[str, int]:
    """Trả về độ trễ phát hành dữ liệu của từng biến, tính theo giờ."""
    return config.get("feature_availability", {})


def lookup_feature_at_offset(
    frame: pd.DataFrame,
    station_column: str,
    timestamp_column: str,
    feature_column: str,
    offset_hours: int,
    *,
    available_at_column: str = "available_at",
    enforce_availability: bool = True,
    source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> pd.Series:
    """Tra giá trị theo đúng ``(station, timestamp + offset)``.

    ``DataFrame.shift`` không phù hợp cho dữ liệu nhiều trạm vì có thể lấy dòng
    cuối của trạm A làm dữ liệu cho trạm B. Hàm này dùng khóa thời gian thực và,
    nếu có cột ``available_at``, chỉ nhận quan trắc đã thực sự có mặt tại thời
    điểm dự báo.
    """
    required = {station_column, timestamp_column, feature_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Thiếu cột để lookup feature: {', '.join(missing)}")

    work = frame.copy()
    if work.duplicated([station_column, timestamp_column]).any():
        raise ValueError(
            "Không thể lookup feature khi có timestamp trùng trong cùng trạm; "
            "hãy xử lý duplicate ở bước data quality trước."
        )

    left = work[[station_column, timestamp_column]].copy()
    left["_row_id"] = range(len(left))
    left["_lookup_timestamp"] = left[timestamp_column] + pd.to_timedelta(offset_hours, unit="h")

    right_columns = [station_column, timestamp_column, feature_column]
    has_availability = available_at_column in work.columns and enforce_availability
    if has_availability:
        right_columns.append(available_at_column)
    right = work[right_columns].copy()
    right = right.rename(
        columns={
            timestamp_column: "_source_timestamp",
            feature_column: "_source_value",
            available_at_column: "_source_available_at",
        }
    )

    merged = left.merge(
        right,
        left_on=[station_column, "_lookup_timestamp"],
        right_on=[station_column, "_source_timestamp"],
        how="left",
        sort=False,
    )

    if has_availability:
        # NaN availability không được coi là đã phát hành; đây là dữ liệu thiếu
        # metadata và phải làm feature missing thay vì vô tình cho qua leakage.
        available = normalize_timestamp_series(
            merged["_source_available_at"],
            source_timezone=source_timezone,
        )
        origins = normalize_timestamp_series(
            merged[timestamp_column],
            source_timezone=source_timezone,
        )
        valid = available.notna() & origins.notna() & (available <= origins)
        merged = merged.loc[valid].copy()
        if not merged.empty:
            merged = merged.sort_values(["_row_id", "_source_available_at"], kind="stable")
            merged = merged.drop_duplicates("_row_id", keep="last")
    else:
        merged = merged.drop_duplicates("_row_id", keep="first")

    values = pd.Series(pd.NA, index=range(len(work)), dtype="object")
    if not merged.empty:
        values.loc[merged["_row_id"].to_numpy()] = merged["_source_value"].to_numpy()
    # Trả lại đúng index của frame đầu vào; builder có thể đã sort nhưng vẫn
    # giữ index cũ, vì vậy Series RangeIndex sẽ gây lệch lag khi assignment.
    values.index = frame.index
    return pd.to_numeric(values, errors="coerce")


def prepare_exogenous_columns(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Chuẩn hóa feature ngoại sinh theo hợp đồng availability."""
    result = frame.copy()
    station = config["data"]["station_column"]
    timestamp = config["data"]["timestamp_column"]
    availability = get_feature_availability(config)
    exogenous = config.get("features", {}).get("exogenous_columns", [])

    for col in exogenous:
        latency = int(availability.get(col, 0))
        if col not in result.columns:
            # Giữ đúng feature schema để imputer xử lý dữ liệu thiếu toàn bộ biến.
            result[col] = pd.NA
            continue
        result[col] = lookup_feature_at_offset(
            result,
            station,
            timestamp,
            col,
            offset_hours=-latency,
            source_timezone=config.get("data", {}).get("source_timezone", DEFAULT_SOURCE_TIMEZONE),
        )

    return result
