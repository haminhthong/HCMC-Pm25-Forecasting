"""Regular hóa chuỗi thời gian theo lưới một giờ.

Đây là chính sách chung cho huấn luyện và dự báo:

* Với tần suất ``freq="h"``, chèn dòng ``NaN`` cho giờ thiếu để lag được tính
  trên lưới thời gian đều.
* Giữ giá trị thiếu là ``NaN``; pipeline feature dùng ``SimpleImputer`` ở bước
  sau để huấn luyện và dự báo có cùng cách xử lý.
* Hàm không làm thay đổi ``DataFrame`` đầu vào.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

HOURLY_FREQ = "h"


def regularize_hourly_series(
    frame: pd.DataFrame,
    timestamp_column: str,
    group_columns: Iterable[str] | None = None,
    freq: str = HOURLY_FREQ,
) -> pd.DataFrame:
    """Đưa từng chuỗi về lưới thời gian đều theo giờ.

    Hàm không xóa dòng hiện có, chỉ chèn các mốc giờ còn thiếu với giá trị
    ``NaN``. Kết quả được sắp xếp ổn định theo nhóm và timestamp để giữ thứ tự
    đầu vào khi có các giá trị trùng nhau.
    """
    if timestamp_column not in frame.columns:
        raise KeyError(f"timestamp_column {timestamp_column!r} không tồn tại trong frame.")

    work = frame.copy()
    work[timestamp_column] = pd.to_datetime(work[timestamp_column], errors="coerce")
    work = work.sort_values([*(group_columns or []), timestamp_column], kind="stable")
    if work[timestamp_column].isna().any():
        raise ValueError(
            "regularize_hourly_series phát hiện timestamp không parse được; "
            "hãy lọc trước khi gọi hàm này."
        )

    inserted = 0
    if group_columns:
        group_columns = list(group_columns)
        pieces = []
        grouping_key = group_columns[0] if len(group_columns) == 1 else group_columns
        for keys, group in work.groupby(grouping_key, sort=False):
            regularized_group = _regularize_single_series(group, timestamp_column, freq)
            if not isinstance(keys, tuple):
                keys = (keys,)
            for col, val in zip(group_columns, keys, strict=False):
                regularized_group[col] = val
            pieces.append(regularized_group)
            inserted += int(regularized_group.attrs.get("_inserted_missing_rows", 0))
        result = pd.concat(pieces, ignore_index=True)
    else:
        result = _regularize_single_series(work, timestamp_column, freq)
        inserted = int(getattr(result, "_inserted_missing_rows", 0))

    result.sort_values(
        by=[*(group_columns or []), timestamp_column],
        kind="stable",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    result.attrs["regularize_hourly_inserted_rows"] = inserted
    return result


def _regularize_single_series(
    frame: pd.DataFrame,
    timestamp_column: str,
    freq: str,
) -> pd.DataFrame:
    """Chèn mốc giờ thiếu cho một chuỗi đơn, không có cột nhóm."""
    if frame.empty:
        frame.attrs["_inserted_missing_rows"] = 0
        return frame
    sorted_frame = frame.sort_values(timestamp_column, kind="stable")
    full_index = pd.date_range(
        start=sorted_frame[timestamp_column].min(),
        end=sorted_frame[timestamp_column].max(),
        freq=freq,
    )
    if full_index.empty:
        sorted_frame.attrs["_inserted_missing_rows"] = 0
        return sorted_frame
    reindexed = sorted_frame.set_index(timestamp_column).reindex(full_index)
    reindexed.index.name = timestamp_column
    # Chỉ số này đếm đúng các mốc không xuất hiện trong dữ liệu ban đầu.
    original_ts = set(sorted_frame[timestamp_column])
    inserted_mask = ~reindexed.index.isin(original_ts)
    reindexed.attrs["_inserted_missing_rows"] = int(inserted_mask.sum())
    return reindexed.reset_index()


def audit_hourly_gaps(
    frame: pd.DataFrame,
    timestamp_column: str,
    group_columns: Iterable[str] | None = None,
    freq: str = HOURLY_FREQ,
) -> dict[str, int]:
    """Tính thống kê khoảng trống theo đúng chính sách regularization."""
    if timestamp_column not in frame.columns:
        raise KeyError(f"timestamp_column {timestamp_column!r} không tồn tại trong frame.")
    expected = pd.to_timedelta(1, unit="h" if freq == HOURLY_FREQ else freq)
    work = frame.copy()
    work[timestamp_column] = pd.to_datetime(work[timestamp_column], errors="coerce")
    if group_columns:
        group_columns = list(group_columns)
        work = work.sort_values([*group_columns, timestamp_column], kind="stable")
        gaps = work.groupby(group_columns, sort=False)[timestamp_column].diff().dropna()
    else:
        gaps = work[timestamp_column].sort_values().diff().dropna()
    irregular = int((gaps != expected).sum())
    return {
        "total_gaps_observed": int(gaps.size),
        "irregular_hourly_gaps": irregular,
        "expected_frequency": freq,
    }
