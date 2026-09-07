"""Hourly time-series regularization (P0.3: shared train/serving gap policy).

The pre-P0 codebase had two divergent policies:

* ``src/data.py::audit_air_quality`` only *reported* irregular gaps
  (``irregular_hourly_gaps`` counter) and let the feature pipeline propagate
  NaN through lag lookups, which the ``SimpleImputer(median)`` later filled.
* ``src/predict.py`` *rejected* the request with ``ValueError`` whenever any
  consecutive gap was not exactly 1 hour.

That asymmetry was a classic training-serving skew: a calibration set that
was perfectly legal during training would be refused at inference. This
module provides a single canonical policy used by both code paths:

* If a frequency is declared (``freq="h"`` in the canonical series), insert
  explicit ``NaN`` rows for missing hours so that lag features are computed
  on a regular grid.
* Missing values are kept as ``NaN``; downstream imputation is the
  responsibility of the feature pipeline (``SimpleImputer``) and is therefore
  identical for training and serving.
* The function is pure: it returns a new ``DataFrame`` and never mutates the
  input.
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
    """Return a DataFrame indexed on a regular hourly grid per group.

    Parameters
    ----------
    frame : DataFrame
        Input observations with at least the ``timestamp_column``.
    timestamp_column : str
        Name of the timestamp column.
    group_columns : iterable of str, optional
        Columns that identify an independent series (e.g. ``["station"]``).
        When provided, gaps are filled within each group. When ``None``, the
        whole frame is treated as one series.
    freq : str
        Pandas frequency alias for the expected cadence. Defaults to ``"h"``
        (hourly).

    Returns
    -------
    DataFrame
        A new frame sorted by group + timestamp. Missing hours are inserted
        as ``NaN`` rows for all non-key columns. The number of inserted
        missing rows is reported via the ``_inserted_missing_rows``
        attribute on the returned frame for diagnostic purposes.

    Notes
    -----
    The function does *not* drop existing rows; it only inserts gap rows.
    Sorting uses ``kind="stable"`` to preserve input ordering for ties.
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
        for keys, group in work.groupby(group_columns, sort=False):
            regularized_group = _regularize_single_series(group, timestamp_column, freq)
            if not isinstance(keys, tuple):
                keys = (keys,)
            for col, val in zip(group_columns, keys, strict=False):
                regularized_group[col] = val
            pieces.append(regularized_group)
            inserted += int(getattr(regularized_group, "_inserted_missing_rows", 0))
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
    """Insert missing rows for a single series (no grouping)."""
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
    # Inserted rows are exactly those whose index was not in the original frame.
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
    """Return gap statistics consistent with the regularization policy.

    The output schema is intentionally compatible with the pre-P0
    ``audit_air_quality`` field ``irregular_hourly_gaps`` so that the two
    audit surfaces can be merged.
    """
    if timestamp_column not in frame.columns:
        raise KeyError(f"timestamp_column {timestamp_column!r} không tồn tại trong frame.")
    expected = pd.to_timedelta(1, unit="h" if freq == HOURLY_FREQ else freq)
    work = frame.copy()
    work[timestamp_column] = pd.to_datetime(work[timestamp_column], errors="coerce")
    if group_columns:
        gaps = work.groupby(list(group_columns))[timestamp_column].diff().dropna()
    else:
        gaps = work[timestamp_column].sort_values().diff().dropna()
    irregular = int((gaps != expected).sum())
    return {
        "total_gaps_observed": int(gaps.size),
        "irregular_hourly_gaps": irregular,
        "expected_frequency": freq,
    }
