from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeRange:
    start: pd.Timestamp
    end: pd.Timestamp


def parse_time_ranges_from_report_txt(path: str | Path) -> list[TimeRange]:
    """
    Extract the manual-operation ranges table from the report txt.
    It looks like pairs of datetimes:
      2025-11-27 06:34:17
      2025-11-28 07:32:29
      24.97
    """
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
    ranges: list[TimeRange] = []

    def try_dt(s: str) -> pd.Timestamp | None:
        try:
            ts = pd.to_datetime(s, errors="raise")
            if ts.year < 2000 or ts.year > 2100:
                return None
            return ts
        except Exception:
            return None

    i = 0
    while i + 1 < len(lines):
        a = try_dt(lines[i])
        b = try_dt(lines[i + 1])
        if a is not None and b is not None and b >= a:
            ranges.append(TimeRange(start=a, end=b))
            i += 2
            continue
        i += 1
    return ranges


def build_manual_mask(index: pd.DatetimeIndex, manual_ranges: list[TimeRange]) -> pd.Series:
    mask = pd.Series(False, index=index)
    for r in manual_ranges:
        mask |= (index >= r.start) & (index <= r.end)
    return mask


def label_deviation_over_threshold(
    actual: pd.Series,
    target: pd.Series,
    *,
    ratio: float = 0.2,
    horizon: int = 0,
) -> pd.Series:
    """
    y_t = 1 if |actual_{t+h}-target_{t+h}| > ratio * |target_{t+h}| else 0
    """
    h = int(horizon)
    a = actual.shift(-h).astype(float)
    t = target.shift(-h).astype(float)
    diff = (a - t).abs()
    thr = ratio * t.abs()
    y = (diff > thr).astype(float)
    return y.rename("y")


def label_deviation_event_over_threshold(
    actual: pd.Series,
    target: pd.Series,
    *,
    ratio: float = 0.2,
    horizon: int = 0,
    event_window: int = 30,
) -> pd.Series:
    """
    Event label to reduce class sparsity.

    y_t = 1 if any future time within [t+h, t+h+event_window-1] satisfies
          |actual-target| > ratio*|target|
        else 0
    """
    base = label_deviation_over_threshold(actual, target, ratio=ratio, horizon=horizon).astype(float)
    w = int(event_window)
    if w <= 1:
        return base.rename("y")
    # mark if there is any exceedance in the next w seconds
    y = base[::-1].rolling(window=w, min_periods=1).max()[::-1]
    return y.rename("y")


def filter_by_liquid_level(
    X: pd.DataFrame,
    y: pd.Series,
    liquid_level: pd.Series,
    *,
    lo: float = 140.0,
    hi: float = 160.0,
) -> tuple[pd.DataFrame, pd.Series]:
    ll = liquid_level.astype(float)
    m = ll.between(lo, hi)
    return X.loc[m], y.loc[m]

