from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


TagAgg = Literal["mean", "last", "mode"]


@dataclass(frozen=True)
class NangangLoadSpec:
    data_dir: Path
    tz: str | None = None
    resample_rule: str = "1s"
    max_tags: int | None = None
    max_rows_per_tag: int | None = None
    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    # Tags that must appear in the loaded frame even when --max_tags truncates the sorted list.
    must_include_tags: list[str] | None = None
    numeric_agg: TagAgg = "mean"
    bool_agg: TagAgg = "mode"
    ffill_limit: int | None = 60


def discover_tags(data_dir: str | Path) -> list[str]:
    p = Path(data_dir)
    def is_ts_value_csv(fp: Path) -> bool:
        try:
            # Fast header sniff: avoid loading large files.
            # Handle UTF-8 BOM and extra whitespace.
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                header = f.readline()
            header = header.lstrip("\ufeff").strip()
            cols = [c.strip().strip('"').strip("'") for c in header.split(",")]
            cols_set = {c.lower() for c in cols if c}
            return ("ts" in cols_set) and ("value" in cols_set)
        except Exception:
            return False

    # Filter out non-OT CSVs (e.g. merged summary tables) in the same directory.
    tags = [f.stem for f in p.glob("*.csv") if is_ts_value_csv(f)]
    tags.sort()
    return tags


def _parse_boolish(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    out = s.map(mapping)
    return out


def _infer_is_boolish(values: pd.Series, sample_n: int = 200) -> bool:
    s = values.dropna().astype(str).str.strip().str.lower()
    if len(s) == 0:
        return False
    s = s.iloc[:sample_n]
    allowed = {"true", "false", "0", "1"}
    return set(s.unique()).issubset(allowed)


def _agg_series(values: pd.Series, agg: TagAgg) -> float:
    if len(values) == 0:
        return np.nan
    if agg == "last":
        return float(values.iloc[-1])
    if agg == "mean":
        return float(np.nanmean(values.to_numpy(dtype=float)))
    if agg == "mode":
        vc = values.value_counts(dropna=True)
        if len(vc) == 0:
            return np.nan
        return float(vc.index[0])
    raise ValueError(f"Unknown agg: {agg}")


def load_single_tag_csv(path: str | Path, *, nrows: int | None = None) -> tuple[pd.Series, bool]:
    """
    Returns:
      - series: indexed by timestamp, values float (boolish encoded as 0/1)
      - is_boolish: inferred from values
    """
    p = Path(path)
    df = pd.read_csv(p, usecols=["ts", "value"], nrows=nrows)
    ts = pd.to_datetime(df["ts"], errors="coerce")
    values_raw = df["value"]

    is_boolish = _infer_is_boolish(values_raw)
    if is_boolish:
        values = _parse_boolish(values_raw)
    else:
        values = pd.to_numeric(values_raw, errors="coerce")

    out = pd.Series(values.to_numpy(dtype=float), index=ts, name=p.stem)
    out = out[~out.index.isna()]
    out = out.sort_index()
    return out, is_boolish


def load_aligned_frame(spec: NangangLoadSpec) -> tuple[pd.DataFrame, dict[str, bool]]:
    """
    Loads all selected tags and aligns them to a uniform 1s grid.
    Returns:
      - df: index is timestamp (DatetimeIndex), columns are tags
      - meta: dict[tag] = is_boolish
    """
    tags = discover_tags(spec.data_dir)

    if spec.include_tags:
        include = set(spec.include_tags)
        tags = [t for t in tags if t in include]
    if spec.exclude_tags:
        exclude = set(spec.exclude_tags)
        tags = [t for t in tags if t not in exclude]
    if spec.max_tags is not None:
        tags = tags[: spec.max_tags]

    must = list(dict.fromkeys(spec.must_include_tags or []))
    for t in must:
        p = Path(spec.data_dir) / f"{t}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Required tag CSV not found: {p}")
    if must:
        missing = [t for t in must if t not in tags]
        if missing:
            if spec.max_tags is not None and len(must) > spec.max_tags:
                raise ValueError(
                    f"must_include_tags has {len(must)} tags but max_tags={spec.max_tags} "
                    "(increase max_tags or reduce required tags)."
                )
            rest = [t for t in tags if t not in missing]
            if spec.max_tags is not None:
                cap = max(0, spec.max_tags - len(missing))
                tags = missing + rest[:cap]
            else:
                tags = missing + rest

    meta: dict[str, bool] = {}
    series_list: list[pd.Series] = []

    for tag in tags:
        s, is_boolish = load_single_tag_csv(spec.data_dir / f"{tag}.csv", nrows=spec.max_rows_per_tag)
        meta[tag] = is_boolish

        # within-tag duplicate timestamps aggregation
        if s.index.has_duplicates:
            agg = spec.bool_agg if is_boolish else spec.numeric_agg
            s = (
                s.groupby(level=0)
                .apply(lambda x: _agg_series(x, agg=agg))
                .astype(float)
                .rename(tag)
            )
        series_list.append(s)

    if not series_list:
        raise ValueError("No tag CSV files selected/found.")

    df = pd.concat(series_list, axis=1)

    if spec.tz:
        # treat timestamps as local then convert
        df.index = df.index.tz_localize(spec.tz, nonexistent="NaT", ambiguous="NaT").tz_convert("UTC")

    df = df.sort_index()
    df = df.resample(spec.resample_rule).mean()

    if spec.ffill_limit is not None and spec.ffill_limit >= 0:
        df = df.ffill(limit=spec.ffill_limit)
    else:
        df = df.ffill()

    return df, meta


def select_feature_tags(
    all_tags: Iterable[str],
    target_tag: str,
    include_target_as_feature: bool = True,
) -> list[str]:
    tags = [t for t in all_tags if t != target_tag]
    if include_target_as_feature:
        tags = [target_tag] + tags
    return tags

