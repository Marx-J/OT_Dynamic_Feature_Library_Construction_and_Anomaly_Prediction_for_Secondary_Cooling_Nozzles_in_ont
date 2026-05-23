from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .tag_catalog import parse_catalog_from_report_txt
from .utils import ensure_dir


@dataclass(frozen=True)
class CombineSpec:
    poc_data_dir: Path
    chart_data_dir: Path
    out_data_dir: Path
    report_txt: Path
    chart_tag_catalog_csv: Path
    out_tag_catalog_csv: Path


def _iter_tag_csvs(data_dir: Path) -> list[Path]:
    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()])


def _try_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink if possible (fast, no extra disk)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)


def combine_ot_sources(spec: CombineSpec) -> dict[str, str]:
    out_data_dir = ensure_dir(spec.out_data_dir)

    # 1) Merge OT csv files into one flat directory (required by loader).
    # Use hardlinks when possible to avoid copying large POC data.
    n_linked = 0
    n_copied = 0
    seen = set()

    for src_dir in [spec.poc_data_dir, spec.chart_data_dir]:
        for fp in _iter_tag_csvs(src_dir):
            name = fp.name
            if name in seen:
                # chart tags are synthetic and should not collide with raw IOT tags; if collide, keep first.
                continue
            seen.add(name)
            dst = out_data_dir / name
            before = dst.exists()
            _try_link_or_copy(fp, dst)
            if before:
                continue
            # best-effort infer: if hardlink succeeded, inode is shared (on NTFS).
            try:
                if os.path.samefile(fp, dst):
                    n_linked += 1
                else:
                    n_copied += 1
            except Exception:
                n_copied += 1

    # 2) Merge tag catalogs: (a) report catalog (raw IOT->name), (b) chart synthetic catalog.
    cat_report = parse_catalog_from_report_txt(spec.report_txt)
    cat_report = cat_report[["iot", "name"]].copy()
    cat_report["iot"] = cat_report["iot"].astype(str)
    cat_report["name"] = cat_report["name"].astype(str)

    cat_chart = pd.read_csv(spec.chart_tag_catalog_csv, encoding="utf-8-sig")
    if "iot" not in cat_chart.columns or "name" not in cat_chart.columns:
        raise ValueError("chart_tag_catalog_csv must have columns: iot,name")
    cat_chart = cat_chart[["iot", "name"]].copy()
    cat_chart["iot"] = cat_chart["iot"].astype(str)
    cat_chart["name"] = cat_chart["name"].astype(str)

    cat_all = pd.concat([cat_report, cat_chart], axis=0, ignore_index=True)
    cat_all = cat_all.dropna(subset=["iot", "name"])
    cat_all = cat_all.drop_duplicates(subset=["iot"], keep="first").sort_values("iot")

    spec.out_tag_catalog_csv.parent.mkdir(parents=True, exist_ok=True)
    cat_all.to_csv(spec.out_tag_catalog_csv, index=False, encoding="utf-8-sig")

    return {
        "out_data_dir": str(out_data_dir),
        "out_tag_catalog_csv": str(spec.out_tag_catalog_csv),
        "n_unique_csv": str(len(seen)),
        "n_linked_est": str(n_linked),
        "n_copied_est": str(n_copied),
        "n_catalog_rows": str(len(cat_all)),
    }

