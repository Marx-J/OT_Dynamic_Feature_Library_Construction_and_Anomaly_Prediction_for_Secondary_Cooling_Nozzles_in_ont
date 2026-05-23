from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


_RE_ISA = re.compile(r"^ISA[0-9a-fA-F]+$")
_RE_IOT = re.compile(r"^\d{13,}$")


@dataclass(frozen=True)
class CatalogRow:
    isa: str | None
    iot: str
    name: str


def parse_catalog_from_report_txt(path: str | Path) -> pd.DataFrame:
    """
    Parse rows like:
      ISAxxxx
      2312xxxxxxxxxxx
      1ST_二冷水_2区_侧弧_阀位_CTL
    """
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
    rows: list[CatalogRow] = []

    i = 0
    while i < len(lines):
        ln = lines[i]
        if _RE_ISA.match(ln) and i + 2 < len(lines):
            isa = ln
            iot = lines[i + 1]
            name = lines[i + 2]
            if _RE_IOT.match(iot) and name and ("_" in name or "二冷" in name):
                rows.append(CatalogRow(isa=isa, iot=iot, name=name))
                i += 3
                continue
        # sometimes report has only iot + name (no ISA)
        if _RE_IOT.match(ln) and i + 1 < len(lines):
            iot = ln
            name = lines[i + 1]
            if name and ("_" in name or "二冷" in name):
                rows.append(CatalogRow(isa=None, iot=iot, name=name))
                i += 2
                continue
        i += 1

    df = pd.DataFrame([r.__dict__ for r in rows]).drop_duplicates(subset=["iot"]).sort_values("iot")
    return df


def save_catalog_csv(df: pd.DataFrame, out_csv: str | Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv

