from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LeakReportRow:
    report_title: str
    section: str  # "initial_check" | "model_vs_manual"
    zone: str
    strand: int
    status: str


_ZONE_KEYS = ["1区_侧弧", "1区_内外弧", "2区_侧弧", "2区_内弧", "2区_外弧", "3区", "4区"]


def _norm_zone(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace(" ", "")
        .replace("\u3000", "")
        .replace("＿", "_")
        .replace("—", "_")
    )


_ZONE_NORM_TO_CANON = {_norm_zone(z): z for z in _ZONE_KEYS}


def _clean_status(s: str) -> str:
    return str(s).strip().replace(" ", "")


def parse_leak_report_txt(path: str | Path) -> pd.DataFrame:
    """
    Parse the '停机二冷水堵漏检查对比报告' txt into a tidy table:
      columns: report_title, section, zone, strand, status
    """
    p = Path(path)
    raw_lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in raw_lines if ln]
    if not lines:
        return pd.DataFrame(columns=["report_title", "section", "zone", "strand", "status"])

    title = lines[0]
    rows: list[LeakReportRow] = []

    def find_idx(token: str) -> int | None:
        for i, ln in enumerate(lines):
            if token in ln:
                return i
        return None

    idx_initial = find_idx("初步检查")
    idx_compare = find_idx("核查与人工检查对比")

    # -----------------------
    # 1) 初步检查矩阵（只取前 7 个分区×7流的矩阵，后面的“阀位筛查/压力筛查”不解析）
    # -----------------------
    if idx_initial is not None:
        start = idx_initial + 1
        end = idx_compare if idx_compare is not None else len(lines)
        seg = lines[start:end]

        h = None
        for i in range(len(seg) - 1):
            if seg[i] == "分区" and seg[i + 1].endswith("流"):
                h = i
                break
        if h is not None:
            header = seg[h : h + 8]  # 分区 + 7流
            strands: list[int | None] = []
            for x in header[1:]:
                try:
                    strands.append(int(x.replace("流", "")))
                except Exception:
                    strands.append(None)

            i = h + 8
            seen_zones: set[str] = set()
            while i < len(seg):
                zc = _ZONE_NORM_TO_CANON.get(_norm_zone(seg[i]))
                if zc is not None and i + 7 < len(seg):
                    vals = seg[i + 1 : i + 8]
                    for j, v in enumerate(vals):
                        st = strands[j]
                        if st is None:
                            continue
                        rows.append(LeakReportRow(title, "initial_check", zc, int(st), _clean_status(v)))
                    seen_zones.add(zc)
                    i += 8
                    if len(seen_zones) >= len(_ZONE_KEYS):
                        break
                else:
                    i += 1

    # -----------------------
    # 2) 核查与人工检查对比（提取“模型”列；空单元格会导致缺失，遇到缺失直接跳过）
    # -----------------------
    if idx_compare is not None:
        seg = lines[idx_compare + 1 :]

        def parse_block(block: list[str], strands: list[int]) -> None:
            if not block or "分区" not in block:
                return
            s0 = block.index("分区")
            i = s0 + 1
            seen_zones: set[str] = set()
            while i < len(block):
                zc = _ZONE_NORM_TO_CANON.get(_norm_zone(block[i]))
                if zc is None:
                    i += 1
                    continue
                j = i + 1
                vals: list[str] = []
                while j < len(block) and (_ZONE_NORM_TO_CANON.get(_norm_zone(block[j])) is None) and (block[j] != "分区"):
                    vals.append(block[j])
                    j += 1
                for k, st in enumerate(strands):
                    idx = 2 * k  # 模型
                    if idx >= len(vals):
                        continue
                    model_status = _clean_status(vals[idx])
                    if not model_status:
                        continue
                    rows.append(LeakReportRow(title, "model_vs_manual", zc, int(st), model_status))
                seen_zones.add(zc)
                i = j
                if len(seen_zones) >= len(_ZONE_KEYS):
                    break

        idxs = [i for i, ln in enumerate(seg) if ln == "分区"]
        if len(idxs) >= 2:
            b1 = seg[idxs[0] : idxs[1]]
            b2 = seg[idxs[1] :]
        else:
            b1, b2 = seg, []

        parse_block(b1, [1, 2, 3, 4])
        parse_block(b2, [5, 6, 7])

    df = pd.DataFrame([r.__dict__ for r in rows])
    if not len(df):
        return pd.DataFrame(columns=["report_title", "section", "zone", "strand", "status"])
    return df.sort_values(["section", "zone", "strand"]).reset_index(drop=True)


def save_leak_report_csv(report_txt: str | Path, out_csv: str | Path) -> Path:
    df = parse_leak_report_txt(report_txt)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out

