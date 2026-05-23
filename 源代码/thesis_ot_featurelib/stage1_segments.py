from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_nangang import NangangLoadSpec, load_aligned_frame
from .labeling import TimeRange, build_manual_mask, filter_by_liquid_level, parse_time_ranges_from_report_txt
from .tag_catalog import parse_catalog_from_report_txt
from .utils import ensure_dir
from .discover import DiscoverSpec, discover_related_tags
from .io_nangang import load_single_tag_csv


@dataclass(frozen=True)
class ManualValveSetSpec:
    enable: bool = True
    # Mark constant-valve periods as manual valve set.
    min_const_hours: float = 0.5

    # Consecutive diff threshold.
    diff_tol_abs: float = 0.05
    diff_tol_rel: float = 0.001

    # If we have a valve/flow "auto-active" boolean series, require it to be mostly 1.
    auto_active_required: bool = True
    auto_active_threshold: float = 0.9


@dataclass(frozen=True)
class StableSegmentSpec:
    # deviation_ratio 保留字段（用于后续诊断/标注口径），
    # 但“稳态段提取”不应依赖 ACT-TRG 偏差，否则会把疑似堵塞/漏水误当成非稳态过滤掉。
    deviation_ratio: float = 0.2
    min_segment_hours: float = 1.0

    # Optional report-aligned liquid level filter.
    enable_liquid_level_filter: bool = False
    liquid_level_lo: float = 140.0
    liquid_level_hi: float = 160.0
    liquid_level_tag: str | None = None  # csv stem (tag id)

    # If we have a boolean valve_ctl / auto-active tag, optionally require it.
    require_auto_active: bool = True

    manual: ManualValveSetSpec = ManualValveSetSpec()


@dataclass(frozen=True)
class CircuitTags:
    circuit_key: str
    flow_act_tag: str | None
    flow_trg_tag: str | None
    valve_pos_tag: str | None
    auto_active_tag: str | None
    pressure_tag: str | None = None


_RE_FLOW_ZONE_STRAND = re.compile(
    r"^(?P<flow>[1-7])ST_二冷水_(?P<zone>[1-7])区(?:(?:_(?P<strand>侧弧|内弧|外弧|内外弧)))?(_|$)"
)
_RE_FLOW_ZONE_STRAND_ALT = re.compile(
    r"^(?P<flow>[1-7])ST_二冷水_(?P<zone>[1-7])区(?:_(?P<strand>侧弧|内弧|外弧|内外弧))?_"
)


def _infer_circuit_key_from_name(tag_name: str) -> str | None:
    m = _RE_FLOW_ZONE_STRAND.match(tag_name.replace(" ", ""))
    if m:
        flow = m.group("flow")
        zone = m.group("zone")
        strand = m.group("strand")
        if strand:
            return f"{flow}ST_二冷水_{zone}区_{strand}"
        return f"{flow}ST_二冷水_{zone}区"
    m = _RE_FLOW_ZONE_STRAND_ALT.match(tag_name.replace(" ", ""))
    if m:
        flow = m.group("flow")
        zone = m.group("zone")
        strand = m.group("strand")
        if strand:
            return f"{flow}ST_二冷水_{zone}区_{strand}"
        return f"{flow}ST_二冷水_{zone}区"
    return None


def _infer_tag_kind_from_name(tag_name: str) -> str | None:
    n = tag_name
    if "阀位_CTL" in n:
        return "auto_active"  # 二冷阀位控制投入/模式（布尔ish）
    if "阀位" in n and "阀位_CTL" not in n:
        return "valve_pos"
    if "流量_TRG" in n:
        return "flow_trg"
    if "流量_ACT" in n:
        return "flow_act"
    if "流量" in n and ("_ACT" in n or "实际" in n):
        return "flow_act"
    if "压力" in n:
        return "pressure"
    if "液位" in n:
        return "liquid_level"
    return None


def build_circuit_tags_from_report_catalog(report_txt: str | Path, tag_catalog_csv: str | Path | None = None) -> dict[str, CircuitTags]:
    """
    Build a best-effort mapping:
      circuit_key -> {flow_act, flow_trg, valve_pos, auto_active}

    If the report only contains a subset of tags, some fields may be None.
    """
    if tag_catalog_csv:
        cat = pd.read_csv(tag_catalog_csv, encoding="utf-8-sig")
    else:
        cat = parse_catalog_from_report_txt(report_txt)

    rows: list[CircuitTags] = []

    # keep first occurrence per (circuit_key, kind)
    tmp: dict[str, dict[str, str]] = {}
    for _, r in cat.iterrows():
        iot = str(r["iot"])
        name = str(r["name"])
        circuit_key = _infer_circuit_key_from_name(name)
        if not circuit_key:
            continue
        kind = _infer_tag_kind_from_name(name)
        if not kind:
            continue
        tmp.setdefault(circuit_key, {})
        if kind not in tmp[circuit_key]:
            tmp[circuit_key][kind] = iot

    out: dict[str, CircuitTags] = {}
    for circuit_key, m in tmp.items():
        out[circuit_key] = CircuitTags(
            circuit_key=circuit_key,
            flow_act_tag=m.get("flow_act"),
            flow_trg_tag=m.get("flow_trg"),
            valve_pos_tag=m.get("valve_pos"),
            auto_active_tag=m.get("auto_active"),
            pressure_tag=m.get("pressure"),
        )
    return out


def _mask_to_time_ranges(index: pd.DatetimeIndex, mask: np.ndarray, *, min_points: int) -> list[TimeRange]:
    if len(index) == 0:
        return []
    if len(mask) != len(index):
        raise ValueError("mask length mismatch")

    mask = mask.astype(bool)
    if mask.sum() < min_points:
        return []

    # run-length encoding on mask
    starts: list[int] = []
    ends: list[int] = []
    in_run = False
    run_start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            run_start = i
        elif not v and in_run:
            in_run = False
            starts.append(run_start)
            ends.append(i - 1)
    if in_run:
        starts.append(run_start)
        ends.append(len(mask) - 1)

    ranges: list[TimeRange] = []
    for s, e in zip(starts, ends):
        n = e - s + 1
        if n < min_points:
            continue
        ranges.append(TimeRange(start=index[s], end=index[e]))
    return ranges


def detect_manual_valve_set_periods(
    *,
    index: pd.DatetimeIndex,
    valve_pos: pd.Series,
    auto_active: pd.Series | None,
    spec: ManualValveSetSpec,
) -> list[TimeRange]:
    if not spec.enable:
        return []
    if valve_pos is None or len(valve_pos) == 0:
        return []

    v = valve_pos.astype(float)
    if v.isna().all():
        return []

    step_seconds = float((index[1] - index[0]).total_seconds()) if len(index) >= 2 else 1.0
    min_const_points = max(1, int(np.ceil(spec.min_const_hours * 3600.0 / step_seconds)))

    # consecutive diff small => constant
    dv = v.diff().abs()
    baseline = float(v.dropna().median())
    tol = float(spec.diff_tol_abs + spec.diff_tol_rel * abs(baseline))
    # dv <= tol 代表“相邻采样点阀位变化很小”，把该点视为常值点；进一步用连续常值段识别人工定阀位。
    const_point_mask = dv.le(tol) & v.notna() & dv.notna()
    const_ranges = _mask_to_time_ranges(index, const_point_mask.to_numpy(), min_points=min_const_points)
    if not const_ranges:
        return []

    if auto_active is None:
        return const_ranges

    aa = auto_active.astype(float)
    aa_mask = aa.gt(0.5).reindex(index).fillna(False).to_numpy()

    out: list[TimeRange] = []
    for r in const_ranges:
        m = (index >= r.start) & (index <= r.end)
        if m.sum() == 0:
            continue
        if not spec.auto_active_required:
            out.append(r)
            continue
        frac = float(aa_mask[m].mean()) if m.any() else 0.0
        if frac >= spec.auto_active_threshold:
            out.append(r)
    return out


def extract_stable_segments(
    *,
    df: pd.DataFrame,
    circuit_tags: CircuitTags,
    spec: StableSegmentSpec,
    manual_ranges_report_txt: str | Path | None = None,
) -> dict[str, Any]:
    """
    Returns:
      dict with fields:
        circuit_key
        stable_ranges_raw (before manual filtering)
        stable_ranges_final (after manual filtering)
        used_tags (what was used / missing)
    """
    flow_act = circuit_tags.flow_act_tag
    flow_trg = circuit_tags.flow_trg_tag

    missing: list[str] = []
    for k, tag in [
        ("flow_act", flow_act),
        ("flow_trg", flow_trg),
    ]:
        if tag is None or tag not in df.columns:
            missing.append(k)
    if missing:
        return {
            "circuit_key": circuit_tags.circuit_key,
            "stable_ranges_raw": [],
            "stable_ranges_final": [],
            "used_tags": {"missing": missing, "present": [c for c in [flow_act, flow_trg] if c and c in df.columns]},
        }

    act = df[flow_act].astype(float)
    trg = df[flow_trg].astype(float)

    valid = act.notna() & trg.notna()

    # 稳态浇铸片段：这里使用“可用工况代理指标”（act/trg 同步存在）
    # 不依赖 ACT-TRG 偏差阈值，否则会把堵塞/漏水相关异常直接过滤掉，
    # 导致后续训练无法学习异常模式。
    stable_point_mask = valid.to_numpy()

    # 可选约束：若存在与控制“自动投入/自动控制”相关的布尔点位，则只在自动投入处统计稳态。
    used_auto_active = None
    if spec.require_auto_active and circuit_tags.auto_active_tag and circuit_tags.auto_active_tag in df.columns:
        used_auto_active = circuit_tags.auto_active_tag
        stable_point_mask &= df[used_auto_active].astype(float).gt(0.5).to_numpy()

    # Optional: liquid level filter (only constrains casting mode stability, not deviation)
    if spec.enable_liquid_level_filter:
        ll_tag = spec.liquid_level_tag
        if ll_tag and ll_tag in df.columns:
            ll = df[ll_tag].astype(float)
            stable_point_mask &= ll.between(float(spec.liquid_level_lo), float(spec.liquid_level_hi)).fillna(False).to_numpy()

    index = df.index
    step_seconds = float((index[1] - index[0]).total_seconds()) if len(index) >= 2 else 1.0
    min_points = max(1, int(np.ceil(spec.min_segment_hours * 3600.0 / step_seconds)))
    stable_ranges_raw = _mask_to_time_ranges(index, stable_point_mask, min_points=min_points)

    # Manual filtering (report-based + data-driven heuristic)
    manual_mask = pd.Series(False, index=index)

    if manual_ranges_report_txt:
        ranges = parse_time_ranges_from_report_txt(manual_ranges_report_txt)
        manual_mask |= build_manual_mask(index, ranges)

    valve_pos = circuit_tags.valve_pos_tag if circuit_tags.valve_pos_tag in df.columns else None
    auto_active_series = None
    if circuit_tags.auto_active_tag and circuit_tags.auto_active_tag in df.columns:
        auto_active_series = df[circuit_tags.auto_active_tag]

    if valve_pos is not None:
        vps = df[valve_pos]
        detected = detect_manual_valve_set_periods(
            index=index,
            valve_pos=vps,
            auto_active=auto_active_series,
            spec=spec.manual,
        )
        if detected:
            # combine detected ranges into mask
            for r in detected:
                manual_mask |= (index >= r.start) & (index <= r.end)

    stable_final_mask = stable_point_mask & (~manual_mask.to_numpy())
    stable_ranges_final = _mask_to_time_ranges(index, stable_final_mask, min_points=min_points)

    def _ranges_to_json(ranges: list[TimeRange]) -> list[dict[str, str]]:
        return [{"start": str(r.start), "end": str(r.end)} for r in ranges]

    return {
        "circuit_key": circuit_tags.circuit_key,
        "stable_ranges_raw": _ranges_to_json(stable_ranges_raw),
        "stable_ranges_final": _ranges_to_json(stable_ranges_final),
        "used_tags": {
            "flow_act_tag": flow_act,
            "flow_trg_tag": flow_trg,
            "auto_active_tag_used": used_auto_active,
            "manual_detection_valve_pos_tag": circuit_tags.valve_pos_tag if valve_pos is not None else None,
            "manual_report_txt_used": bool(manual_ranges_report_txt),
        },
    }


def run_extract_stable_segments(
    *,
    data_dir: str | Path,
    report_txt: str | Path | None,
    tag_catalog_csv: str | Path | None,
    out_dir: str | Path,
    config: dict[str, Any],
) -> Path:
    out_dir = ensure_dir(out_dir)

    stable_spec = StableSegmentSpec(
        deviation_ratio=float(config.get("deviation_ratio", 0.2)),
        min_segment_hours=float(config.get("min_segment_hours", 1.0)),
        enable_liquid_level_filter=bool(config.get("enable_liquid_level_filter", False)),
        liquid_level_lo=float(config.get("liquid_level_lo", 140.0)),
        liquid_level_hi=float(config.get("liquid_level_hi", 160.0)),
        liquid_level_tag=config.get("liquid_level_tag"),
        require_auto_active=bool(config.get("require_auto_active", True)),
        manual=ManualValveSetSpec(**config.get("manual", {})),
    )

    if not (report_txt or tag_catalog_csv):
        raise ValueError("Provide at least one of report_txt or tag_catalog_csv")

    circuit_tags_map = build_circuit_tags_from_report_catalog(
        report_txt=report_txt or Path(""),
        tag_catalog_csv=tag_catalog_csv,
    )

    # Optionally discover missing ACT tag from TRG.
    if bool(config.get("auto_discover_flow_act", True)):
        discover_max_rows = (
            config.get("discover_max_rows_per_tag", None)
            or config.get("max_rows_per_tag", None)
            or 80_000
        )
        discover_top_k = int(config.get("discover_top_k_related", 30))
        score_deviation_ratio = float(config.get("deviation_ratio", 0.2))

        def _align_two(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
            if a.index.has_duplicates:
                a = a.groupby(level=0).mean()
            if b.index.has_duplicates:
                b = b.groupby(level=0).mean()
            df = pd.concat([a.rename("a"), b.rename("b")], axis=1).sort_index()
            df = df.resample(str(config.get("resample_rule", "1s"))).mean().ffill(limit=int(config.get("ffill_limit", 60)))
            return df["a"], df["b"]

        def _score_act_tag(act_tag: str, trg_s: pd.Series) -> float:
            a_s, _ = load_single_tag_csv(Path(data_dir) / f"{act_tag}.csv", nrows=int(discover_max_rows))
            aa, bb = _align_two(trg_s, a_s)  # aa=trg, bb=act? keep semantic as in stable calc below
            # stable if |act-trg| <= ratio*|trg|
            act = bb.astype(float)
            trg = aa.astype(float)
            valid = act.notna() & trg.notna()
            diff = (act - trg).abs()
            denom = trg.abs()
            eps = 1e-9
            stable = valid & denom.gt(eps) & diff.le(score_deviation_ratio * denom)
            # 稳态得分：在当前取样窗口内满足稳态判据的比例越高越好。
            if stable.sum() == 0:
                return 0.0
            return float(stable.mean())

        for k, ct in list(circuit_tags_map.items()):
            if ct.flow_act_tag is not None:
                continue
            if not ct.flow_trg_tag:
                continue

            trg_tag = str(ct.flow_trg_tag)
            trg_s, _ = load_single_tag_csv(Path(data_dir) / f"{trg_tag}.csv", nrows=int(discover_max_rows))

            rel = discover_related_tags(
                data_dir,
                DiscoverSpec(
                    target_tag=trg_tag,
                    max_rows_per_tag=int(discover_max_rows),
                    top_k=discover_top_k,
                ),
            )
            if rel.empty:
                continue

            best_act = None
            best_score = -1.0
            for cand in rel["tag"].astype(str).tolist():
                if cand == trg_tag:
                    continue
                sc = _score_act_tag(cand, trg_s)
                # 选“稳态得分最高”的 ACT，尽量避免自动发现把 TRG 配错 ACT。
                if sc > best_score:
                    best_score = sc
                    best_act = cand

            if best_act:
                circuit_tags_map[k] = CircuitTags(
                    circuit_key=ct.circuit_key,
                    flow_act_tag=str(best_act),
                    flow_trg_tag=ct.flow_trg_tag,
                    valve_pos_tag=ct.valve_pos_tag,
                    auto_active_tag=ct.auto_active_tag,
                )

    # For each circuit, load only its required tags.
    segments: list[dict[str, Any]] = []
    all_tags_needed = set()
    circuit_list = list(circuit_tags_map.values())
    for ct in circuit_list:
        for t in [ct.flow_act_tag, ct.flow_trg_tag, ct.valve_pos_tag, ct.auto_active_tag, stable_spec.liquid_level_tag]:
            if t:
                all_tags_needed.add(str(t))
        if ct.pressure_tag:
            all_tags_needed.add(str(ct.pressure_tag))

    load_spec = NangangLoadSpec(
        data_dir=Path(data_dir),
        include_tags=sorted(all_tags_needed),
        resample_rule=str(config.get("resample_rule", "1s")),
        ffill_limit=int(config.get("ffill_limit", 60)),
        max_rows_per_tag=config.get("max_rows_per_tag"),
    )
    df_all, _meta = load_aligned_frame(load_spec)

    for ct in circuit_list:
        # build a per-circuit dataframe view (keep required cols)
        cols = [t for t in [ct.flow_act_tag, ct.flow_trg_tag, ct.valve_pos_tag, ct.auto_active_tag, stable_spec.liquid_level_tag] if t and t in df_all.columns]
        if not cols:
            continue
        df = df_all[cols]
        res = extract_stable_segments(
            df=df,
            circuit_tags=ct,
            spec=stable_spec,
            manual_ranges_report_txt=str(config.get("manual_ranges_report_txt") or report_txt) if (config.get("manual_ranges_report_txt") or report_txt) else None,
        )
        segments.append(res)

    out_path = out_dir / "stable_segments.json"
    out_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

