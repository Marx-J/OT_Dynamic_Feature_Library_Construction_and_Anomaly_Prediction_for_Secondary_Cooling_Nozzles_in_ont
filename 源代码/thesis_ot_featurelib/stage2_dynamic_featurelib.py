from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .io_nangang import NangangLoadSpec, load_aligned_frame
from .stage1_segments import build_circuit_tags_from_report_catalog
from .utils import ensure_dir


@dataclass(frozen=True)
class WindowSpec:
    # 1 小时
    window_seconds: int = 3600
    # 允许重叠：默认 50% 重叠
    overlap_seconds: int = 1800
    # POC/调参时控制上限（可选）
    max_windows_total: int | None = None


@dataclass(frozen=True)
class TextGenSpec:
    # 如果目标/实际偏差很小，则描述为“稳定/正常”倾向
    deviation_ratio_normal: float = 0.2
    # 用于文字描述的截断精度
    float_round_ndigits: int = 3


@dataclass(frozen=True)
class EmbeddingSpec:
    # 默认用 TF-IDF 向量化（离线、可复现）
    embedding_type: str = "tfidf"
    max_features: int = 4096
    ngram_range: tuple[int, int] = (1, 2)


def _iter_windows_from_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    window_seconds: int,
    overlap_seconds: int,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    step = int(window_seconds - overlap_seconds)
    if step <= 0:
        raise ValueError("overlap_seconds must be smaller than window_seconds")

    cur = start
    win_delta = pd.Timedelta(seconds=window_seconds)
    step_delta = pd.Timedelta(seconds=step)
    while cur + win_delta <= end:
        yield cur, cur + win_delta
        cur = cur + step_delta


def iter_windows_from_stable_segments(
    stable_segments_json_path: str | Path,
    window_spec: WindowSpec,
) -> list[dict[str, Any]]:
    segs = json.loads(Path(stable_segments_json_path).read_text(encoding="utf-8", errors="ignore"))

    # Build per-circuit window lists first.
    by_circuit: dict[str, list[dict[str, Any]]] = {}
    for item in segs:
        circuit_key = item.get("circuit_key")
        if not circuit_key:
            continue
        ranges = item.get("stable_ranges_final") or []
        out: list[dict[str, Any]] = []
        for r in ranges:
            rs = pd.to_datetime(r["start"])
            re = pd.to_datetime(r["end"])
            for wstart, wend in _iter_windows_from_range(
                rs,
                re,
                window_seconds=int(window_spec.window_seconds),
                overlap_seconds=int(window_spec.overlap_seconds),
            ):
                out.append(
                    {
                        "circuit_key": circuit_key,
                        "window_start": str(wstart),
                        "window_end": str(wend),
                    }
                )
        if out:
            by_circuit[str(circuit_key)] = out

    if not by_circuit:
        return []

    max_total = window_spec.max_windows_total
    if not max_total:
        # No cap: return concatenated windows (deterministic circuit order)
        windows: list[dict[str, Any]] = []
        for ck in sorted(by_circuit.keys()):
            windows.extend(by_circuit[ck])
        return windows

    # With cap: sample windows per circuit across the *full time span* (not only from the start),
    # then interleave circuits to keep the final list diverse.
    circuits = sorted(by_circuit.keys())
    n_c = len(circuits)
    budget = int(max_total)

    # Target windows per circuit (ceil so we can fill budget even if some circuits are short).
    k_per = int(np.ceil(budget / max(1, n_c)))

    sampled_by_circuit: dict[str, list[dict[str, Any]]] = {}
    for ck in circuits:
        wlist = by_circuit[ck]
        if len(wlist) <= k_per:
            sampled_by_circuit[ck] = wlist
            continue
        # Evenly spaced indices across the whole list (cover early->late, including stop-time vicinity).
        idxs = np.linspace(0, len(wlist) - 1, num=k_per, dtype=int)
        # de-dup in case of tiny lists
        idxs = np.unique(idxs)
        sampled_by_circuit[ck] = [wlist[int(i)] for i in idxs.tolist()]

    ptr = {ck: 0 for ck in circuits}
    windows_out: list[dict[str, Any]] = []
    while len(windows_out) < budget:
        progressed = False
        for ck in circuits:
            i = ptr[ck]
            if i >= len(sampled_by_circuit[ck]):
                continue
            windows_out.append(sampled_by_circuit[ck][i])
            ptr[ck] = i + 1
            progressed = True
            if len(windows_out) >= budget:
                break
        if not progressed:
            break
    return windows_out


def _safe_float(x: float | np.floating | None, nd: int) -> float | None:
    if x is None:
        return None
    if isinstance(x, (np.floating, float)) and (np.isnan(x) or np.isinf(x)):
        return None
    return round(float(x), nd)


def generate_window_text(
    circuit_key: str,
    df: pd.DataFrame,
    tags: dict[str, str | None],
    spec: TextGenSpec,
) -> str:
    """
    Rule-based 文本生成（离线、可复现）。
    后续你可把这个函数替换成“调用LLM生成描述”的版本。
    """
    flow_act = tags.get("flow_act_tag")
    flow_trg = tags.get("flow_trg_tag")
    valve_pos = tags.get("valve_pos_tag")
    pressure = tags.get("pressure_tag")
    auto_active_tag = tags.get("auto_active_tag")

    parts: list[str] = []
    parts.append(f"回路 {circuit_key} 的二冷喷嘴冷却窗口运行特征如下：")

    def stats_for(tag: str | None) -> tuple[float | None, float | None, float | None]:
        if not tag or tag not in df.columns:
            return None, None, None
        s = df[tag].astype(float)
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            return None, None, None
        mean = float(s.mean())
        std = float(s.std(ddof=0)) if len(s) > 1 else 0.0
        # 斜率用 (last-first)/(n-1)
        if len(s) >= 2:
            slope = float((s.iloc[-1] - s.iloc[0]) / float(len(s) - 1))
        else:
            slope = 0.0
        return mean, std, slope

    act_mean, act_std, act_slope = stats_for(flow_act)
    trg_mean, trg_std, trg_slope = stats_for(flow_trg)
    v_mean, v_std, v_slope = stats_for(valve_pos)
    p_mean, p_std, p_slope = stats_for(pressure)

    if act_mean is not None and trg_mean is not None:
        ratio = float((df[flow_act].astype(float) - df[flow_trg].astype(float)).abs().mean() / (abs(df[flow_trg].astype(float)).replace(0, np.nan).mean() + 1e-9))
        ratio = _safe_float(ratio, spec.float_round_ndigits)
        parts.append(
            f"流量：实际均值≈{_safe_float(act_mean, spec.float_round_ndigits)}，目标均值≈{_safe_float(trg_mean, spec.float_round_ndigits)}；"
            f"实际-目标平均偏差相对比率≈{ratio}。"
        )
    if v_mean is not None:
        parts.append(
            f"阀位：均值≈{_safe_float(v_mean, spec.float_round_ndigits)}，波动标准差≈{_safe_float(v_std, spec.float_round_ndigits)}，"
            f"趋势斜率≈{_safe_float(v_slope, spec.float_round_ndigits)}。"
        )
    if p_mean is not None:
        parts.append(
            f"压力：均值≈{_safe_float(p_mean, spec.float_round_ndigits)}，波动标准差≈{_safe_float(p_std, spec.float_round_ndigits)}，"
            f"趋势斜率≈{_safe_float(p_slope, spec.float_round_ndigits)}。"
        )

    if auto_active_tag and auto_active_tag in df.columns:
        a = df[auto_active_tag].astype(float)
        a = a.replace([np.inf, -np.inf], np.nan).dropna()
        if len(a) > 0:
            true_ratio = float(a.gt(0.5).mean())
            parts.append(f"自动控制投入比率≈{_safe_float(true_ratio, spec.float_round_ndigits)}。")

    # 潜在异常判断（简化、可写进论文）
    if flow_act and flow_trg and flow_act in df.columns and flow_trg in df.columns:
        act = df[flow_act].astype(float)
        trg = df[flow_trg].astype(float)
        diff_ratio = ((act - trg).abs() / (trg.abs().replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
        diff_ratio_mean = float(diff_ratio.mean(skipna=True))
        if math.isnan(diff_ratio_mean):
            abnormal = "数据不足，无法给出异常倾向。"
        elif diff_ratio_mean <= spec.deviation_ratio_normal:
            abnormal = "偏差水平较低，窗口倾向于稳定/正常运行（后续可结合更长窗口趋势判断堵塞或漏水演化）。"
        else:
            # 区分堵塞/漏水倾向：仅基于 ACT-TRG 的符号给出“候选方向”
            # 注：堵塞通常对应 ACT < TRG（冷却能力不足导致实际流量低于目标）；漏水情形下 ACT 与 TRG 的相对关系需结合压力/阀位。
            rel = ((act - trg) / (trg.abs().replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
            rel_mean = float(rel.mean(skipna=True))
            if math.isnan(rel_mean):
                abnormal = "偏差水平较高，可能存在堵塞或漏水诱发的流量不匹配，需要进一步结合阀位与压力趋势判别。"
            elif rel_mean < -spec.deviation_ratio_normal:
                abnormal = "偏差水平较高且 ACT 相对低于 TRG，窗口更倾向于堵塞（后续可结合阀位/压力同步变化进一步确认）。"
            elif rel_mean > spec.deviation_ratio_normal:
                abnormal = "偏差水平较高且 ACT 相对高于 TRG，窗口更倾向于漏水或控制/压力异常（后续可结合阀位/压力同步变化进一步确认）。"
            else:
                abnormal = "偏差水平较高，但 ACT-TRG 的相对偏置不够明确；需要进一步结合阀位与压力趋势判别堵塞/漏水。"
        parts.append("诊断倾向：" + abnormal)
    else:
        parts.append("诊断倾向：相关点位缺失，无法完成偏差判别。")

    return "".join(parts)


def build_dynamic_featurelib_stage2(
    *,
    stable_segments_json_path: str | Path,
    data_dir: str | Path,
    report_txt: str | Path,
    tag_catalog_csv: str | Path | None,
    out_dir: str | Path,
    window_spec: WindowSpec,
    text_spec: TextGenSpec,
    embed_spec: EmbeddingSpec,
    max_windows_debug: int | None = None,
    max_rows_per_tag: int | None = None,
    resample_rule: str = "1s",
    ffill_limit: int | None = 60,
) -> Path:
    out_dir = ensure_dir(out_dir)

    windows = iter_windows_from_stable_segments(stable_segments_json_path, window_spec)
    if max_windows_debug:
        windows = windows[: int(max_windows_debug)]
    if not windows:
        # still write empty meta for pipeline stability
        (out_dir / "windows_texts.csv").write_text("circuit_key,window_start,window_end,description\n", encoding="utf-8")
        return out_dir

    # Circuit->tag mapping
    circuit_tags_map = build_circuit_tags_from_report_catalog(report_txt=report_txt, tag_catalog_csv=tag_catalog_csv)

    # IMPORTANT:
    # Stage1 may have auto-discovered ACT/TRG tags even when the report catalog is incomplete.
    # stable_segments.json includes those tags in `used_tags`. We must override mapping with them.
    segs_raw = json.loads(Path(stable_segments_json_path).read_text(encoding="utf-8", errors="ignore"))
    used_tags_by_circuit: dict[str, dict[str, Any]] = {}
    for item in segs_raw:
        ck = item.get("circuit_key")
        if not ck:
            continue
        ut = item.get("used_tags") or {}
        used_tags_by_circuit[str(ck)] = ut

    # Collect tags required for those windows (only from circuits that appear)
    circuits_in_windows = sorted(set(w["circuit_key"] for w in windows if w.get("circuit_key")))
    tags_needed: set[str] = set()
    for ck in circuits_in_windows:
        ct = circuit_tags_map.get(ck)
        ut = used_tags_by_circuit.get(ck, {})

        # Override act/trg from stage1 (if available)
        flow_act = ut.get("flow_act_tag") if isinstance(ut, dict) else None
        flow_trg = ut.get("flow_trg_tag") if isinstance(ut, dict) else None

        for t in [
            flow_act,
            flow_trg,
            getattr(ct, "valve_pos_tag", None) if ct else None,
            getattr(ct, "pressure_tag", None) if ct else None,
            getattr(ct, "auto_active_tag", None) if ct else None,
        ]:
            if t:
                tags_needed.add(str(t))

    load_spec = NangangLoadSpec(
        data_dir=Path(data_dir),
        include_tags=sorted(tags_needed),
        resample_rule=resample_rule,
        ffill_limit=ffill_limit,
        max_rows_per_tag=max_rows_per_tag,
    )
    df_all, _meta = load_aligned_frame(load_spec)

    # Build window texts
    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for w in windows:
        ck = w["circuit_key"]
        ct = circuit_tags_map.get(ck)
        ut = used_tags_by_circuit.get(ck, {})

        flow_act_override = ut.get("flow_act_tag")
        flow_trg_override = ut.get("flow_trg_tag")

        wstart = pd.to_datetime(w["window_start"])
        wend = pd.to_datetime(w["window_end"])

        # Slice with inclusive start, exclusive end for robustness.
        df_w = df_all.loc[(df_all.index >= wstart) & (df_all.index <= wend)]

        tags = {
            "flow_act_tag": flow_act_override if flow_act_override else (ct.flow_act_tag if ct else None),
            "flow_trg_tag": flow_trg_override if flow_trg_override else (ct.flow_trg_tag if ct else None),
            "valve_pos_tag": ct.valve_pos_tag if ct else None,
            "pressure_tag": ct.pressure_tag if ct else None,
            "auto_active_tag": ct.auto_active_tag if ct else None,
        }
        desc = generate_window_text(ck, df_w, tags, text_spec)
        texts.append(desc)
        rows.append(
            {
                "circuit_key": ck,
                "window_start": w["window_start"],
                "window_end": w["window_end"],
                "description": desc,
            }
        )

    texts_df = pd.DataFrame(rows)
    texts_csv = out_dir / "windows_texts.csv"
    texts_df.to_csv(texts_csv, index=False, encoding="utf-8-sig")

    # Text -> vector (embedding)
    vectorizer_path = out_dir / "tfidf_vectorizer.pkl"
    if embed_spec.embedding_type != "tfidf":
        raise ValueError(f"Unsupported embedding_type={embed_spec.embedding_type!r}")

    vect = TfidfVectorizer(max_features=int(embed_spec.max_features), ngram_range=embed_spec.ngram_range)
    X = vect.fit_transform(texts_df["description"].tolist())
    sparse.save_npz(str(out_dir / "tfidf_vectors.npz"), X)
    joblib.dump(vect, vectorizer_path)

    # Simple search index metadata
    meta = {
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "window_spec": window_spec.__dict__,
        "text_spec": text_spec.__dict__,
        "embed_spec": embed_spec.__dict__,
        "vectorizer_path": str(vectorizer_path),
        "vectors_path": str(out_dir / "tfidf_vectors.npz"),
        "texts_path": str(texts_csv),
    }
    (out_dir / "featurelib_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def search_featurelib_tfidf(
    *,
    featurelib_dir: str | Path,
    query_text: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    featurelib_dir = Path(featurelib_dir)
    vect: TfidfVectorizer = joblib.load(str(featurelib_dir / "tfidf_vectorizer.pkl"))
    X = sparse.load_npz(str(featurelib_dir / "tfidf_vectors.npz"))
    texts_df = pd.read_csv(featurelib_dir / "windows_texts.csv", encoding="utf-8-sig")
    q = vect.transform([query_text])
    sims = cosine_similarity(q, X).ravel()
    idx = np.argsort(-sims)[: int(top_k)]
    out: list[dict[str, Any]] = []
    for i in idx:
        out.append(
            {
                "rank": int(np.where(idx == i)[0][0] + 1),
                "circuit_key": str(texts_df.iloc[i]["circuit_key"]),
                "window_start": str(texts_df.iloc[i]["window_start"]),
                "window_end": str(texts_df.iloc[i]["window_end"]),
                "description": str(texts_df.iloc[i]["description"]),
                "score": float(sims[i]),
            }
        )
    return out

