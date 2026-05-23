from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io_nangang import NangangLoadSpec, discover_tags, load_single_tag_csv
from .quality import QualitySpec, assess_tag_series


@dataclass(frozen=True)
class DiscoverSpec:
    target_tag: str
    max_rows_per_tag: int = 80_000
    max_tags: int | None = None
    top_k: int = 30
    resample_rule: str = "1s"
    ffill_limit: int = 60
    quality: QualitySpec = QualitySpec(min_rows=5_000)


def _align_two(a: pd.Series, b: pd.Series, resample_rule: str, ffill_limit: int) -> tuple[pd.Series, pd.Series]:
    # protect against duplicate timestamps in raw csv
    if a.index.has_duplicates:
        a = a.groupby(level=0).mean()
    if b.index.has_duplicates:
        b = b.groupby(level=0).mean()

    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).sort_index()
    df = df.resample(resample_rule).mean().ffill(limit=ffill_limit)
    return df["a"], df["b"]


def discover_related_tags(data_dir: str | Path, spec: DiscoverSpec) -> pd.DataFrame:
    data_dir = Path(data_dir)
    tags = discover_tags(data_dir)
    if spec.max_tags is not None:
        tags = tags[: spec.max_tags]

    if spec.target_tag not in tags:
        raise ValueError(f"target_tag {spec.target_tag} not found in {data_dir}")

    target_s, _ = load_single_tag_csv(data_dir / f"{spec.target_tag}.csv", nrows=spec.max_rows_per_tag)

    rows = []
    for tag in tags:
        if tag == spec.target_tag:
            continue
        s, _ = load_single_tag_csv(data_dir / f"{tag}.csv", nrows=spec.max_rows_per_tag)
        qa = assess_tag_series(s, spec.quality)
        if not qa["ok"]:
            continue
        a, b = _align_two(target_s, s, spec.resample_rule, spec.ffill_limit)
        m = a.notna() & b.notna()
        if m.sum() < 200:
            continue
        corr = float(np.corrcoef(a[m].to_numpy(), b[m].to_numpy())[0, 1])

        # quick lag scan around +/-60s
        best_lag = 0
        best_corr = corr
        for lag in (-60, -30, -10, -5, 5, 10, 30, 60):
            bb = b.shift(lag)
            mm = a.notna() & bb.notna()
            if mm.sum() < 200:
                continue
            c = float(np.corrcoef(a[mm].to_numpy(), bb[mm].to_numpy())[0, 1])
            if abs(c) > abs(best_corr):
                best_corr = c
                best_lag = lag

        rows.append(
            {
                "tag": tag,
                "corr": corr,
                "best_corr": best_corr,
                "best_lag_s": best_lag,
                "missing_rate": qa["missing_rate"],
                "constant_rate": qa["constant_rate"],
                "jump_rate": qa["jump_rate"],
                "n": qa["n"],
            }
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.sort_values("best_corr", key=lambda s: s.abs(), ascending=False).head(int(spec.top_k))
    return df


def discover_liquid_level_candidates(data_dir: str | Path, *, max_rows_per_tag: int = 80_000, top_k: int = 20) -> pd.DataFrame:
    """
    Heuristic: liquid level often around 140-160, numeric, not too constant.
    We don't know the tag name, so we rank by:
      - median close to 150
      - IQR not too small
      - values frequently in [140,160]
    """
    data_dir = Path(data_dir)
    tags = discover_tags(data_dir)
    rows = []
    for tag in tags:
        s, _ = load_single_tag_csv(data_dir / f"{tag}.csv", nrows=max_rows_per_tag)
        x = s.dropna().astype(float)
        if len(x) < 2000:
            continue
        med = float(x.median())
        q25 = float(x.quantile(0.25))
        q75 = float(x.quantile(0.75))
        iqr = q75 - q25
        in_band = float(x.between(140, 160).mean())
        # score: close to 150 + has band occupancy + moderate variability
        score = -abs(med - 150.0) + 20.0 * in_band + min(5.0, iqr / 5.0)
        rows.append({"tag": tag, "median": med, "iqr": iqr, "in_140_160_rate": in_band, "score": score, "n": int(len(x))})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("score", ascending=False).head(int(top_k))


def suggest_deviation_pair(
    data_dir: str | Path,
    *,
    trg_tag: str,
    max_rows_per_tag: int = 80_000,
    top_k_related: int = 60,
) -> dict:
    """
    Suggest an (actual, target) pair for deviation20pct labeling when only TRG tag is known.
    Strategy:
      - find top correlated tags with TRG
      - compute scale/MAE/deviation rate
      - choose a tag with similar scale (mean ratio within [0.5, 2])
        and deviation20_rate in [1%, 30%] as ACT candidate.
    """
    rel = discover_related_tags(
        data_dir,
        DiscoverSpec(target_tag=trg_tag, max_rows_per_tag=max_rows_per_tag, top_k=top_k_related),
    )
    if rel.empty:
        return {"trg_tag": trg_tag, "act_tag": None, "reason": "no_related_tags"}

    data_dir = Path(data_dir)

    def load(tag: str) -> pd.Series:
        s, _ = load_single_tag_csv(data_dir / f"{tag}.csv", nrows=max_rows_per_tag)
        if s.index.has_duplicates:
            s = s.groupby(level=0).mean()
        return s

    t = load(trg_tag)
    candidates = []
    for tag in rel["tag"].astype(str).tolist():
        s = load(tag)
        a, b = _align_two(t, s, "1s", 60)
        m = a.notna() & b.notna()
        if m.sum() < 2000:
            continue
        aa = a[m].astype(float)
        bb = b[m].astype(float)
        t_mean = float(aa.mean())
        s_mean = float(bb.mean())
        ratio = float(s_mean / (t_mean + 1e-9))
        diff = (bb - aa).abs()
        dev20 = float((diff > 0.2 * aa.abs()).mean())
        mae = float(diff.mean())
        candidates.append({"tag": tag, "mean_ratio_to_trg": ratio, "dev20_rate": dev20, "mae": mae, "corr": float(rel.loc[rel["tag"] == tag, "best_corr"].iloc[0])})

    if not candidates:
        return {"trg_tag": trg_tag, "act_tag": None, "reason": "no_candidates_after_alignment"}

    cand_df = pd.DataFrame(candidates)
    # filter by scale similarity and a non-trivial but not crazy dev rate
    filt = cand_df[(cand_df["mean_ratio_to_trg"].between(0.5, 2.0)) & (cand_df["dev20_rate"].between(0.01, 0.30))]
    if len(filt) == 0:
        # fallback: closest by MAE among scale-similar
        filt = cand_df[cand_df["mean_ratio_to_trg"].between(0.5, 2.0)]
    if len(filt) == 0:
        filt = cand_df

    best = filt.sort_values(["dev20_rate", "mae"], ascending=[False, True]).iloc[0].to_dict()
    return {"trg_tag": trg_tag, "act_tag": str(best["tag"]), "act_stats": best, "n_candidates": int(len(cand_df))}

