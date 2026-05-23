from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualitySpec:
    min_rows: int = 5_000
    max_missing_rate: float = 0.6
    max_constant_rate: float = 0.98


def assess_tag_series(s: pd.Series, spec: QualitySpec) -> dict:
    n = int(len(s))
    if n == 0:
        return {"ok": False, "n": 0, "missing_rate": 1.0, "constant_rate": 1.0, "jump_rate": 0.0}

    missing_rate = float(s.isna().mean())
    x = s.dropna()
    if len(x) <= 1:
        constant_rate = 1.0
        jump_rate = 0.0
    else:
        xx = pd.to_numeric(x, errors="coerce")
        if xx.notna().all():
            v = xx.to_numpy(dtype=float, copy=False)
            constant_rate = float(pd.Series(v).nunique(dropna=True) / max(1, len(v)))
            constant_rate = float(1.0 - constant_rate)
            d = np.diff(v)
            jump_rate = float(np.mean(d != 0.0))
        else:
            # fallback for non-numeric tags
            v = x.astype(str).to_numpy(copy=False)
            constant_rate = float(pd.Series(v).nunique(dropna=True) / max(1, len(v)))
            constant_rate = float(1.0 - constant_rate)
            jump_rate = float(np.mean(v[1:] != v[:-1]))

    ok = (n >= int(spec.min_rows)) and (missing_rate <= float(spec.max_missing_rate)) and (constant_rate <= float(spec.max_constant_rate))
    return {
        "ok": bool(ok),
        "n": n,
        "missing_rate": missing_rate,
        "constant_rate": constant_rate,
        "jump_rate": jump_rate,
    }

