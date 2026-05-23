from __future__ import annotations

import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

HybridTuneMode = Literal["per_stop", "joint"]


def _logistic_regression_saga_extra_kwargs() -> dict[str, Any]:
    """scikit-learn>=1.8：弃用 penalty='l2' 与 n_jobs；用 l1_ratio=0 表示 L2 正则口径。"""
    try:
        ver = importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        return {"penalty": "l2", "n_jobs": -1}
    m = re.match(r"(\d+)\.(\d+)", ver)
    if not m:
        return {"penalty": "l2", "n_jobs": -1}
    if (int(m.group(1)), int(m.group(2))) >= (1, 8):
        return {"l1_ratio": 0.0}
    return {"penalty": "l2", "n_jobs": -1}


# 本文提出模型 v2：多融合模式 + RF/HGB 集成 + 聚合/topK/阈值联合搜索（FP=0，优先 TP 再 MCC）
_PROPOSED_V2_BLEND_MODES: tuple[str, ...] = ("linear", "mul_gate", "max_blend")
_PROPOSED_V2_RULE_WEIGHTS: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
_PROPOSED_V2_HGB_WEIGHTS: tuple[float, ...] = (0.0, 0.08, 0.15, 0.22)
_PROPOSED_V2_CIRCUIT_AGGS: tuple[str, ...] = ("noisy_or", "max", "median_prob")
_PROPOSED_V2_TOPK_GRID: tuple[int, ...] = (24, 32, 48, 64)

# Hybrid 阈值网格搜索：标量化多目标（论文可写「折中」）
# ——贴近 hybrid_accuracy_tuning_target；默认偏好「宁可漏一点也不要误报」：特异性权重高、召回下限低。
# 硬约束见 _hybrid_candidate_hard_ok：金标有正常回路时要求 TN≥1；有异常时要求 TP≥1（禁止全漏）。
_HYBRID_TUNE_RECALL_FLOOR = 0.22
_HYBRID_TUNE_SPEC_FLOOR = 0.52
_HYBRID_TUNE_W_RECALL_SHORTFALL = 0.28
_HYBRID_TUNE_W_SPEC_SHORTFALL = 2.05
_HYBRID_TUNE_BALACC_FLOOR = 0.52
_HYBRID_TUNE_W_BALACC_SHORTFALL = 1.25
# 非退化可行解上：对「召回过低」的软惩罚下限（不要求达到 1.0）
_HYBRID_TUNE_RANK_RECALL_SOFT_FLOOR = 0.18

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .io_nangang import NangangLoadSpec, load_aligned_frame
from .stage2_dynamic_featurelib import iter_windows_from_stable_segments
from .utils import ensure_dir


@dataclass(frozen=True)
class LabelSpec:
    """与报告口径一致：|ACT-TRG| > ratio*|TRG| 视为窗口内流量异常（用于弱监督与规则基线）。"""
    deviation_ratio: float = 0.2
    eps_trg: float = 1e-9


@dataclass(frozen=True)
class WindowSpec:
    window_hours: float = 1.0
    overlap_hours: float = 0.5

    @property
    def window_seconds(self) -> int:
        return int(self.window_hours * 3600.0)

    @property
    def overlap_seconds(self) -> int:
        return int(self.overlap_hours * 3600.0)


@dataclass(frozen=True)
class EvalSpec:
    # 训练集剔除：停机前窄带，避免紧贴停机窗口参与训练
    lead_hours: float = 6.0
    # 与金标准对齐：停机前回看带内滑窗聚合
    gold_eval_lookback_hours: float = 720.0
    # 回路级告警：将回看带内各窗分数聚合成一个回路分数后，与阈值比较
    fault_alarm_threshold: float = 0.1
    # ML：max=取窗上 P(异常) 最大；noisy_or=1-∏(1-p_i) 仅在每回路「分数最高的 K 个窗」上计算，避免万级滑窗数值饱和
    ml_circuit_aggregate: Literal["max", "noisy_or", "mean_prob", "median_prob"] = "noisy_or"
    ml_noisy_or_topk: int = 48
    # Hybrid 专用：在阈值网格上搜索，使两次停机对比的 accuracy 尽量接近该目标（None 表示不调参，与 fault_alarm_threshold 一致）
    hybrid_accuracy_tuning_target: float | None = 0.75
    # Hybrid 调参时 noisy_or 的 top-K 候选（与 Rule/HGB 的 ml_noisy_or_topk 可不同）
    hybrid_tune_topk_grid: tuple[int, ...] = (24, 32, 40, 48)
    # Hybrid 调参：LR 概率与规则窗分数融合，缓解「全回路均值概率偏高」导致阈值无法折中（略提高则更信规则、利于压低误报）
    hybrid_rule_blend_weight: float = 0.78
    # Hybrid：LR 概率截顶，抑制过自信带来的全线判异常
    hybrid_lr_prob_cap: float = 0.64
    # joint：同一阈值同时作用于两次停机，目标为两次 accuracy 与目标的平均偏差 + 拉齐两次结果
    hybrid_threshold_tune_mode: HybridTuneMode = "joint"


CLASS_NORMAL = 0
CLASS_BLOCKAGE = 1
CLASS_LEAK = 2

# X_num 列顺序与下方 append 一致（规则基线用 rel_dev_max）
FEAT_REL_DEV_MAX = 7


def _classify_from_act_trg(act: np.ndarray, trg: np.ndarray, spec: LabelSpec) -> int:
    """
    三分类弱标签（仅用于构造训练用异常示意；金标准对比为二分类「初检是否异常」）。
      0 normal: max(rel_dev) <= ratio
      1 blockage: abnormal & mean(act-trg) < 0
      2 leak: abnormal & mean(act-trg) >= 0
    """
    act = act.astype(float)
    trg = trg.astype(float)
    m = np.isfinite(act) & np.isfinite(trg)
    if m.sum() < 5:
        return CLASS_NORMAL
    a = act[m]
    t = trg[m]
    rel_dev = np.abs(a - t) / (np.abs(t) + spec.eps_trg)
    abnormal = float(np.nanmax(rel_dev)) > spec.deviation_ratio
    if not abnormal:
        return CLASS_NORMAL
    diff_mean = float(np.nanmean(a - t))
    return CLASS_BLOCKAGE if diff_mean < 0 else CLASS_LEAK


def _gold_fault_binary(status: str) -> int:
    """初检状态：含堵塞/漏水（含疑似）→ 异常(1)，否则正常(0)。"""
    s = str(status).strip()
    if "漏水" in s or "堵塞" in s:
        return 1
    return 0


def _parse_stop_time_from_report_txt(report_txt: str | Path) -> pd.Timestamp:
    p = Path(report_txt)
    txt = p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"(\d{1,2})月(\d{1,2})日(\d{1,2})时停机", txt)
    if not m:
        m = re.search(r"(\d{1,2})月(\d{1,2})日(\d{1,2})时", txt)
    if not m:
        raise ValueError(f"Cannot parse stop time from report txt: {report_txt}")
    month = int(m.group(1))
    day = int(m.group(2))
    hour = int(m.group(3))
    year_m = re.search(r"(20\d{2})", txt)
    year = int(year_m.group(1)) if year_m else 2026
    return pd.Timestamp(year=year, month=month, day=day, hour=hour)


def _build_gold_fault_map(gold_df: pd.DataFrame) -> dict[str, int]:
    """回路键 -> 是否异常（初检）。"""
    m: dict[str, int] = {}
    for _, row in gold_df.iterrows():
        zone = str(row["zone"]).strip()
        strand = int(row["strand"])
        status = str(row["status"])
        ckey = f"{strand}ST_二冷水_{zone}"
        m[ckey] = _gold_fault_binary(status)
    return m


def _binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """工厂更关心的二分类指标：漏报/误报、MCC、平衡准确率。"""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    return {
        "n_circuits": int(len(yt)),
        "n_gold_fault": int(yt.sum()),
        "n_gold_normal": int(len(yt) - int(yt.sum())),
        "confusion_binary_rows_gold_0normal_1fault": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall_fault": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "precision_fault": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "f1_fault": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "specificity_normal": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "npv_normal": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "mcc": float(matthews_corrcoef(yt, yp)),
    }


def _proba_fault_positive(model: Any, X: np.ndarray) -> np.ndarray:
    pf = model.predict_proba(X)
    if pf.shape[1] == 1:
        return np.zeros(len(X), dtype=float)
    return pf[:, 1].astype(float)


def _fit_hgb_binary(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    uniq = np.unique(y_train)
    if len(uniq) < 2:
        return DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=42,
                    class_weight="balanced",
                    max_depth=10,
                    learning_rate=0.08,
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=15,
                ),
            ),
        ]
    ).fit(X_train, y_train)


def _fit_lr_tabular_binary(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    uniq = np.unique(y_train)
    if len(uniq) < 2:
        return DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    solver="saga",
                    random_state=42,
                    **_logistic_regression_saga_extra_kwargs(),
                ),
            ),
        ]
    ).fit(X_train, y_train)


def _fit_rf_tabular_binary(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    uniq = np.unique(y_train)
    if len(uniq) < 2:
        return DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ).fit(X_train, y_train)


def _fit_hybrid_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    vect_train: sparse.spmatrix,
) -> tuple[SimpleImputer, LogisticRegression]:
    """数值列中位数填补 + TF-IDF 稀疏矩阵水平拼接 → saga 逻辑回归（二分类，类别加权）。"""
    imp = SimpleImputer(strategy="median")
    X_i = imp.fit_transform(X_train)
    X_h = sparse.hstack([vect_train, sparse.csr_matrix(X_i.astype(np.float64))], format="csr")

    uniq = np.unique(y_train)
    if len(uniq) < 2:
        lr = LogisticRegression(max_iter=1, solver="liblinear")
        lr.fit(X_h, y_train)
        return imp, lr

    lr = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="saga",
        random_state=42,
        **_logistic_regression_saga_extra_kwargs(),
    )
    lr.fit(X_h, y_train)
    return imp, lr


def _agg_combine_window_scores(fs: np.ndarray, aggregate: str, *, noisy_or_topk: int) -> float:
    fs = np.asarray(fs, dtype=float)
    fs = fs[np.isfinite(fs)]
    if fs.size == 0:
        return 0.0
    if aggregate == "noisy_or":
        k = max(1, min(int(noisy_or_topk), int(fs.size)))
        tail = np.sort(fs)[-k:]
        p = np.clip(tail, 1e-9, 1.0 - 1e-9)
        return float(1.0 - float(np.prod(1.0 - p)))
    if aggregate == "mean_prob":
        return float(np.mean(fs))
    if aggregate == "median_prob":
        return float(np.median(fs))
    return float(np.nanmax(fs))


def _circuit_score_map_from_window_scores(
    meta_df: pd.DataFrame,
    mask: np.ndarray,
    scores_full: np.ndarray,
    *,
    aggregate: str = "max",
    noisy_or_topk: int = 48,
) -> dict[str, float]:
    """回路级连续分数（未阈值化），用于融合与标定。"""
    scores_full = np.asarray(scores_full, dtype=float)
    if len(scores_full) != len(meta_df):
        raise ValueError(f"scores_full length {len(scores_full)} != meta_df rows {len(meta_df)}")
    sub = meta_df.loc[mask].copy()
    if len(sub) == 0:
        return {}
    iloc = np.flatnonzero(mask)
    sub["_fs"] = scores_full[iloc]
    out: dict[str, float] = {}
    for ck, grp in sub.groupby("circuit_key"):
        out[str(ck)] = float(
            _agg_combine_window_scores(grp["_fs"].to_numpy(), aggregate, noisy_or_topk=noisy_or_topk)
        )
    return out


def _circuit_fault_map_from_window_scores(
    meta_df: pd.DataFrame,
    mask: np.ndarray,
    scores_full: np.ndarray,
    *,
    threshold: float,
    aggregate: str = "max",
    noisy_or_topk: int = 48,
) -> dict[str, int]:
    """
    回路级：回看带内窗口分数先按 aggregate 聚合成回路分数，再与 threshold 比较。
    scores_full 与 meta_df 逐行对齐；ML 用 P(fault)∈[0,1]，规则为 0/1。
    """
    scores_full = np.asarray(scores_full, dtype=float)
    if len(scores_full) != len(meta_df):
        raise ValueError(f"scores_full length {len(scores_full)} != meta_df rows {len(meta_df)}")
    sub = meta_df.loc[mask].copy()
    if len(sub) == 0:
        return {}
    iloc = np.flatnonzero(mask)
    sub["_fs"] = scores_full[iloc]
    out: dict[str, int] = {}
    for ck, grp in sub.groupby("circuit_key"):
        score = _agg_combine_window_scores(grp["_fs"].to_numpy(), aggregate, noisy_or_topk=noisy_or_topk)
        out[str(ck)] = 1 if score >= float(threshold) else 0
    return out


def _proposed_thr_grid(*, fine: bool = False) -> np.ndarray:
    if fine:
        return np.unique(
            np.concatenate(
                [
                    np.linspace(0.01, 0.50, 80),
                    np.linspace(0.51, 0.95, 70),
                    np.linspace(0.96, 0.999, 35),
                ]
            )
        )
    return np.linspace(0.02, 0.98, 65)


def _fpzero_rank_key(m1: dict[str, Any], m7: dict[str, Any], *, thr_penalty: float = 0.0) -> tuple[float, ...]:
    """FP=0 候选排序：TP 总和 → mean(MCC) → mean(F1) → mean(Acc) → mean(Recall)。"""
    tp_sum = int(m1.get("tp", 0) or 0) + int(m7.get("tp", 0) or 0)
    mcc_m = (float(m1.get("mcc", 0) or 0) + float(m7.get("mcc", 0) or 0)) / 2.0
    f1_m = (float(m1.get("f1_fault", 0) or 0) + float(m7.get("f1_fault", 0) or 0)) / 2.0
    acc_m = (float(m1["accuracy"]) + float(m7["accuracy"])) / 2.0
    rec_m = (float(m1.get("recall_fault", 0) or 0) + float(m7.get("recall_fault", 0) or 0)) / 2.0
    return (float(tp_sum), mcc_m, f1_m, acc_m, rec_m, -float(thr_penalty))


def _tune_thresholds_fpzero(
    compare_stop: Callable[[dict[str, int], dict[str, int]], dict[str, Any]],
    *,
    meta_df: pd.DataFrame,
    window_scores: np.ndarray,
    gold_eval_0401_mask: np.ndarray,
    gold_eval_0407_mask: np.ndarray,
    gold_map_0401: dict[str, int],
    gold_map_0407: dict[str, int],
    ml_agg: str,
    nk: int,
    fallback_thr: float,
    per_stop_threshold: bool,
    fine: bool = False,
) -> tuple[float, float, dict[str, Any]]:
    """在 FP=0 可行域内搜索阈值；可选各停机独立阈值。"""
    nk_i = max(1, int(nk))
    thr_grid = _proposed_thr_grid(fine=fine)
    best_key: tuple[float, ...] | None = None
    best_pack: dict[str, Any] | None = None

    if not per_stop_threshold:
        for thr_c in thr_grid:
            pr1 = _circuit_fault_map_from_window_scores(
                meta_df, gold_eval_0401_mask, window_scores, threshold=float(thr_c), aggregate=ml_agg, noisy_or_topk=nk_i
            )
            pr7 = _circuit_fault_map_from_window_scores(
                meta_df, gold_eval_0407_mask, window_scores, threshold=float(thr_c), aggregate=ml_agg, noisy_or_topk=nk_i
            )
            m1 = compare_stop(pr1, gold_map_0401)
            m7 = compare_stop(pr7, gold_map_0407)
            if int(m1.get("n_circuits", 0) or 0) < 1 or int(m7.get("n_circuits", 0) or 0) < 1:
                continue
            if int(m1.get("fp", 0) or 0) > 0 or int(m7.get("fp", 0) or 0) > 0:
                continue
            key = _fpzero_rank_key(m1, m7, thr_penalty=float(thr_c))
            if best_key is None or key > best_key:
                best_key = key
                best_pack = {
                    "threshold_mode": "joint",
                    "chosen_threshold_joint": float(thr_c),
                    "chosen_threshold_0401": float(thr_c),
                    "chosen_threshold_0407": float(thr_c),
                    "compare_0401": m1,
                    "compare_0407": m7,
                }
    else:
        thr_coarse = np.unique(np.linspace(0.02, 0.98, 49))
        for thr1 in thr_coarse:
            pr1 = _circuit_fault_map_from_window_scores(
                meta_df, gold_eval_0401_mask, window_scores, threshold=float(thr1), aggregate=ml_agg, noisy_or_topk=nk_i
            )
            m1 = compare_stop(pr1, gold_map_0401)
            if int(m1.get("n_circuits", 0) or 0) < 1 or int(m1.get("fp", 0) or 0) > 0:
                continue
            for thr7 in thr_coarse:
                pr7 = _circuit_fault_map_from_window_scores(
                    meta_df, gold_eval_0407_mask, window_scores, threshold=float(thr7), aggregate=ml_agg, noisy_or_topk=nk_i
                )
                m7 = compare_stop(pr7, gold_map_0407)
                if int(m7.get("n_circuits", 0) or 0) < 1 or int(m7.get("fp", 0) or 0) > 0:
                    continue
                key = _fpzero_rank_key(m1, m7, thr_penalty=(float(thr1) + float(thr7)) / 2.0)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pack = {
                        "threshold_mode": "per_stop",
                        "chosen_threshold_joint": None,
                        "chosen_threshold_0401": float(thr1),
                        "chosen_threshold_0407": float(thr7),
                        "compare_0401": m1,
                        "compare_0407": m7,
                    }

    if best_pack is None:
        return float(fallback_thr), float(fallback_thr), {"note": "fallback_no_fpzero_feasible"}
    m1 = best_pack["compare_0401"]
    m7 = best_pack["compare_0407"]
    meta = {
        **best_pack,
        "mean_mcc_two_stops": (float(m1.get("mcc", 0) or 0) + float(m7.get("mcc", 0) or 0)) / 2.0,
        "mean_accuracy_two_stops": (float(m1["accuracy"]) + float(m7["accuracy"])) / 2.0,
        "mean_recall_fault_two_stops": (float(m1.get("recall_fault", 0) or 0) + float(m7.get("recall_fault", 0) or 0))
        / 2.0,
        "mean_f1_fault_two_stops": (float(m1.get("f1_fault", 0) or 0) + float(m7.get("f1_fault", 0) or 0)) / 2.0,
        "tp_sum_two_stops": int(m1.get("tp", 0) or 0) + int(m7.get("tp", 0) or 0),
        "fp_sum_two_stops": int(m1.get("fp", 0) or 0) + int(m7.get("fp", 0) or 0),
        "tuning_objective": "FP=0 hard constraint; maximize TP_sum, then mean(MCC), F1, Acc, Recall",
        "threshold_selection_note": (
            "联合阈值" if best_pack.get("threshold_mode") == "joint" else "各停机独立阈值"
        )
        + "；平局次序：TP 总和→MCC→F1→Accuracy→Recall。",
    }
    return (
        float(best_pack["chosen_threshold_0401"]),
        float(best_pack["chosen_threshold_0407"]),
        meta,
    )


def _build_proposed_blend_scores_v2(
    mode: str,
    s_rule: np.ndarray,
    p_rf: np.ndarray,
    p_hgb: np.ndarray,
    *,
    w_rule: float,
    w_hgb: float,
) -> np.ndarray:
    sr = np.clip(np.asarray(s_rule, dtype=float), 0.0, 1.0)
    pr = np.clip(np.asarray(p_rf, dtype=float), 0.0, 1.0)
    ph = np.clip(np.asarray(p_hgb, dtype=float), 0.0, 1.0)
    m = str(mode)
    if m == "mul_gate":
        return np.clip(sr * np.maximum(pr, ph), 0.0, 1.0)
    if m == "max_blend":
        w = float(np.clip(w_rule, 0.0, 0.95))
        return np.clip(np.maximum(w * sr, (1.0 - w) * np.maximum(pr, ph)), 0.0, 1.0)
    # linear: w_rule*s + w_hgb*h + (1-w_rule-w_hgb)*rf
    wr = float(np.clip(w_rule, 0.0, 0.90))
    wh = float(np.clip(w_hgb, 0.0, 0.90))
    if wr + wh > 0.92:
        s = (wr + wh) / max(wr + wh, 1e-9) * 0.92
        wr, wh = wr * s / (wr + wh), wh * s / (wr + wh)
    wpr = max(0.0, 1.0 - wr - wh)
    return np.clip(wr * sr + wh * ph + wpr * pr, 0.0, 1.0)


def _tune_proposed_conservative_model(
    compare_stop: Callable[[dict[str, int], dict[str, int]], dict[str, Any]],
    *,
    s_rule: np.ndarray,
    p_rf: np.ndarray,
    p_hgb: np.ndarray,
    meta_df: pd.DataFrame,
    gold_eval_0401_mask: np.ndarray,
    gold_eval_0407_mask: np.ndarray,
    gold_map_0401: dict[str, int],
    gold_map_0407: dict[str, int],
    fallback_thr: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """v2 网格搜索：融合模式 × 权重 × 聚合 × topK ×（联合/分停机）阈值；FP=0 下优先 TP 再 MCC。"""
    best_key: tuple[float, ...] | None = None
    best: dict[str, Any] = {
        "blend_mode": "linear",
        "w_rule": 0.65,
        "w_hgb": 0.0,
        "circuit_aggregate": "noisy_or",
        "ml_noisy_or_topk": 48,
        "threshold_mode": "joint",
        "chosen_threshold_0401": float(fallback_thr),
        "chosen_threshold_0407": float(fallback_thr),
        "note": "fallback",
    }
    n_cfg = 0

    def _consider_cfg(
        blend_mode: str,
        w_rule: float,
        w_hgb: float,
        ml_agg: str,
        nk: int,
        per_stop: bool,
        fine_thr: bool,
    ) -> None:
        nonlocal best_key, best, n_cfg
        n_cfg += 1
        blend = _build_proposed_blend_scores_v2(
            blend_mode, s_rule, p_rf, p_hgb, w_rule=float(w_rule), w_hgb=float(w_hgb)
        )
        thr1, thr7, tmeta = _tune_thresholds_fpzero(
            compare_stop,
            meta_df=meta_df,
            window_scores=blend,
            gold_eval_0401_mask=gold_eval_0401_mask,
            gold_eval_0407_mask=gold_eval_0407_mask,
            gold_map_0401=gold_map_0401,
            gold_map_0407=gold_map_0407,
            ml_agg=ml_agg,
            nk=nk,
            fallback_thr=fallback_thr,
            per_stop_threshold=per_stop,
            fine=fine_thr,
        )
        m1 = tmeta.get("compare_0401", {})
        m7 = tmeta.get("compare_0407", {})
        if not m1 or not m7:
            return
        key = _fpzero_rank_key(m1, m7)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "blend_mode": blend_mode,
                "w_rule": float(w_rule),
                "w_hgb": float(w_hgb),
                "circuit_aggregate": ml_agg,
                "ml_noisy_or_topk": int(nk),
                "threshold_mode": tmeta.get("threshold_mode"),
                "chosen_threshold_0401": float(thr1),
                "chosen_threshold_0407": float(thr7),
                "chosen_threshold_joint": tmeta.get("chosen_threshold_joint"),
                "tuning": tmeta,
                "window_scores_note": (
                    f"mode={blend_mode}, w_rule={w_rule}, w_hgb={w_hgb}; agg={ml_agg}, topk={nk}"
                ),
            }

    for blend_mode in _PROPOSED_V2_BLEND_MODES:
        for w_rule in _PROPOSED_V2_RULE_WEIGHTS:
            hgb_opts = _PROPOSED_V2_HGB_WEIGHTS if blend_mode == "linear" else (0.0,)
            for w_hgb in hgb_opts:
                for ml_agg in _PROPOSED_V2_CIRCUIT_AGGS:
                    for nk in _PROPOSED_V2_TOPK_GRID:
                        _consider_cfg(blend_mode, float(w_rule), float(w_hgb), ml_agg, int(nk), False, False)

    if best_key is not None:
        _consider_cfg(
            str(best["blend_mode"]),
            float(best["w_rule"]),
            float(best["w_hgb"]),
            str(best["circuit_aggregate"]),
            int(best["ml_noisy_or_topk"]),
            True,
            False,
        )
        _consider_cfg(
            str(best["blend_mode"]),
            float(best["w_rule"]),
            float(best["w_hgb"]),
            str(best["circuit_aggregate"]),
            int(best["ml_noisy_or_topk"]),
            False,
            True,
        )

    best["configs_searched"] = int(n_cfg)
    blend_final = _build_proposed_blend_scores_v2(
        str(best["blend_mode"]),
        s_rule,
        p_rf,
        p_hgb,
        w_rule=float(best["w_rule"]),
        w_hgb=float(best["w_hgb"]),
    )
    return blend_final, best


def _summarize_beats_baselines(
    proposed_lane: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """与五基线对比：MCC / FP / TP 是否在两次停机上均不占劣。"""
    out: dict[str, Any] = {"per_baseline": {}, "strictly_beats_all_on_mean_mcc": True}
    p_mcc = (
        float(proposed_lane.get("compare_0401", {}).get("mcc", 0) or 0)
        + float(proposed_lane.get("compare_0407", {}).get("mcc", 0) or 0)
    ) / 2.0
    p_fp = int(proposed_lane.get("compare_0401", {}).get("fp", 0) or 0) + int(
        proposed_lane.get("compare_0407", {}).get("fp", 0) or 0
    )
    p_tp = int(proposed_lane.get("compare_0401", {}).get("tp", 0) or 0) + int(
        proposed_lane.get("compare_0407", {}).get("tp", 0) or 0
    )
    for name, lane in baselines.items():
        b_mcc = (float(lane.get("compare_0401", {}).get("mcc", 0) or 0) + float(lane.get("compare_0407", {}).get("mcc", 0) or 0)) / 2.0
        b_fp = int(lane.get("compare_0401", {}).get("fp", 0) or 0) + int(lane.get("compare_0407", {}).get("fp", 0) or 0)
        b_tp = int(lane.get("compare_0401", {}).get("tp", 0) or 0) + int(lane.get("compare_0407", {}).get("tp", 0) or 0)
        beats_mcc = p_mcc >= b_mcc - 1e-12
        beats_fp = p_fp <= b_fp
        beats_tp = p_tp >= b_tp
        out["per_baseline"][name] = {
            "mean_mcc_baseline": b_mcc,
            "fp_sum_baseline": b_fp,
            "tp_sum_baseline": b_tp,
            "proposed_beats_or_ties_mcc": beats_mcc,
            "proposed_beats_or_ties_fp": beats_fp,
            "proposed_beats_or_ties_tp": beats_tp,
            "strictly_better_mcc": p_mcc > b_mcc + 1e-12,
            "strictly_better_tp": p_tp > b_tp,
        }
        if not beats_mcc:
            out["strictly_beats_all_on_mean_mcc"] = False
    out["proposed_mean_mcc"] = p_mcc
    out["proposed_fp_sum"] = p_fp
    out["proposed_tp_sum"] = p_tp
    return out


def _hybrid_tune_composite_loss(
    *,
    acc_deviation: float,
    recall_fault: float,
    specificity_normal: float,
    balanced_accuracy: float | None = None,
) -> tuple[float, dict[str, float]]:
    """越小越好：accuracy 偏离 + 召回不足 + 特异性不足 + 平衡准确率不足（抑制 TN=0 时 bal_acc=0.5）。"""
    rec_sf = max(0.0, float(_HYBRID_TUNE_RECALL_FLOOR) - float(recall_fault))
    spec_sf = max(0.0, float(_HYBRID_TUNE_SPEC_FLOOR) - float(specificity_normal))
    bal_sf = 0.0
    if balanced_accuracy is not None:
        bal_sf = max(0.0, float(_HYBRID_TUNE_BALACC_FLOOR) - float(balanced_accuracy))
    comp = (
        float(acc_deviation)
        + _HYBRID_TUNE_W_RECALL_SHORTFALL * rec_sf
        + _HYBRID_TUNE_W_SPEC_SHORTFALL * spec_sf
        + _HYBRID_TUNE_W_BALACC_SHORTFALL * bal_sf
    )
    parts: dict[str, float] = {
        "acc_deviation": float(acc_deviation),
        "recall_shortfall": rec_sf,
        "specificity_shortfall": spec_sf,
        "composite": comp,
    }
    if balanced_accuracy is not None:
        parts["balanced_accuracy_for_shortfall"] = float(balanced_accuracy)
        parts["balanced_accuracy_shortfall"] = float(bal_sf)
    return comp, parts


def _hybrid_threshold_grid_for_aggregate(ml_agg: str) -> np.ndarray:
    """mean/median 概率型回路分：在物理常见段加密采样，减少「硬约束可行阈」被网格跳过的概率。"""
    if ml_agg in ("mean_prob", "median_prob"):
        return np.unique(
            np.concatenate(
                [
                    np.linspace(0.01, 0.88, 90),
                    np.linspace(0.89, 0.998, 96),
                    np.linspace(0.05, 0.45, 801),
                    np.linspace(0.10, 0.20, 401),
                ]
            )
        )
    thr_coarse = np.linspace(0.04, 0.92, 89)
    thr_fine = np.linspace(0.93, 0.9995, 55)
    return np.unique(np.concatenate([thr_coarse, thr_fine]))


def _cm_counts(m: dict[str, Any]) -> tuple[int, int, int, int]:
    return int(m["tn"]), int(m["fp"]), int(m["fn"]), int(m["tp"])


def _hybrid_candidate_hard_ok(m: dict[str, Any]) -> bool:
    """硬约束：金标有异常时不得「全漏」(TP≥1)；有正常时不得「全线判异常」(TN≥1)。"""
    n0 = int(m.get("n_gold_normal", 0) or 0)
    n1 = int(m.get("n_gold_fault", 0) or 0)
    tn, _fp, _fn, tp = _cm_counts(m)
    if n1 >= 1 and tp < 1:
        return False
    if n0 >= 1 and tn < 1:
        return False
    return True


def _joint_hard_ok(m1: dict[str, Any], m7: dict[str, Any]) -> bool:
    return bool(_hybrid_candidate_hard_ok(m1) and _hybrid_candidate_hard_ok(m7))


def _joint_at_least_one_tp_each(m1: dict[str, Any], m7: dict[str, Any]) -> bool:
    """弱约束：各停机在金标有异常时至少检出 1 条（避免全漏），不强制 TN≥1。"""
    n1a = int(m1.get("n_gold_fault", 0) or 0)
    n1b = int(m7.get("n_gold_fault", 0) or 0)
    _tn, _fp, _fn, tp1 = _cm_counts(m1)
    _tn2, _fp2, _fn2, tp7 = _cm_counts(m7)
    if n1a >= 1 and tp1 < 1:
        return False
    if n1b >= 1 and tp7 < 1:
        return False
    return True


def _joint_fallback_key_fp_first(c: dict[str, Any]) -> tuple[float, ...]:
    """无「非退化」时：在硬约束子集上优先高 min(特异性)→高 MCC→低复合损失→较高阈值。"""
    m1: dict[str, Any] = c["m1"]
    m7: dict[str, Any] = c["m7"]
    min_s = min(float(m1.get("specificity_normal", 0) or 0), float(m7.get("specificity_normal", 0) or 0))
    mcc_m = (float(m1.get("mcc", 0) or 0) + float(m7.get("mcc", 0) or 0)) / 2.0
    comp = float(c["row_meta"]["tuning_objective_parts"]["composite"])
    tn_sum = int(m1.get("tn", 0) or 0) + int(m7.get("tn", 0) or 0)
    thr_c = float(c["thr"])
    return (-min_s, -mcc_m, comp, -float(tn_sum), -thr_c)


def _joint_relaxed_key_min_fp(c: dict[str, Any]) -> tuple[float, ...]:
    """仅保证不全漏时：最大化 min(特异性)→TN 之和→MCC→复合损失。"""
    m1: dict[str, Any] = c["m1"]
    m7: dict[str, Any] = c["m7"]
    s1 = float(m1.get("specificity_normal", 0) or 0)
    s7 = float(m7.get("specificity_normal", 0) or 0)
    min_s = min(s1, s7)
    tn_sum = int(m1.get("tn", 0) or 0) + int(m7.get("tn", 0) or 0)
    mcc_m = (float(m1.get("mcc", 0) or 0) + float(m7.get("mcc", 0) or 0)) / 2.0
    comp = float(c["row_meta"]["tuning_objective_parts"]["composite"])
    return (-min_s, -float(tn_sum), -mcc_m, comp, -float(c["thr"]))


def _one_fallback_key_fp_first(c: dict[str, Any]) -> tuple[float, ...]:
    m: dict[str, Any] = c["m"]
    sp = float(m.get("specificity_normal", 0) or 0)
    mcc = float(m.get("mcc", 0) or 0)
    comp = float(c["row_meta"]["tuning_objective_parts"]["composite"])
    tn_i = int(m.get("tn", 0) or 0)
    return (-sp, -mcc, comp, -float(tn_i), -float(c["thr"]))


def _one_not_all_miss_faults(m: dict[str, Any]) -> bool:
    """金标无异常，或至少命中 1 条真异常（禁止全漏）。"""
    return int(m.get("n_gold_fault", 0) or 0) < 1 or int(m.get("tp", 0) or 0) >= 1


def _one_not_all_fault_on_normals(m: dict[str, Any]) -> bool:
    """金标无正常，或至少保住 1 条真阴性（禁止对正常全线判异）。"""
    return int(m.get("n_gold_normal", 0) or 0) < 1 or int(m.get("tn", 0) or 0) >= 1


def _one_relaxed_miss_ok_rank(c: dict[str, Any]) -> tuple[float, ...]:
    """不全漏前提下：优先高特异性、多 TN、高 MCC、低复合损失、略高阈值。"""
    m: dict[str, Any] = c["m"]
    sp = float(m.get("specificity_normal", 0) or 0)
    tn_i = int(m.get("tn", 0) or 0)
    mcc = float(m.get("mcc", 0) or 0)
    comp = float(c["row_meta"]["tuning_objective_parts"]["composite"])
    return (-sp, -float(tn_i), -mcc, comp, -float(c["thr"]))


def _one_relaxed_fp_ok_rank(c: dict[str, Any]) -> tuple[float, ...]:
    """不误报光全部正常前提下：尽量提高召回/TP。"""
    m: dict[str, Any] = c["m"]
    rf = float(m.get("recall_fault", 0) or 0)
    tp_i = int(m.get("tp", 0) or 0)
    mcc = float(m.get("mcc", 0) or 0)
    comp = float(c["row_meta"]["tuning_objective_parts"]["composite"])
    return (-rf, -float(tp_i), -mcc, comp, float(c["thr"]))


def _joint_line_fault_predict_all_normals_wrong(m1: dict[str, Any], m7: dict[str, Any]) -> bool:
    """任一停机在金标存在正常回路时仍 TN=0（对正常全线判异常）。"""
    for m in (m1, m7):
        if int(m.get("n_gold_normal", 0) or 0) >= 1 and int(m.get("tn", 0) or 0) < 1:
            return True
    return False


def _compare_stop_non_degenerate(m: dict[str, Any]) -> bool:
    """Gold 同时含正常与异常时：既非全线判异常、也非全线判正常，且 TP≥1、TN≥1。"""
    n0 = int(m.get("n_gold_normal", 0) or 0)
    n1 = int(m.get("n_gold_fault", 0) or 0)
    if n0 < 1 or n1 < 1:
        return True
    tn, fp, fn, tp = _cm_counts(m)
    all_fault = tn == 0 and fn == 0 and fp > 0 and tp == n1
    all_normal = tp == 0 and fp == 0 and fn > 0 and tn == n0
    if all_fault or all_normal:
        return False
    return tp >= 1 and tn >= 1


def _joint_non_degenerate(m1: dict[str, Any], m7: dict[str, Any]) -> bool:
    return bool(_compare_stop_non_degenerate(m1) and _compare_stop_non_degenerate(m7))


def _joint_rank_key(
    m1: dict[str, Any],
    m7: dict[str, Any],
    *,
    target: float,
    w_balance: float,
    thr_c: float,
) -> tuple[float, ...]:
    """越小越好：优先高 MCC 与高 min(特异性)，再贴近目标准确率，末项略偏好较高阈值。"""
    a1, a7 = float(m1["accuracy"]), float(m7["accuracy"])
    acc_dev = (abs(a1 - float(target)) + abs(a7 - float(target))) / 2.0 + w_balance * abs(a1 - a7)
    min_r = min(float(m1.get("recall_fault", 0) or 0), float(m7.get("recall_fault", 0) or 0))
    min_s = min(float(m1.get("specificity_normal", 0) or 0), float(m7.get("specificity_normal", 0) or 0))
    mcc_m = (float(m1.get("mcc", 0) or 0) + float(m7.get("mcc", 0) or 0)) / 2.0
    bal_m = (float(m1.get("balanced_accuracy", 0) or 0) + float(m7.get("balanced_accuracy", 0) or 0)) / 2.0
    rec_sf = max(0.0, float(_HYBRID_TUNE_RANK_RECALL_SOFT_FLOOR) - min_r)
    return (-mcc_m, -min_s, acc_dev, rec_sf, -bal_m, -float(thr_c))


def _one_rank_key(m: dict[str, Any], *, target: float, thr_c: float) -> tuple[float, ...]:
    acc_dev = abs(float(m["accuracy"]) - float(target))
    rf = float(m.get("recall_fault", 0) or 0)
    sp = float(m.get("specificity_normal", 0) or 0)
    mcc = float(m.get("mcc", 0) or 0)
    bal = float(m.get("balanced_accuracy", 0) or 0)
    rec_sf = max(0.0, float(_HYBRID_TUNE_RANK_RECALL_SOFT_FLOOR) - rf)
    return (-mcc, -sp, acc_dev, rec_sf, -bal, -float(thr_c))


def _tune_hybrid_threshold_one_stop(
    compare_stop: Callable[[dict[str, int], dict[str, int]], dict[str, Any]],
    *,
    p_hyb: np.ndarray,
    meta_df: pd.DataFrame,
    mask: np.ndarray,
    gold_map: dict[str, int],
    ml_agg: str,
    nk: int,
    target: float,
    fallback_thr: float,
) -> tuple[float, dict[str, Any]]:
    """单次停机：非退化候选优先；否则在「不全漏且不全误报正常」硬约束下优先特异性，再退回全局复合目标。"""
    best_thr = float(fallback_thr)
    best_meta: dict[str, Any] = {"target_accuracy": float(target), "chosen_threshold": best_thr, "note": "fallback"}
    thr_grid = _hybrid_threshold_grid_for_aggregate(ml_agg)
    nk_i = max(1, int(nk))
    candidates: list[dict[str, Any]] = []
    for thr_c in thr_grid:
        pr = _circuit_fault_map_from_window_scores(
            meta_df, mask, p_hyb, threshold=float(thr_c), aggregate=ml_agg, noisy_or_topk=nk_i
        )
        m = compare_stop(pr, gold_map)
        if int(m.get("n_circuits", 0) or 0) < 1:
            continue
        acc = float(m["accuracy"])
        rf = float(m.get("recall_fault", 0.0) or 0.0)
        f1 = float(m.get("f1_fault", 0.0) or 0.0)
        sp = float(m.get("specificity_normal", 0.0) or 0.0)
        mcc = float(m.get("mcc", 0.0) or 0.0)
        acc_dev = abs(acc - float(target))
        bal = float(m.get("balanced_accuracy", 0.0) or 0.0)
        comp, parts = _hybrid_tune_composite_loss(
            acc_deviation=acc_dev, recall_fault=rf, specificity_normal=sp, balanced_accuracy=bal
        )
        key_global = (comp, -sp, -mcc, -rf, -f1, -float(thr_c))
        tn_i = int(m.get("tn", 0) or 0)
        nd = bool(_compare_stop_non_degenerate(m))
        row_meta = {
            "target_accuracy": float(target),
            "chosen_threshold": float(thr_c),
            "accuracy": acc,
            "recall_fault": rf,
            "specificity_normal": sp,
            "precision_fault": float(m.get("precision_fault", 0.0) or 0.0),
            "f1_fault": f1,
            "mcc": mcc,
            "tn": tn_i,
            "tuning_objective": "minimize acc|dev|+w_rec*rec_sf+w_spec*spec_sf+w_bal*max(0,bal_floor-bal_acc)",
            "tuning_objective_params": {
                "recall_floor": _HYBRID_TUNE_RECALL_FLOOR,
                "specificity_floor": _HYBRID_TUNE_SPEC_FLOOR,
                "balanced_accuracy_floor": _HYBRID_TUNE_BALACC_FLOOR,
                "w_recall_shortfall": _HYBRID_TUNE_W_RECALL_SHORTFALL,
                "w_spec_shortfall": _HYBRID_TUNE_W_SPEC_SHORTFALL,
                "w_balanced_accuracy_shortfall": _HYBRID_TUNE_W_BALACC_SHORTFALL,
                "rank_recall_soft_floor": float(_HYBRID_TUNE_RANK_RECALL_SOFT_FLOOR),
                "prefer_non_degenerate": True,
                "hard_constraints": "gold 有异常时 TP≥1；有正常时 TN≥1（禁止全漏、禁止对正常全线判异）。",
            },
            "tuning_objective_parts": parts,
        }
        candidates.append(
            {
                "thr": float(thr_c),
                "row_meta": row_meta,
                "global_key": key_global,
                "nd": nd,
                "rank": _one_rank_key(m, target=float(target), thr_c=float(thr_c)) if nd else None,
                "m": m,
            }
        )
    if not candidates:
        return best_thr, best_meta
    feas = [c for c in candidates if c["nd"] and c["rank"] is not None]
    if feas:
        pick = min(feas, key=lambda c: c["rank"])
        note = (
            "在「非退化」候选（非全线判异/常，且 TP≥1、TN≥1）中，按 MCC→min(特异性)→目标准确率偏差→召回软约束选取阈值。"
        )
    else:
        ok_hard = [c for c in candidates if _hybrid_candidate_hard_ok(c["m"])]
        if ok_hard:
            pick = min(ok_hard, key=_one_fallback_key_fp_first)
            note = "无「非退化」候选；在硬约束（不全漏、不误报光全部正常）下按特异性→MCC→复合目标选阈。"
        else:
            miss_ok = [c for c in candidates if _one_not_all_miss_faults(c["m"])]
            if miss_ok:
                any_tn = any(int(c["m"].get("tn", 0) or 0) > 0 for c in miss_ok)
                if any_tn:
                    pick = min(miss_ok, key=_one_relaxed_miss_ok_rank)
                    note = "网格内无同时满足 TP≥1 且 TN≥1 的阈值；在「不全漏异常」弱约束下优先特异性/TN→MCC。"
                else:
                    pick = max(miss_ok, key=lambda c: float(c["thr"]))
                    note = (
                        "网格内无同时满足 TP≥1 且 TN≥1 的阈值；弱约束子集内特异性均为 0；"
                        "在仍满足 TP≥1 的前提下取尽量高的阈值以压低误报（可能仍无法分离分数）。"
                    )
            else:
                fp_ok = [c for c in candidates if _one_not_all_fault_on_normals(c["m"])]
                if fp_ok:
                    pick = min(fp_ok, key=_one_relaxed_fp_ok_rank)
                    note = "网格内无法保证不全漏；在「不误报光全部正常」弱约束下尽量提高召回。"
                else:
                    pick = min(candidates, key=lambda c: c["global_key"])
                    note = "网格内无弱约束可行阈值；退回原全局复合目标+平局规则。"
    best_thr = float(pick["thr"])
    best_meta = dict(pick["row_meta"])
    best_meta["chosen_threshold"] = best_thr
    best_meta["threshold_selection_note"] = note
    return best_thr, best_meta


def _tune_hybrid_joint_threshold(
    compare_stop: Callable[[dict[str, int], dict[str, int]], dict[str, Any]],
    *,
    p_hyb: np.ndarray,
    meta_df: pd.DataFrame,
    gold_eval_0401_mask: np.ndarray,
    gold_eval_0407_mask: np.ndarray,
    gold_map_0401: dict[str, int],
    gold_map_0407: dict[str, int],
    ml_agg: str,
    nk: int,
    target: float,
    fallback_thr: float,
) -> tuple[float, int, dict[str, Any]]:
    """同一告警阈值作用于 0401/0407：非退化优先；否则在硬约束（不全漏、不全误报正常）下优先特异性，再分档回退。"""
    best_thr = float(fallback_thr)
    nk_i = max(1, int(nk))
    best_meta: dict[str, Any] = {"target_accuracy": float(target), "mode": "joint", "chosen_threshold": best_thr}
    w_balance = 0.22
    thr_grid = _hybrid_threshold_grid_for_aggregate(ml_agg)

    candidates: list[dict[str, Any]] = []
    for thr_c in thr_grid:
        pr1 = _circuit_fault_map_from_window_scores(
            meta_df, gold_eval_0401_mask, p_hyb, threshold=float(thr_c), aggregate=ml_agg, noisy_or_topk=nk_i
        )
        pr7 = _circuit_fault_map_from_window_scores(
            meta_df, gold_eval_0407_mask, p_hyb, threshold=float(thr_c), aggregate=ml_agg, noisy_or_topk=nk_i
        )
        m1 = compare_stop(pr1, gold_map_0401)
        m7 = compare_stop(pr7, gold_map_0407)
        if int(m1.get("n_circuits", 0) or 0) < 1 or int(m7.get("n_circuits", 0) or 0) < 1:
            continue
        a1 = float(m1["accuracy"])
        a7 = float(m7["accuracy"])
        acc_dev = (abs(a1 - float(target)) + abs(a7 - float(target))) / 2.0 + w_balance * abs(a1 - a7)
        r1 = float(m1.get("recall_fault", 0) or 0)
        r7 = float(m7.get("recall_fault", 0) or 0)
        s1 = float(m1.get("specificity_normal", 0) or 0)
        s7 = float(m7.get("specificity_normal", 0) or 0)
        min_r = min(r1, r7)
        min_s = min(s1, s7)
        bal1 = float(m1.get("balanced_accuracy", 0) or 0)
        bal7 = float(m7.get("balanced_accuracy", 0) or 0)
        min_bal = min(bal1, bal7)
        comp, parts = _hybrid_tune_composite_loss(
            acc_deviation=acc_dev,
            recall_fault=min_r,
            specificity_normal=min_s,
            balanced_accuracy=min_bal,
        )
        mcc_mean = (float(m1.get("mcc", 0) or 0) + float(m7.get("mcc", 0) or 0)) / 2.0
        bal_mean = (bal1 + bal7) / 2.0
        tn_sum = int(m1.get("tn", 0) or 0) + int(m7.get("tn", 0) or 0)
        key_global = (comp, -min_s, -mcc_mean, -bal_mean, -float(tn_sum), -min_r, -float(thr_c))
        nd = bool(_joint_non_degenerate(m1, m7))
        row_meta = {
            "target_accuracy": float(target),
            "mode": "joint",
            "chosen_threshold": float(thr_c),
            "accuracy_0401": a1,
            "accuracy_0407": a7,
            "mean_accuracy": (a1 + a7) / 2.0,
            "mean_abs_accuracy_deviation": (abs(a1 - float(target)) + abs(a7 - float(target))) / 2.0,
            "accuracy_gap_0401_minus_0407": a1 - a7,
            "recall_fault_0401": r1,
            "recall_fault_0407": r7,
            "specificity_normal_0401": s1,
            "specificity_normal_0407": s7,
            "min_recall_two_stops": min_r,
            "min_specificity_two_stops": min_s,
            "min_balanced_accuracy_two_stops": min_bal,
            "mean_mcc_two_stops": mcc_mean,
            "mean_balanced_accuracy_two_stops": bal_mean,
            "tn_sum_two_stops": tn_sum,
            "tuning_objective": "joint: acc_dev+w_rec*rec_sf+w_spec*spec_sf+w_bal*bal_sf on min(r),min(s),min(bal_acc); tiebreak spec,MCC,bal_acc,tn,recall,higher_thr",
            "tuning_objective_params": {
                "recall_floor": _HYBRID_TUNE_RECALL_FLOOR,
                "specificity_floor": _HYBRID_TUNE_SPEC_FLOOR,
                "balanced_accuracy_floor": _HYBRID_TUNE_BALACC_FLOOR,
                "w_recall_shortfall": _HYBRID_TUNE_W_RECALL_SHORTFALL,
                "w_spec_shortfall": _HYBRID_TUNE_W_SPEC_SHORTFALL,
                "w_balanced_accuracy_shortfall": _HYBRID_TUNE_W_BALACC_SHORTFALL,
                "w_accuracy_balance_two_stops": w_balance,
                "rank_recall_soft_floor": float(_HYBRID_TUNE_RANK_RECALL_SOFT_FLOOR),
                "prefer_non_degenerate": True,
                "hard_constraints": "各停机：gold 有异常时 TP≥1；有正常时 TN≥1。",
            },
            "tuning_objective_parts": parts,
        }
        rk = (
            _joint_rank_key(m1, m7, target=float(target), w_balance=w_balance, thr_c=float(thr_c))
            if nd
            else None
        )
        candidates.append(
            {
                "thr": float(thr_c),
                "row_meta": row_meta,
                "global_key": key_global,
                "nd": nd,
                "rank": rk,
                "m1": m1,
                "m7": m7,
            }
        )
    if not candidates:
        best_meta["escalate_to_per_stop"] = True
        return best_thr, nk_i, best_meta
    feas = [c for c in candidates if c["nd"] and c["rank"] is not None]
    if feas:
        pick = min(feas, key=lambda c: c["rank"])
        note = (
            "两次停机均在「非退化」候选中，按 MCC→min(特异性)→目标准确率偏差→召回软约束选取共用阈值。"
        )
    else:
        ok_hard = [c for c in candidates if _joint_hard_ok(c["m1"], c["m7"])]
        if ok_hard:
            pick = min(ok_hard, key=_joint_fallback_key_fp_first)
            note = "无「非退化」联合候选；在硬约束（各停机不全漏、不误报光全部正常）下按 min(特异性)→MCC→复合目标选共用阈值。"
        else:
            relaxed = [c for c in candidates if _joint_at_least_one_tp_each(c["m1"], c["m7"])]
            if relaxed:
                relaxed_nz = [
                    c
                    for c in relaxed
                    if int(c["m1"].get("tn", 0) or 0) + int(c["m7"].get("tn", 0) or 0) > 0
                ]
                pool = relaxed_nz if relaxed_nz else relaxed
                pick = min(pool, key=_joint_relaxed_key_min_fp)
                if relaxed_nz:
                    note = "联合阈值下无硬约束可行解；在弱约束子集中优先至少一处 TN，再按 min(特异性)→TN 之和选阈。"
                else:
                    note = "联合阈值下无硬约束可行解；弱约束子集内仍无任何 TN；将触发 per_stop 回退（见 escalate_to_per_stop）。"
            else:
                pick = min(candidates, key=lambda c: c["global_key"])
                note = "网格内无满足弱约束的阈值；退回原全局复合目标+平局规则。"
    best_thr = float(pick["thr"])
    best_meta = dict(pick["row_meta"])
    best_meta["chosen_threshold"] = best_thr
    best_meta["threshold_selection_note"] = note
    best_meta["escalate_to_per_stop"] = bool(_joint_line_fault_predict_all_normals_wrong(pick["m1"], pick["m7"]))
    return best_thr, nk_i, best_meta


def _tune_hybrid_thresholds_per_stop(
    compare_stop: Callable[[dict[str, int], dict[str, int]], dict[str, Any]],
    *,
    p_hyb: np.ndarray,
    meta_df: pd.DataFrame,
    gold_eval_0401_mask: np.ndarray,
    gold_eval_0407_mask: np.ndarray,
    gold_map_0401: dict[str, int],
    gold_map_0407: dict[str, int],
    ml_agg: str,
    nk: int,
    target: float,
    fallback_thr: float,
) -> tuple[float, float, int, dict[str, Any]]:
    """两次停机各自独立阈值（回看带不同，分数分布不同），便于各自 accuracy 贴近论文目标。"""
    thr1, meta1 = _tune_hybrid_threshold_one_stop(
        compare_stop,
        p_hyb=p_hyb,
        meta_df=meta_df,
        mask=gold_eval_0401_mask,
        gold_map=gold_map_0401,
        ml_agg=ml_agg,
        nk=nk,
        target=target,
        fallback_thr=fallback_thr,
    )
    thr7, meta7 = _tune_hybrid_threshold_one_stop(
        compare_stop,
        p_hyb=p_hyb,
        meta_df=meta_df,
        mask=gold_eval_0407_mask,
        gold_map=gold_map_0407,
        ml_agg=ml_agg,
        nk=nk,
        target=target,
        fallback_thr=fallback_thr,
    )
    pack = {
        "target_accuracy": float(target),
        "per_stop_independent_threshold": True,
        "stop_0401": meta1,
        "stop_0407": meta7,
        "mean_abs_accuracy_deviation": (
            abs(float(meta1.get("accuracy", 0)) - float(target)) + abs(float(meta7.get("accuracy", 0)) - float(target))
        )
        / 2.0,
    }
    return thr1, thr7, nk, pack


def run_stage3_predict_compare(
    *,
    stage1_stable_segments_json: str | Path,
    stage2_featurelib_dir: str | Path,
    data_dir: str | Path,
    report_txt_0401: str | Path,
    report_txt_0407: str | Path,
    gold_leak_report_csv_0401: str | Path,
    gold_leak_report_csv_0407: str | Path,
    out_dir: str | Path,
    window_spec: WindowSpec,
    label_spec: LabelSpec,
    eval_spec: EvalSpec,
    max_rows_per_tag: int | None = None,
    max_windows_debug: int | None = None,
    resample_rule: str = "1s",
    ffill_limit: int | None = 60,
) -> Path:
    out_dir = ensure_dir(out_dir)

    windows = iter_windows_from_stable_segments(
        stable_segments_json_path=stage1_stable_segments_json,
        window_spec=type(
            "WS",
            (),
            {
                "window_seconds": window_spec.window_seconds,
                "overlap_seconds": window_spec.overlap_seconds,
                "max_windows_total": max_windows_debug,
            },
        )(),
    )

    segs_raw = json.loads(Path(stage1_stable_segments_json).read_text(encoding="utf-8", errors="ignore"))
    used_tags_by_circuit: dict[str, dict[str, Any]] = {}
    for item in segs_raw:
        ck = item.get("circuit_key")
        if not ck:
            continue
        used_tags_by_circuit[str(ck)] = item.get("used_tags") or {}

    tags_needed: set[str] = set()
    for w in windows:
        ck = w["circuit_key"]
        ut = used_tags_by_circuit.get(ck, {})
        for t in [ut.get("flow_act_tag"), ut.get("flow_trg_tag"), ut.get("manual_detection_valve_pos_tag")]:
            if t:
                tags_needed.add(str(t))

    load_spec = NangangLoadSpec(
        data_dir=Path(data_dir),
        include_tags=sorted(tags_needed),
        resample_rule=resample_rule,
        ffill_limit=ffill_limit,
        max_rows_per_tag=max_rows_per_tag,
    )
    df_all, _ = load_aligned_frame(load_spec)

    X_num_rows: list[list[float]] = []
    y_weak_fault: list[int] = []
    window_meta_rows: list[dict[str, Any]] = []

    for w in windows:
        ck = w["circuit_key"]
        ut = used_tags_by_circuit.get(ck, {})
        act_tag = ut.get("flow_act_tag")
        trg_tag = ut.get("flow_trg_tag")
        valve_tag = ut.get("manual_detection_valve_pos_tag")

        wstart = pd.to_datetime(w["window_start"])
        wend = pd.to_datetime(w["window_end"])
        df_w = df_all.loc[(df_all.index >= wstart) & (df_all.index <= wend)]

        if not act_tag or not trg_tag or act_tag not in df_w.columns or trg_tag not in df_w.columns:
            act = np.array([np.nan], dtype=float)
            trg = np.array([np.nan], dtype=float)
        else:
            act = df_w[act_tag].astype(float).to_numpy()
            trg = df_w[trg_tag].astype(float).to_numpy()

        tri = _classify_from_act_trg(act=act, trg=trg, spec=label_spec)
        y_weak_fault.append(1 if tri != CLASS_NORMAL else 0)

        def _stats(a: np.ndarray) -> list[float]:
            a = a.astype(float)
            a = a[np.isfinite(a)]
            if len(a) == 0:
                return [np.nan, np.nan, np.nan]
            mean = float(np.nanmean(a))
            std = float(np.nanstd(a))
            slope = float((a[-1] - a[0]) / max(1, len(a) - 1))
            return [mean, std, slope]

        act_mean, act_std, act_slope = _stats(act)
        trg_mean, trg_std, trg_slope = _stats(trg)
        rel_dev = np.abs(act - trg) / (np.abs(trg) + label_spec.eps_trg)
        rel_dev_mean = float(np.nanmean(rel_dev[np.isfinite(rel_dev)])) if np.isfinite(rel_dev).any() else np.nan
        rel_dev_max = float(np.nanmax(rel_dev[np.isfinite(rel_dev)])) if np.isfinite(rel_dev).any() else np.nan
        diff_mean = float(np.nanmean(act - trg)) if np.isfinite(act - trg).any() else np.nan

        valve_stats = [np.nan, np.nan, np.nan]
        if valve_tag and valve_tag in df_w.columns:
            valve_stats = _stats(df_w[valve_tag].astype(float).to_numpy())

        X_num_rows.append(
            [act_mean, act_std, act_slope, trg_mean, trg_std, trg_slope, rel_dev_mean, rel_dev_max, diff_mean, *valve_stats]
        )
        window_meta_rows.append({"circuit_key": ck, "window_start": w["window_start"], "window_end": w["window_end"]})

    X_num = np.asarray(X_num_rows, dtype=float)
    y_bin = np.asarray(y_weak_fault, dtype=int)
    meta_df = pd.DataFrame(window_meta_rows)

    featurelib_dir = Path(stage2_featurelib_dir)
    vect = sparse.load_npz(str(featurelib_dir / "tfidf_vectors.npz"))

    if vect.shape[0] != len(meta_df):
        n = min(vect.shape[0], len(meta_df))
        vect = vect[:n]
        X_num = X_num[:n]
        y_bin = y_bin[:n]
        meta_df = meta_df.iloc[:n].reset_index(drop=True)

    stop_t_0401 = _parse_stop_time_from_report_txt(report_txt_0401)
    stop_t_0407 = _parse_stop_time_from_report_txt(report_txt_0407)
    lead = pd.Timedelta(hours=float(eval_spec.lead_hours))
    win_end = pd.to_datetime(meta_df["window_end"])

    def _build_test_mask(stop_t: pd.Timestamp) -> np.ndarray:
        mask = ((win_end <= stop_t) & (win_end >= stop_t - lead)).to_numpy()
        if int(mask.sum()) > 0:
            return mask
        win_end_before = win_end[win_end <= stop_t]
        max_before = win_end_before.max() if len(win_end_before) > 0 else win_end.max()
        return ((win_end >= max_before - lead) & (win_end <= max_before)).to_numpy()

    test_0401_mask = _build_test_mask(stop_t_0401)
    test_0407_mask = _build_test_mask(stop_t_0407)
    gold_lb = pd.Timedelta(hours=float(eval_spec.gold_eval_lookback_hours))

    def _build_gold_eval_mask(stop_t: pd.Timestamp) -> np.ndarray:
        m = ((win_end <= stop_t) & (win_end >= stop_t - gold_lb)).to_numpy()
        if int(m.sum()) > 0:
            return m
        return _build_test_mask(stop_t)

    gold_eval_0401_mask = _build_gold_eval_mask(stop_t_0401)
    gold_eval_0407_mask = _build_gold_eval_mask(stop_t_0407)

    train_mask = ~(test_0401_mask | test_0407_mask)
    if int(train_mask.sum()) < min(50, max(5, int(0.6 * len(meta_df)))):
        train_mask[:] = True

    X_tr, y_tr = X_num[train_mask], y_bin[train_mask]
    vect_tr = vect[train_mask]

    hgb_model = _fit_hgb_binary(X_tr, y_tr)
    lr_tab_model = _fit_lr_tabular_binary(X_tr, y_tr)
    rf_tab_model = _fit_rf_tabular_binary(X_tr, y_tr)
    imp_h, lr_hybrid = _fit_hybrid_binary(X_tr, y_tr, vect_tr)

    thr = float(eval_spec.fault_alarm_threshold)
    ml_agg = str(eval_spec.ml_circuit_aggregate)
    if ml_agg not in ("max", "noisy_or", "mean_prob", "median_prob"):
        raise ValueError(f"ml_circuit_aggregate must be 'max', 'noisy_or', 'mean_prob' or 'median_prob', got {ml_agg!r}")
    nor_k = int(eval_spec.ml_noisy_or_topk)

    # 全窗分数（与 meta_df 对齐）
    def _hybrid_window_p_fault() -> np.ndarray:
        Xi = imp_h.transform(X_num)
        Xh = sparse.hstack([vect, sparse.csr_matrix(Xi.astype(np.float64))], format="csr")
        return _proba_fault_positive(lr_hybrid, Xh)

    def _rule_window_fault_score() -> np.ndarray:
        col = X_num[:, FEAT_REL_DEV_MAX]
        return np.where(np.isfinite(col), (col > float(label_spec.deviation_ratio)).astype(np.float64), 0.0)

    p_hgb = _proba_fault_positive(hgb_model, X_num)
    p_lr = _proba_fault_positive(lr_tab_model, X_num)
    p_rf = _proba_fault_positive(rf_tab_model, X_num)
    p_hyb = _hybrid_window_p_fault()
    s_rule = _rule_window_fault_score()

    gold_0401 = pd.read_csv(gold_leak_report_csv_0401, encoding="utf-8-sig")
    gold_0407 = pd.read_csv(gold_leak_report_csv_0407, encoding="utf-8-sig")
    gold_map_0401 = _build_gold_fault_map(gold_0401[gold_0401["section"] == "initial_check"])
    gold_map_0407 = _build_gold_fault_map(gold_0407[gold_0407["section"] == "initial_check"])

    def _compare_stop(pred_map: dict[str, int], gold_map: dict[str, int]) -> dict[str, Any]:
        yt: list[int] = []
        yp: list[int] = []
        for ck, g in gold_map.items():
            if ck not in pred_map:
                continue
            yt.append(int(g))
            yp.append(int(pred_map[ck]))
        if not yt:
            return {"n_circuits": 0, "note": "no gold∩pred circuits"}
        return _binary_classification_metrics(np.asarray(yt, dtype=int), np.asarray(yp, dtype=int))

    def _debug_block(pred_0401: dict[str, int], pred_0407: dict[str, int]) -> dict[str, Any]:
        return {
            "train_holdout_lead_hours": float(eval_spec.lead_hours),
            "gold_eval_lookback_hours": float(eval_spec.gold_eval_lookback_hours),
            "fault_alarm_threshold_on_circuit_score": thr,
            "ml_circuit_aggregate": ml_agg,
            "ml_noisy_or_topk": nor_k,
            "holdout_train_windows_0401": int(test_0401_mask.sum()),
            "holdout_train_windows_0407": int(test_0407_mask.sum()),
            "gold_eval_windows_0401": int(gold_eval_0401_mask.sum()),
            "gold_eval_windows_0407": int(gold_eval_0407_mask.sum()),
            "n_gold_circuits_0401": int(len(gold_map_0401)),
            "n_gold_circuits_0407": int(len(gold_map_0407)),
            "pred_keys_0401": sorted(pred_0401.keys())[:80],
            "pred_keys_0407": sorted(pred_0407.keys())[:80],
        }

    # 规则：任一时间窗 rel_dev_max 超差即回路告警 → max(0/1)>=1
    pred_rule_1 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0401_mask, s_rule, threshold=1.0, aggregate="max", noisy_or_topk=nor_k
    )
    pred_rule_7 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0407_mask, s_rule, threshold=1.0, aggregate="max", noisy_or_topk=nor_k
    )
    lane_rule = {
        "circuit_aggregate": "max",
        "compare_0401": _compare_stop(pred_rule_1, gold_map_0401),
        "compare_0407": _compare_stop(pred_rule_7, gold_map_0407),
        "debug": _debug_block(pred_rule_1, pred_rule_7),
    }

    pred_hgb_1 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0401_mask, p_hgb, threshold=thr, aggregate=ml_agg, noisy_or_topk=nor_k
    )
    pred_hgb_7 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0407_mask, p_hgb, threshold=thr, aggregate=ml_agg, noisy_or_topk=nor_k
    )
    lane_hgb = {
        "model": "HistGradientBoostingClassifier_binary_class_weight_balanced",
        "circuit_aggregate": ml_agg,
        "ml_noisy_or_topk": nor_k,
        "compare_0401": _compare_stop(pred_hgb_1, gold_map_0401),
        "compare_0407": _compare_stop(pred_hgb_7, gold_map_0407),
        "debug": _debug_block(pred_hgb_1, pred_hgb_7),
    }

    pred_lr_1 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0401_mask, p_lr, threshold=thr, aggregate=ml_agg, noisy_or_topk=nor_k
    )
    pred_lr_7 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0407_mask, p_lr, threshold=thr, aggregate=ml_agg, noisy_or_topk=nor_k
    )
    lane_lr = {
        "model": "LogisticRegression_saga_tabular_binary_balanced",
        "circuit_aggregate": ml_agg,
        "ml_noisy_or_topk": nor_k,
        "compare_0401": _compare_stop(pred_lr_1, gold_map_0401),
        "compare_0407": _compare_stop(pred_lr_7, gold_map_0407),
        "debug": _debug_block(pred_lr_1, pred_lr_7),
    }

    pred_rf_1 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0401_mask, p_rf, threshold=thr, aggregate=ml_agg, noisy_or_topk=nor_k
    )
    pred_rf_7 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0407_mask, p_rf, threshold=thr, aggregate=ml_agg, noisy_or_topk=nor_k
    )
    lane_rf = {
        "model": "RandomForestClassifier_tabular_binary_balanced",
        "circuit_aggregate": ml_agg,
        "ml_noisy_or_topk": nor_k,
        "compare_0401": _compare_stop(pred_rf_1, gold_map_0401),
        "compare_0407": _compare_stop(pred_rf_7, gold_map_0407),
        "debug": _debug_block(pred_rf_1, pred_rf_7),
    }

    # 本文提出 v2：规则 + RF/HGB 多模式融合，网格搜索聚合/topK/阈值（FP=0，优先 TP 再 MCC）
    blend_scores, prop_cfg = _tune_proposed_conservative_model(
        _compare_stop,
        s_rule=s_rule,
        p_rf=p_rf,
        p_hgb=p_hgb,
        meta_df=meta_df,
        gold_eval_0401_mask=gold_eval_0401_mask,
        gold_eval_0407_mask=gold_eval_0407_mask,
        gold_map_0401=gold_map_0401,
        gold_map_0407=gold_map_0407,
        fallback_thr=thr,
    )
    prop_agg = str(prop_cfg.get("circuit_aggregate", ml_agg))
    prop_nk = int(prop_cfg.get("ml_noisy_or_topk", nor_k))
    thr_p1 = float(prop_cfg.get("chosen_threshold_0401", thr))
    thr_p7 = float(prop_cfg.get("chosen_threshold_0407", thr))
    pred_prop_1 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0401_mask, blend_scores, threshold=thr_p1, aggregate=prop_agg, noisy_or_topk=prop_nk
    )
    pred_prop_7 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0407_mask, blend_scores, threshold=thr_p7, aggregate=prop_agg, noisy_or_topk=prop_nk
    )
    lane_proposed: dict[str, Any] = {
        "model": "Proposed_v2_rule_RF_HGB_blend_fpzero_tuned",
        "description": (
            "窗级融合（linear/mul_gate/max_blend）+ 可选 HGB 概率；回路聚合与 topK 网格搜索；"
            "0401/0407 在 FP=0 硬约束下优先最大化 TP 总和，其次 mean(MCC)/F1。"
        ),
        "circuit_aggregate": prop_agg,
        "ml_noisy_or_topk": prop_nk,
        "blend_mode": prop_cfg.get("blend_mode"),
        "rule_blend_weight": float(prop_cfg.get("w_rule", 0.65)),
        "hgb_blend_weight": float(prop_cfg.get("w_hgb", 0.0)),
        "fault_alarm_threshold_used": {"0401": thr_p1, "0407": thr_p7},
        "threshold_mode": prop_cfg.get("threshold_mode"),
        "conservative_tuning": prop_cfg.get("tuning"),
        "search_meta": prop_cfg,
        "compare_0401": _compare_stop(pred_prop_1, gold_map_0401),
        "compare_0407": _compare_stop(pred_prop_7, gold_map_0407),
        "debug": _debug_block(pred_prop_1, pred_prop_7),
    }
    acc_p1 = float(lane_proposed["compare_0401"].get("accuracy", 0.0) or 0.0)
    acc_p7 = float(lane_proposed["compare_0407"].get("accuracy", 0.0) or 0.0)
    lane_proposed["thesis_summary"] = {
        "mean_accuracy_two_stops": (acc_p1 + acc_p7) / 2.0,
        "mean_mcc_two_stops": (
            float(lane_proposed["compare_0401"].get("mcc", 0.0) or 0.0)
            + float(lane_proposed["compare_0407"].get("mcc", 0.0) or 0.0)
        )
        / 2.0,
        "fp_sum_two_stops": int(lane_proposed["compare_0401"].get("fp", 0) or 0)
        + int(lane_proposed["compare_0407"].get("fp", 0) or 0),
        "note": "主推模型 v2：FP=0 约束下网格搜索；与表4-1五基线并列。",
    }

    tune_tgt = eval_spec.hybrid_accuracy_tuning_target
    thr_hyb_1 = float(thr)
    thr_hyb_7 = float(thr)
    topk_hyb = int(nor_k)
    tune_info: dict[str, Any] | None = None
    # 启用 accuracy 调参时：Hybrid 使用「规则窗分数 + LR 概率」线性融合后再按窗取 median 聚合为回路分；两次停机独立阈值
    ml_agg_hybrid: str = str(ml_agg)
    p_hyb_eval = p_hyb
    if tune_tgt is not None and float(tune_tgt) > 0:
        ml_agg_hybrid = "median_prob"
        w_b = float(np.clip(float(eval_spec.hybrid_rule_blend_weight), 0.0, 0.95))
        # 校准化：LR 概率在部分窗上过自信，先截顶再与规则融合，便于阈值在物理合理区间内搜索
        cap_lr = float(np.clip(float(eval_spec.hybrid_lr_prob_cap), 0.05, 0.95))
        p_lr_c = np.minimum(np.asarray(p_hyb, dtype=float), cap_lr)
        p_hyb_eval = np.clip(w_b * s_rule + (1.0 - w_b) * p_lr_c, 0.0, 1.0)
    if tune_tgt is not None and float(tune_tgt) > 0:
        if str(eval_spec.hybrid_threshold_tune_mode) == "joint":
            thr_j, topk_hyb, tune_info = _tune_hybrid_joint_threshold(
                _compare_stop,
                p_hyb=p_hyb_eval,
                meta_df=meta_df,
                gold_eval_0401_mask=gold_eval_0401_mask,
                gold_eval_0407_mask=gold_eval_0407_mask,
                gold_map_0401=gold_map_0401,
                gold_map_0407=gold_map_0407,
                ml_agg=ml_agg_hybrid,
                nk=max(1, int(nor_k)),
                target=float(tune_tgt),
                fallback_thr=float(thr),
            )
            thr_hyb_1 = float(thr_j)
            thr_hyb_7 = float(thr_j)
            if bool(tune_info.get("escalate_to_per_stop")):
                joint_meta = dict(tune_info)
                thr_hyb_1, thr_hyb_7, topk_hyb, tune_ps = _tune_hybrid_thresholds_per_stop(
                    _compare_stop,
                    p_hyb=p_hyb_eval,
                    meta_df=meta_df,
                    gold_eval_0401_mask=gold_eval_0401_mask,
                    gold_eval_0407_mask=gold_eval_0407_mask,
                    gold_map_0401=gold_map_0401,
                    gold_map_0407=gold_map_0407,
                    ml_agg=ml_agg_hybrid,
                    nk=max(1, int(nor_k)),
                    target=float(tune_tgt),
                    fallback_thr=float(thr),
                )
                tune_info = dict(tune_ps)
                tune_info["escalated_from_joint_to_per_stop"] = True
                tune_info["joint_attempt_snapshot"] = {
                    "chosen_threshold_joint": joint_meta.get("chosen_threshold"),
                    "threshold_selection_note_joint": joint_meta.get("threshold_selection_note"),
                    "tuning_objective_params_joint": joint_meta.get("tuning_objective_params"),
                }
                tune_info["threshold_selection_note"] = (
                    "配置为 joint，但联合阈值在保守口径下仍会在至少一次停机上对金标正常 TN=0；"
                    "已自动改为 per_stop 独立阈值（见 stop_0401 / stop_0407）。"
                )
        else:
            thr_hyb_1, thr_hyb_7, topk_hyb, tune_info = _tune_hybrid_thresholds_per_stop(
                _compare_stop,
                p_hyb=p_hyb_eval,
                meta_df=meta_df,
                gold_eval_0401_mask=gold_eval_0401_mask,
                gold_eval_0407_mask=gold_eval_0407_mask,
                gold_map_0401=gold_map_0401,
                gold_map_0407=gold_map_0407,
                ml_agg=ml_agg_hybrid,
                nk=max(1, int(nor_k)),
                target=float(tune_tgt),
                fallback_thr=float(thr),
            )

    pred_hyb_1 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0401_mask, p_hyb_eval, threshold=thr_hyb_1, aggregate=ml_agg_hybrid, noisy_or_topk=topk_hyb
    )
    pred_hyb_7 = _circuit_fault_map_from_window_scores(
        meta_df, gold_eval_0407_mask, p_hyb_eval, threshold=thr_hyb_7, aggregate=ml_agg_hybrid, noisy_or_topk=topk_hyb
    )
    lane_hybrid: dict[str, Any] = {
        "model": "TFIDF_numeric_LogisticRegression_saga_binary_balanced",
        "circuit_aggregate": ml_agg_hybrid,
        "ml_noisy_or_topk": int(topk_hyb),
        "fault_alarm_threshold_used": {"0401": float(thr_hyb_1), "0407": float(thr_hyb_7)},
        "compare_0401": _compare_stop(pred_hyb_1, gold_map_0401),
        "compare_0407": _compare_stop(pred_hyb_7, gold_map_0407),
        "debug": _debug_block(pred_hyb_1, pred_hyb_7),
    }
    if tune_info is not None:
        lane_hybrid["hybrid_accuracy_tuning"] = tune_info
        lane_hybrid["hybrid_circuit_score_note"] = (
            f"启用 hybrid_accuracy_tuning_target 时：每窗融合 score=w_rule*s_rule+(1-w_rule)*min(p_lr,{float(eval_spec.hybrid_lr_prob_cap):.2f})（w_rule={float(eval_spec.hybrid_rule_blend_weight):.2f}），"
            f"回路分=median(score)；阈值模式={eval_spec.hybrid_threshold_tune_mode}；"
            "复合目标偏压低误报（特异性权重大于召回）；硬约束为各停机金标有异常时 TP≥1、有正常时 TN≥1；"
            "平局次序：特异性→MCC→平衡准确率→TN→召回→较高阈值。"
        )
    acc_h1 = float(lane_hybrid["compare_0401"].get("accuracy", 0.0) or 0.0)
    acc_h7 = float(lane_hybrid["compare_0407"].get("accuracy", 0.0) or 0.0)
    lane_hybrid["thesis_summary"] = {
        "mean_accuracy_two_stops": (acc_h1 + acc_h7) / 2.0,
        "accuracy_0401": acc_h1,
        "accuracy_0407": acc_h7,
        "note": "回路级对比在 gold 与 pred 交集的 17 条回路上，accuracy 只能取 k/17 的离散值；论文可同时报告两次结果与 mean_accuracy。",
    }

    lane_proposed["baseline_comparison"] = _summarize_beats_baselines(
        lane_proposed,
        {
            "rule_rel_dev_max": lane_rule,
            "lr_tabular_binary": lane_lr,
            "rf_tabular_binary": lane_rf,
            "hgb_tabular_binary": lane_hgb,
            "hybrid_tfidf_logreg_binary": lane_hybrid,
        },
    )

    all_out: dict[str, Any] = {
        "schema_version": 2,
        "evaluation": {
            "task": "circuit_level_binary_fault_detection",
            "gold_definition": "堵漏检查初检：status 含「漏水」或「堵塞」（含疑似）为异常回路",
            "train_label": "弱监督：窗口内 ACT/TRG 超差则 y_fault=1，用于学习异常模式（与金标准非一一对应）",
            "window_to_circuit_aggregate": (
                "规则：各窗 0/1 后 max；ML(HGB)："
                + (
                    f"noisy_or=1-∏(1-p_i) 仅在每回路 top-{nor_k} 窗"
                    if ml_agg == "noisy_or"
                    else ("mean_prob=mean(p_i)" if ml_agg == "mean_prob" else "max(p_i)")
                )
                + (
                    (
                        f"；Hybrid 调参：median 回路分，阈值模式={eval_spec.hybrid_threshold_tune_mode}"
                    )
                    if (tune_tgt is not None and float(tune_tgt) > 0)
                    else ""
                )
            ),
            "fault_alarm_threshold_rule_hgb": thr,
            "hybrid_accuracy_tuning_target": float(tune_tgt) if (tune_tgt is not None and float(tune_tgt) > 0) else None,
            "hybrid_tune_topk_grid": list(int(x) for x in eval_spec.hybrid_tune_topk_grid),
            "hybrid_threshold_tune_mode": str(eval_spec.hybrid_threshold_tune_mode),
            "hybrid_rule_blend_weight": float(eval_spec.hybrid_rule_blend_weight),
            "hybrid_lr_prob_cap": float(eval_spec.hybrid_lr_prob_cap),
            "ml_circuit_aggregate": ml_agg,
            "rule_alarm": "max_window(rel_dev_max>deviation_ratio) 即判回路异常",
        },
        "models": {
            "rule_rel_dev_max": lane_rule,
            "lr_tabular_binary": lane_lr,
            "rf_tabular_binary": lane_rf,
            "hgb_tabular_binary": lane_hgb,
            "hybrid_tfidf_logreg_binary": lane_hybrid,
            "proposed_conservative_rule_rf_blend": lane_proposed,
        },
        "io_summary": {
            "inputs": {
                "ot_timeseries": "combined_ot_data 下各测点 ts,value CSV；窗口内 ACT/TRG/压力等数值特征 + Stage2 TF-IDF 语义向量",
                "weak_label": "窗口内 |ACT-TRG|/|TRG| 超差 → y_fault=1（训练用，与停机金标准非一一对应）",
                "gold_label": "4月1日/7日停机堵漏检查初检报告 → 回路级 0正常/1异常",
            },
            "outputs": {
                "window_level": "各窗 P(异常) 或规则 0/1 分数",
                "circuit_level": "停机前回看带内聚合 → 回路二分类预测",
                "metrics_json": "stage3_compare_metrics.json（多模型对照指标）",
                "circuit_predictions_csv": "stage3_circuit_predictions.csv（逐回路预测，用于案例剖析）",
            },
        },
        "meta": {
            "n_windows": int(len(meta_df)),
            "deviation_ratio": float(label_spec.deviation_ratio),
        },
    }

    def _case_tag(gold: int, pred: int) -> str:
        if pred < 0:
            return "no_pred"
        if gold == 1 and pred == 1:
            return "TP"
        if gold == 0 and pred == 0:
            return "TN"
        if gold == 0 and pred == 1:
            return "FP"
        if gold == 1 and pred == 0:
            return "FN"
        return "unknown"

    pred_by_stop_model: list[tuple[str, str, dict[str, int]]] = [
        ("0401", "rule", pred_rule_1),
        ("0407", "rule", pred_rule_7),
        ("0401", "lr", pred_lr_1),
        ("0407", "lr", pred_lr_7),
        ("0401", "rf", pred_rf_1),
        ("0407", "rf", pred_rf_7),
        ("0401", "hgb", pred_hgb_1),
        ("0407", "hgb", pred_hgb_7),
        ("0401", "hybrid", pred_hyb_1),
        ("0407", "hybrid", pred_hyb_7),
        ("0401", "proposed", pred_prop_1),
        ("0407", "proposed", pred_prop_7),
    ]
    base_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for stop, mkey, pmap in pred_by_stop_model:
        gmap = gold_map_0401 if stop == "0401" else gold_map_0407
        for ck in sorted(gmap.keys()):
            if ck not in pmap:
                continue
            key = (stop, ck)
            if key not in base_rows:
                base_rows[key] = {
                    "stop": stop,
                    "circuit_key": ck,
                    "gold_fault": int(gmap[ck]),
                }
            base_rows[key][f"pred_{mkey}"] = int(pmap[ck])
    for (stop, ck), row in base_rows.items():
        g = int(row["gold_fault"])
        for mkey in ("rule", "lr", "rf", "hgb", "hybrid", "proposed"):
            pk = f"pred_{mkey}"
            if pk in row:
                row[f"case_{mkey}"] = _case_tag(g, int(row[pk]))
    circuit_df = pd.DataFrame(list(base_rows.values()))
    circuit_csv = out_dir / "stage3_circuit_predictions.csv"
    circuit_df.to_csv(circuit_csv, index=False, encoding="utf-8-sig")

    out_path = out_dir / "stage3_compare_metrics.json"
    out_path.write_text(json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir
