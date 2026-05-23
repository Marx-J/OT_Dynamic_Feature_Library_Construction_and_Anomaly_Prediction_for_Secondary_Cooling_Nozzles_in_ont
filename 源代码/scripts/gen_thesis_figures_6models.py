"""
生成毕业论文用「六模型对照」图表与表格图（PNG/PDF/CSV）。

用法（thesis_code 目录）:
  python scripts/gen_thesis_figures_6models.py
  python scripts/gen_thesis_figures_6models.py --json outputs/retrain_combined/stage3_multimodel_full/stage3_compare_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "outputs" / "retrain_combined" / "stage3_multimodel_full" / "stage3_compare_metrics.json"
OUT_DIR = ROOT / "outputs" / "retrain_combined" / "thesis_figures_6models"

# 五基线 + 提出模型（顺序：基线在前，提出模型最后并高亮）
MODEL_SPECS: list[tuple[str, str, bool]] = [
    ("rule_rel_dev_max", "规则基线", False),
    ("lr_tabular_binary", "逻辑回归", False),
    ("rf_tabular_binary", "随机森林", False),
    ("hgb_tabular_binary", "梯度提升(HGB)", False),
    ("hybrid_tfidf_logreg_binary", "TF-IDF+LR混合", False),
    ("proposed_conservative_rule_rf_blend", "本文提出模型", True),
]

METRIC_FIELDS = [
    ("recall_fault", "异常类召回率"),
    ("precision_fault", "异常类精确率"),
    ("f1_fault", "异常类 F1"),
    ("balanced_accuracy", "平衡准确率"),
    ("mcc", "MCC"),
    ("accuracy", "准确率"),
]

STOP_LABELS = {"0401": "4月1日停机", "0407": "4月7日停机"}
BASELINE_COLOR = "#5B9BD5"
PROPOSED_COLOR = "#C00000"
STOP_COLORS = {"0401": "#4472C4", "0407": "#ED7D31"}


def _setup_chinese_font() -> None:
    for c in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "SimSun"]:
        if c in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [c]
            plt.rcParams["font.family"] = "sans-serif"
            break
    plt.rcParams["axes.unicode_minus"] = False


def _load_rows(data: dict) -> pd.DataFrame:
    models = data["models"]
    rows: list[dict] = []
    for key, label, is_prop in MODEL_SPECS:
        lane = models.get(key, {})
        for stop in ("0401", "0407"):
            m = lane.get(f"compare_{stop}", {})
            if not m or int(m.get("n_circuits", 0) or 0) == 0:
                continue
            rows.append(
                {
                    "model_key": key,
                    "model_label": label,
                    "is_proposed": is_prop,
                    "stop": stop,
                    "stop_label": STOP_LABELS[stop],
                    "n_circuits": int(m["n_circuits"]),
                    "tn": int(m.get("tn", 0)),
                    "fp": int(m.get("fp", 0)),
                    "fn": int(m.get("fn", 0)),
                    "tp": int(m.get("tp", 0)),
                    **{f: float(m.get(f, 0) or 0) for f, _ in METRIC_FIELDS},
                }
            )
    return pd.DataFrame(rows)


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _cm2x2(block: dict) -> np.ndarray:
    """[[TN, FP],[FN,TP]]，行=金标(0正常/1异常)，列=预测。"""
    rows = block.get("confusion_binary_rows_gold_0normal_1fault")
    if rows:
        return np.array(rows, dtype=float)
    return np.array(
        [[block.get("tn", 0), block.get("fp", 0)], [block.get("fn", 0), block.get("tp", 0)]],
        dtype=float,
    )


def _plot_confusion_on_ax(
    ax: plt.Axes,
    cm: np.ndarray,
    title: str,
    *,
    is_proposed: bool = False,
    show_colorbar: bool = False,
) -> None:
    vmax = max(float(cm.max()), 1.0)
    cmap = "OrRd" if is_proposed else "Blues"
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{cell_labels[i][j]}\n{int(v)}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if v > vmax * 0.55 else "black",
        )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测正常", "预测异常"], fontsize=8)
    ax.set_yticklabels(["金标正常", "金标异常"], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold" if is_proposed else "normal")
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_confusion_six_grid(data: dict, stop: str, out_dir: Path) -> None:
    """单次停机：2×3 六模型混淆矩阵。"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), dpi=150)
    models = data["models"]
    for ax, (key, label, is_prop) in zip(axes.flat, MODEL_SPECS, strict=True):
        block = models[key][f"compare_{stop}"]
        cm = _cm2x2(block)
        mcc = float(block.get("mcc", 0) or 0)
        _plot_confusion_on_ax(
            ax,
            cm,
            f"{label}\nMCC={mcc:.3f}",
            is_proposed=is_prop,
        )
    fig.suptitle(
        f"{STOP_LABELS[stop]} — 六模型回路级混淆矩阵（行=金标，列=预测；n=17）",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig_num = "10" if stop == "0401" else "11"
    _save_fig(fig, out_dir, f"fig{fig_num}_confusion_matrices_{stop}")


def _plot_confusion_six_by_two_stops(data: dict, out_dir: Path) -> None:
    """6 行 × 2 列：每行一模型，左 4/1 右 4/7（附录用）。"""
    fig, axes = plt.subplots(6, 2, figsize=(7.5, 16), dpi=150)
    models = data["models"]
    for row, (key, label, is_prop) in enumerate(MODEL_SPECS):
        for col, stop in enumerate(("0401", "0407")):
            block = models[key][f"compare_{stop}"]
            cm = _cm2x2(block)
            mcc = float(block.get("mcc", 0) or 0)
            sub_title = f"{label} · {STOP_LABELS[stop]}\nMCC={mcc:.3f}"
            if col == 0:
                sub_title = label + f"\n{STOP_LABELS[stop]}  MCC={mcc:.3f}"
            else:
                sub_title = f"{STOP_LABELS[stop]}  MCC={mcc:.3f}"
            _plot_confusion_on_ax(axes[row, col], cm, sub_title, is_proposed=is_prop)
    fig.suptitle("六模型两次停机混淆矩阵对照（行=金标，列=预测）", fontsize=13, y=1.005)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig12_confusion_matrices_6x2")


def _plot_confusion_individual(data: dict, out_dir: Path) -> None:
    """单模型×停机 PNG，便于 Word 排版。"""
    sub = out_dir / "confusion_matrices"
    sub.mkdir(parents=True, exist_ok=True)
    models = data["models"]
    for key, label, is_prop in MODEL_SPECS:
        for stop in ("0401", "0407"):
            block = models[key][f"compare_{stop}"]
            cm = _cm2x2(block)
            mcc = float(block.get("mcc", 0) or 0)
            fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=150)
            _plot_confusion_on_ax(
                ax,
                cm,
                f"{label} — {STOP_LABELS[stop]}（MCC={mcc:.3f}）",
                is_proposed=is_prop,
                show_colorbar=True,
            )
            stem = f"cm_{key}_{stop}"
            for ext in ("png", "pdf"):
                fig.savefig(sub / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)


def _plot_grouped_metric(df: pd.DataFrame, field: str, ylabel: str, out_dir: Path, stem: str) -> None:
    labels = [lbl for _, lbl, _ in MODEL_SPECS]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    for i, stop in enumerate(("0401", "0407")):
        sub = df[df["stop"] == stop].set_index("model_key")
        vals = [float(sub.loc[k, field]) if k in sub.index else 0.0 for k, _, _ in MODEL_SPECS]
        offset = -w / 2 if stop == "0401" else w / 2
        colors = [
            PROPOSED_COLOR if prop else BASELINE_COLOR for _, _, prop in MODEL_SPECS
        ]
        bars = ax.bar(x + offset, vals, width=w, label=STOP_LABELS[stop], color=colors, alpha=0.85 if stop == "0401" else 1.0, edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if v > 0.01:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.12)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"六模型 {ylabel} 对比（回路级，n=17）")
    fig.tight_layout()
    _save_fig(fig, out_dir, stem)


def _plot_tp_fp(df: pd.DataFrame, out_dir: Path) -> None:
    labels = [lbl for _, lbl, _ in MODEL_SPECS]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    for i, stop in enumerate(("0401", "0407")):
        sub = df[df["stop"] == stop].set_index("model_key")
        tp = [float(sub.loc[k, "tp"]) if k in sub.index else 0 for k, _, _ in MODEL_SPECS]
        fp = [float(sub.loc[k, "fp"]) if k in sub.index else 0 for k, _, _ in MODEL_SPECS]
        offset = -w / 2 if stop == "0401" else w / 2
        ax.bar(x + offset - 0.09, tp, width=w / 2 - 0.02, label=f"{STOP_LABELS[stop]} TP" if i == 0 else "_nolegend_", color="#70AD47", edgecolor="black", linewidth=0.4)
        ax.bar(x + offset + 0.09, fp, width=w / 2 - 0.02, label=f"{STOP_LABELS[stop]} FP" if i == 0 else "_nolegend_", color="#FF6B6B", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("回路数")
    ax.set_title("六模型 TP / FP 对比（越少 FP 越好，TP 越高越好）")
    handles = [
        Patch(facecolor="#70AD47", label="TP（命中异常）"),
        Patch(facecolor="#FF6B6B", label="FP（正常误报）"),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig06_tp_fp_comparison")


def _plot_mean_mcc_radar(df: pd.DataFrame, out_dir: Path) -> None:
    """两次停机平均 MCC 柱状（提出模型高亮）。"""
    labels = [lbl for _, lbl, _ in MODEL_SPECS]
    mcc_mean = []
    for key, _, prop in MODEL_SPECS:
        sub = df[df["model_key"] == key]["mcc"]
        mcc_mean.append(float(sub.mean()) if len(sub) else 0.0)
    colors = [PROPOSED_COLOR if prop else BASELINE_COLOR for _, _, prop in MODEL_SPECS]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.bar(range(len(labels)), mcc_mean, color=colors, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in mcc_mean], fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("两次停机平均 MCC")
    ax.set_ylim(0, max(mcc_mean) * 1.25 + 0.05)
    ax.set_title("六模型平均 MCC 对比（提出模型为红色）")
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig07_mean_mcc_highlight")


def _plot_heatmap(df: pd.DataFrame, field: str, title: str, out_dir: Path, stem: str) -> None:
    mat = np.zeros((len(MODEL_SPECS), 2))
    for i, (key, _, _) in enumerate(MODEL_SPECS):
        for j, stop in enumerate(("0401", "0407")):
            row = df[(df["model_key"] == key) & (df["stop"] == stop)]
            mat[i, j] = float(row[field].iloc[0]) if len(row) else np.nan
    fig, ax = plt.subplots(figsize=(5.5, 6), dpi=150)
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1)
    labels_y = [lbl for _, lbl, _ in MODEL_SPECS]
    ax.set_xticks([0, 1])
    ax.set_xticklabels([STOP_LABELS["0401"], STOP_LABELS["0407"]])
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    _save_fig(fig, out_dir, stem)


def _plot_table_image(df: pd.DataFrame, out_dir: Path, which: str) -> None:
    if which == "baseline":
        keys = {k for k, _, p in MODEL_SPECS if not p}
        title = "表4-1  五模型公平对照（统一阈值0.1）"
        fname = "table_4_1_baselines"
    elif which == "proposed":
        keys = {k for k, _, p in MODEL_SPECS if p}
        title = "表4-2  本文提出模型（FP=0约束标定）"
        fname = "table_4_2_proposed"
    else:
        keys = {k for k, _, _ in MODEL_SPECS}
        title = "表4-3  六模型完整指标对照"
        fname = "table_4_3_all_six_models"

    sub = df[df["model_key"].isin(keys)].copy()
    cols = ["model_label", "stop_label", "recall_fault", "precision_fault", "f1_fault", "balanced_accuracy", "mcc", "tp", "fp", "fn", "tn"]
    disp = sub[cols].copy()
    disp.columns = ["模型", "停机", "Recall", "Prec", "F1", "BalAcc", "MCC", "TP", "FP", "FN", "TN"]
    for c in disp.columns[2:8]:
        disp[c] = disp[c].map(lambda x: f"{float(x):.3f}")
    disp.to_csv(out_dir / f"{fname}.csv", index=False, encoding="utf-8-sig")

    fig_h = 0.45 + 0.32 * len(disp)
    fig, ax = plt.subplots(figsize=(14, fig_h), dpi=150)
    ax.axis("off")
    tbl = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)
    ax.set_title(title, fontsize=12, pad=16)
    fig.tight_layout()
    _save_fig(fig, out_dir, fname)


def _plot_six_panel_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """一张大图：MCC / F1 / Recall / Precision 四指标 × 两次停机。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    metrics = [
        ("mcc", "MCC"),
        ("f1_fault", "异常类 F1"),
        ("recall_fault", "异常类召回率"),
        ("precision_fault", "异常类精确率"),
    ]
    labels = [lbl for _, lbl, _ in MODEL_SPECS]
    x = np.arange(len(labels))
    w = 0.35
    for ax, (field, ylab) in zip(axes.flat, metrics):
        for stop, color in STOP_COLORS.items():
            sub = df[df["stop"] == stop].set_index("model_key")
            vals = [float(sub.loc[k, field]) if k in sub.index else 0 for k, _, _ in MODEL_SPECS]
            off = -w / 2 if stop == "0401" else w / 2
            ax.bar(x + off, vals, w, label=STOP_LABELS[stop], color=color, alpha=0.9, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(ylab)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(ylab)
    fig.suptitle("六模型主要指标对照（全量 80,481 窗；回路级 n=17）", fontsize=14, y=1.01)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig00_six_models_summary_4panel")


def _write_captions(out_dir: Path, json_path: Path) -> None:
    text = """图0  六模型主要指标四宫格对照（MCC、F1、召回率、精确率）。
图1  六模型异常类召回率对比。
图2  六模型异常类精确率对比。
图3  六模型异常类 F1 对比。
图4  六模型 MCC 对比。
图5  六模型平衡准确率对比。
图6  六模型 TP 与 FP 对比（绿色为命中异常，红色为误报正常）。
图7  六模型两次停机平均 MCC（提出模型以红色柱标示）。
图8  六模型 MCC 热力图。
图9  六模型 F1 热力图。
图10  4月1日停机六模型回路级混淆矩阵（2×3 子图）。
图11  4月7日停机六模型回路级混淆矩阵（2×3 子图）。
图12  六模型×两次停机混淆矩阵总览（6×2，适合附录）。
confusion_matrices/  各模型各停机单独混淆矩阵图（cm_<model>_<stop>.png）。

表4-1（图）五模型公平对照指标表。
表4-2（图）本文提出模型指标表。
表4-3（图）六模型完整指标表。

数据来源：""" + str(json_path.resolve())
    (out_dir / "figure_captions_6models_zh.txt").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    data = json.loads(args.json.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    _setup_chinese_font()

    df = _load_rows(data)
    df.to_csv(args.out / "six_models_metrics_flat.csv", index=False, encoding="utf-8-sig")

    _plot_six_panel_summary(df, args.out)
    for field, ylab in METRIC_FIELDS:
        stem = f"fig_{field}"
        _plot_grouped_metric(df, field, ylab, args.out, stem)

    _plot_tp_fp(df, args.out)
    _plot_mean_mcc_radar(df, args.out)
    _plot_heatmap(df, "mcc", "六模型 MCC 热力图", args.out, "fig08_heatmap_mcc")
    _plot_heatmap(df, "f1_fault", "六模型 F1 热力图", args.out, "fig09_heatmap_f1")

    _plot_table_image(df, args.out, "baseline")
    _plot_table_image(df, args.out, "proposed")
    _plot_table_image(df, args.out, "all")

    _plot_confusion_six_grid(data, "0401", args.out)
    _plot_confusion_six_grid(data, "0407", args.out)
    _plot_confusion_six_by_two_stops(data, args.out)
    _plot_confusion_individual(data, args.out)

    _write_captions(args.out, args.json)

    print("OK. 六模型图表已保存:", args.out.resolve())
    for p in sorted(args.out.glob("*")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
