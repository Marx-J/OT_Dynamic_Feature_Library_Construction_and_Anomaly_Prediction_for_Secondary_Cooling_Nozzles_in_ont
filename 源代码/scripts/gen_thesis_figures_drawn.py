"""
生成论文「自绘」示意图：图1-1 技术路线、图2-1 数据规模、图3-1 特征库框架。

用法:
  python scripts/gen_thesis_figures_drawn.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "retrain_combined" / "thesis_figures_drawn"


def _setup_chinese_font() -> None:
    for c in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "SimSun"]:
        if c in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [c]
            plt.rcParams["font.family"] = "sans-serif"
            break
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc="#E7EEF7", ec="#2F5597", fontsize=9):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#404040",
        )
    )


def plot_fig1_1() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, 1.0, 1.8, 1.0, "多源 OT\n时序数据\n(POC+折点)"),
        (2.4, 1.0, 1.8, 1.0, "Stage1\n稳定浇铸\n片段抽取"),
        (4.6, 1.0, 1.8, 1.0, "Stage2\n动态语义\n特征库"),
        (6.8, 1.0, 1.8, 1.0, "Stage3\n多模型\n回路预测"),
        (9.0, 1.0, 2.0, 1.0, "停机堵漏初检\n金标准验证\n(n=17)"),
    ]
    for x, y, w, h, t in boxes:
        _box(ax, (x, y), w, h, t)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        _arrow(ax, x1 + 0.05, 1.5, x2 - 0.05, 1.5)
    ax.text(6, 0.35, "第2章 → 第3章 → 第4章", ha="center", fontsize=9, color="#555555")
    ax.set_title("图1-1  技术路线示意图", fontsize=12, pad=8)
    fig.tight_layout()
    _save(fig, "fig1_1_technical_route")


def plot_fig2_1() -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    labels = ["Stage3\n滑窗总数", "单次停机\n金标回路", "Gold∩Pred\n可评估回路"]
    vals = [80481, 49, 17]
    colors = ["#5B9BD5", "#FFC000", "#C55A11"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
    ax.bar_label(bars, labels=[f"{v:,}" if v > 1000 else str(v) for v in vals], fontsize=11)
    ax.set_ylabel("数量")
    ax.set_title("图2-1  数据规模与评估回路数量示意", fontsize=12)
    ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout()
    _save(fig, "fig2_1_data_scale")


def plot_fig3_1() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.5), dpi=150)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    items = [
        (0.3, 1.2, 1.7, 1.1, "稳定片段\n输入"),
        (2.3, 1.2, 1.7, 1.1, "1h窗\n0.5h步长"),
        (4.3, 1.2, 1.7, 1.1, "工程语义\n文本生成"),
        (6.3, 1.2, 1.7, 1.1, "TF-IDF\n4096维"),
        (8.3, 1.2, 2.0, 1.1, "特征库持久化\n文本+向量+元数据"),
    ]
    for x, y, w, h, t in items:
        _box(ax, (x, y), w, h, t, fc="#E2F0D9", ec="#548235")
    for i in range(len(items) - 1):
        x1 = items[i][0] + items[i][2]
        x2 = items[i + 1][0]
        _arrow(ax, x1 + 0.05, 1.75, x2 - 0.05, 1.75)
    ax.text(5.5, 0.45, "全量规模：80,481 窗口 × 4,096 维（第3章）", ha="center", fontsize=9, color="#444444")
    ax.set_title("图3-1  OT 动态语义特征库构建框架", fontsize=12, pad=8)
    fig.tight_layout()
    _save(fig, "fig3_1_featurelib_framework")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _setup_chinese_font()
    plot_fig1_1()
    plot_fig2_1()
    plot_fig3_1()
    print("OK:", OUT.resolve())
    for p in sorted(OUT.glob("fig*")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
