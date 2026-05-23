"""
从 Stage3 输出生成论文用「多模型对照表 + 典型成功/失败案例」Markdown。

用法（thesis_code 目录）：
  python tools/export_stage3_thesis_report.py --metrics_dir outputs/retrain_combined/stage3_multimodel_full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "rule_rel_dev_max": "规则基线（相对偏差阈值）",
    "lr_tabular_binary": "逻辑回归（数值特征）",
    "rf_tabular_binary": "随机森林（数值特征）",
    "hgb_tabular_binary": "直方图梯度提升（数值特征）",
    "hybrid_tfidf_logreg_binary": "TF-IDF+数值 逻辑回归（语义混合）",
    "proposed_conservative_rule_rf_blend": "【本文提出】规则+RF/HGB保守融合 v2（FP=0，TP→MCC标定）",
}

BASELINE_MODEL_KEYS = (
    "rule_rel_dev_max",
    "lr_tabular_binary",
    "rf_tabular_binary",
    "hgb_tabular_binary",
    "hybrid_tfidf_logreg_binary",
)


def _load_metrics(metrics_dir: Path) -> dict:
    p = metrics_dir / "stage3_compare_metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _metrics_table(metrics: dict) -> str:
    lines = [
        "| 模型 | 停机 | n | Recall(异常) | Precision(异常) | F1(异常) | BalAcc | MCC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in MODEL_LABELS.items():
        lane = metrics.get("models", {}).get(key, {})
        for stop, tag in (("compare_0401", "4/1"), ("compare_0407", "4/7")):
            m = lane.get(stop, {})
            if not m or int(m.get("n_circuits", 0) or 0) == 0:
                continue
            lines.append(
                f"| {label} | {tag} | {m['n_circuits']} | "
                f"{m.get('recall_fault', 0):.3f} | {m.get('precision_fault', 0):.3f} | "
                f"{m.get('f1_fault', 0):.3f} | {m.get('balanced_accuracy', 0):.3f} | "
                f"{m.get('mcc', 0):.3f} |"
            )
    return "\n".join(lines) + "\n"


def _pick_cases(circuit_csv: Path, model_key: str = "hybrid", per_type: int = 2) -> str:
    df = pd.read_csv(circuit_csv, encoding="utf-8-sig", dtype={"stop": str})
    df["stop"] = df["stop"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    case_col = f"case_{model_key}"
    if case_col not in df.columns:
        return f"（未找到列 {case_col}）\n"
    parts: list[str] = []
    for stop in sorted(df["stop"].astype(str).unique()):
        sub = df[df["stop"].astype(str) == stop]
        parts.append(f"\n### 停机 {stop}（以 {model_key} 模型为例）\n")
        for tag, title in (
            ("TP", "典型成功案例：命中异常回路"),
            ("TN", "典型成功案例：正确识别正常回路"),
            ("FP", "典型失败案例：正常回路误报"),
            ("FN", "典型失败案例：异常回路漏报"),
        ):
            rows = sub[sub[case_col] == tag].head(per_type)
            if rows.empty:
                parts.append(f"- **{title}**：本次输出中无 {tag} 样本。\n")
                continue
            parts.append(f"- **{title}**：\n")
            for _, r in rows.iterrows():
                parts.append(
                    f"  - 回路 `{r['circuit_key']}`：金标={'异常' if int(r['gold_fault']) == 1 else '正常'}，"
                    f"预测={'异常' if int(r[f'pred_{model_key}']) == 1 else '正常'}\n"
                )
    return "".join(parts)


def _io_block(metrics: dict) -> str:
    io = metrics.get("io_summary", {})
    if not io:
        return ""
    ins = io.get("inputs", {})
    outs = io.get("outputs", {})
    lines = ["## 实验输入与输出（论文可直接引用）\n", "**输入**\n"]
    for k, v in ins.items():
        lines.append(f"- {k}：{v}\n")
    lines.append("\n**输出**\n")
    for k, v in outs.items():
        lines.append(f"- {k}：{v}\n")
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", type=Path, required=True)
    ap.add_argument("--case_model", type=str, default="proposed", help="案例剖析选用的模型列前缀")
    ap.add_argument("--out_md", type=Path, default=None)
    args = ap.parse_args()

    metrics_dir = args.metrics_dir.resolve()
    metrics = _load_metrics(metrics_dir)
    circuit_csv = metrics_dir / "stage3_circuit_predictions.csv"

    md_parts = [
        "# Stage3 多模型对照实验报告（自动生成）\n\n",
        _io_block(metrics),
        "\n## 表4-1 五模型公平对照（统一阈值 0.1）\n\n",
        "说明：五类基线模型在相同回路级评估协议下对比。\n\n",
        _metrics_table({**metrics, "models": {k: metrics.get("models", {}).get(k, {}) for k in BASELINE_MODEL_KEYS}}),
        "\n## 表4-2 本文提出模型（规则+RF 保守融合，FP=0 约束下 MCC 标定）\n\n",
        _metrics_table(
            {
                **metrics,
                "models": {
                    "proposed_conservative_rule_rf_blend": metrics.get("models", {}).get(
                        "proposed_conservative_rule_rf_blend", {}
                    )
                },
            }
        ),
    ]
    prop = metrics.get("models", {}).get("proposed_conservative_rule_rf_blend", {})
    if prop.get("search_meta"):
        sm = prop["search_meta"]
        md_parts.append(
            f"\n**v2 最优结构**：mode={sm.get('blend_mode')}, w_rule={sm.get('w_rule')}, "
            f"w_hgb={sm.get('w_hgb')}, agg={sm.get('circuit_aggregate')}, topK={sm.get('ml_noisy_or_topk')}, "
            f"阈值模式={sm.get('threshold_mode')}, thr_0401/0407="
            f"{prop.get('fault_alarm_threshold_used')}\n"
        )
    bc = prop.get("baseline_comparison", {})
    if bc:
        md_parts.append(
            f"\n**相对五基线**：mean_MCC={bc.get('proposed_mean_mcc', 0):.3f}, "
            f"TP合计={bc.get('proposed_tp_sum')}, FP合计={bc.get('proposed_fp_sum')}, "
            f"mean_MCC优于全部基线={bc.get('strictly_beats_all_on_mean_mcc')}\n"
        )
    if circuit_csv.is_file():
        md_parts.append("\n## 典型成功与失败案例\n\n")
        md_parts.append(_pick_cases(circuit_csv, model_key=args.case_model))

    out_md = args.out_md or (metrics_dir / "stage3_thesis_report.md")
    out_md.write_text("".join(md_parts), encoding="utf-8")
    print("Wrote:", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
