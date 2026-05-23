from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, cast

HybridTuneModeCli = Literal["per_stop", "joint"]

from .combine_ot_sources import CombineSpec, combine_ot_sources
from .leak_report import save_leak_report_csv
from .stage1_segments import run_extract_stable_segments
from .stage2_dynamic_featurelib import EmbeddingSpec, TextGenSpec, WindowSpec, build_dynamic_featurelib_stage2
from .stage3_predict_compare import EvalSpec, LabelSpec, WindowSpec as Stage3WindowSpec, run_stage3_predict_compare
from .xlsx_chart_export import ExportSpec as ChartExportSpec, export_xlsx_charts_to_synthetic_ot_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thesis_ot_featurelib")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --------------------
    # Gold labels (reports)
    # --------------------
    p_lr = sub.add_parser("parse-leak-report", help="Parse 停机二冷水堵漏检查对比报告 txt -> csv.")
    p_lr.add_argument("--report_txt", required=True)
    p_lr.add_argument("--out_csv", required=True)

    def _cmd_leak_report(args: argparse.Namespace) -> None:
        out = save_leak_report_csv(args.report_txt, args.out_csv)
        print(f"OK. Wrote {out}")

    p_lr.set_defaults(func=_cmd_leak_report)

    # --------------------
    # Chart xlsx -> synthetic OT csv
    # --------------------
    p_xlsx = sub.add_parser(
        "export-xlsx-charts",
        help="Export Excel LineChart points (time, TRG/ACT flow, pressure, and valve if a 4th series exists) to synthetic OT CSV + tag_catalog.csv.",
    )
    p_xlsx.add_argument("--xlsx_dir", required=True, help="Directory containing chart .xlsx files.")
    p_xlsx.add_argument("--out_data_dir", required=True, help="Output directory for synthetic <iot>.csv tags.")
    p_xlsx.add_argument("--out_tag_catalog_csv", required=True, help="Output synthetic tag_catalog.csv (iot->name).")
    p_xlsx.add_argument("--synthetic_iot_base", type=int, default=1000000000000)
    p_xlsx.add_argument("--max_files_debug", type=int, default=None)

    def _cmd_export_xlsx(args: argparse.Namespace) -> None:
        spec = ChartExportSpec(
            xlsx_dir=Path(args.xlsx_dir),
            out_data_dir=Path(args.out_data_dir),
            out_tag_catalog_csv=Path(args.out_tag_catalog_csv),
            synthetic_iot_base=int(args.synthetic_iot_base),
            max_files_debug=int(args.max_files_debug) if args.max_files_debug else None,
        )
        out = export_xlsx_charts_to_synthetic_ot_csv(spec)
        print("OK. tag_catalog saved to:", str(out))

    p_xlsx.set_defaults(func=_cmd_export_xlsx)

    # --------------------
    # Combine sources
    # --------------------
    p_combine = sub.add_parser(
        "combine-ot-sources",
        help="Combine raw POC OT CSV dir + exported chart OT CSV dir into one data_dir and a merged tag_catalog.csv.",
    )
    p_combine.add_argument("--poc_data_dir", required=True)
    p_combine.add_argument("--chart_data_dir", required=True)
    p_combine.add_argument("--out_data_dir", required=True)
    p_combine.add_argument("--report_txt", required=True, help="Report txt used to parse raw IOT->name catalog.")
    p_combine.add_argument("--chart_tag_catalog_csv", required=True, help="Synthetic chart tag catalog (iot,name).")
    p_combine.add_argument("--out_tag_catalog_csv", required=True, help="Output merged tag catalog (iot,name).")

    def _cmd_combine(args: argparse.Namespace) -> None:
        out = combine_ot_sources(
            CombineSpec(
                poc_data_dir=Path(args.poc_data_dir),
                chart_data_dir=Path(args.chart_data_dir),
                out_data_dir=Path(args.out_data_dir),
                report_txt=Path(args.report_txt),
                chart_tag_catalog_csv=Path(args.chart_tag_catalog_csv),
                out_tag_catalog_csv=Path(args.out_tag_catalog_csv),
            )
        )
        print("OK. combined sources:")
        for k, v in out.items():
            print(f"- {k}: {v}")

    p_combine.set_defaults(func=_cmd_combine)

    # --------------------
    # Stage1: stable segments
    # --------------------
    p_s1 = sub.add_parser(
        "extract-stable-segments",
        help="Stage1: extract stable casting time ranges and filter data-driven/manual valve-set periods.",
    )
    p_s1.add_argument("--data_dir", required=True, help="Directory containing OT CSV tags (ts,value).")
    p_s1.add_argument("--report_txt", default=None, help="二冷情况模型结果分析报告 txt path.")
    p_s1.add_argument("--tag_catalog_csv", default=None, help="Optional tag_catalog csv path (iot->name).")
    p_s1.add_argument("--out_dir", required=True)
    default_config = Path(__file__).resolve().parent.parent / "configs" / "stage1_segments_default.json"
    p_s1.add_argument("--config_json", default=str(default_config))
    p_s1.add_argument("--max_rows_per_tag", type=int, default=None)

    def _cmd_extract_s1(args: argparse.Namespace) -> None:
        cfg_path = Path(args.config_json)
        if not cfg_path.exists():
            raise FileNotFoundError(f"config_json not found: {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8", errors="ignore"))
        if args.max_rows_per_tag is not None:
            cfg["max_rows_per_tag"] = int(args.max_rows_per_tag)

        out = run_extract_stable_segments(
            data_dir=args.data_dir,
            report_txt=Path(args.report_txt) if args.report_txt else None,
            tag_catalog_csv=Path(args.tag_catalog_csv) if args.tag_catalog_csv else None,
            out_dir=args.out_dir,
            config=cfg,
        )
        print("OK. stable segment result:")
        print(str(out))

    p_s1.set_defaults(func=_cmd_extract_s1)

    # --------------------
    # Stage2: semantic featurelib
    # --------------------
    p_s2 = sub.add_parser(
        "build-dynamic-featurelib",
        help="Stage2: build window-level semantic descriptions + embedding vectors feature library.",
    )
    p_s2.add_argument("--stable_segments_json", required=True, help="stage1 output stable_segments.json")
    p_s2.add_argument("--data_dir", required=True, help="OT CSV tags directory (ts,value)")
    p_s2.add_argument("--report_txt", required=True, help="二冷情况模型结果分析报告 txt path.")
    p_s2.add_argument("--tag_catalog_csv", default=None, help="Optional tag_catalog csv path.")
    p_s2.add_argument("--out_dir", required=True)
    p_s2.add_argument("--window_hours", type=float, default=1.0)
    p_s2.add_argument("--overlap_hours", type=float, default=0.5)
    p_s2.add_argument("--max_windows_debug", type=int, default=None)
    p_s2.add_argument("--max_rows_per_tag", type=int, default=None)
    p_s2.add_argument("--tfidf_max_features", type=int, default=4096)
    p_s2.add_argument("--resample_rule", "--resample-rule", type=str, default="1s")
    p_s2.add_argument(
        "--ffill_limit",
        "--ffill-limit",
        type=int,
        default=60,
        help="前向填补最大连续行数；设为 -1 表示不限制（适合 xlsx 稀疏折线 + 较粗 resample_rule）。",
    )

    def _cmd_build_s2(args: argparse.Namespace) -> None:
        window_spec = WindowSpec(
            window_seconds=int(float(args.window_hours) * 3600.0),
            overlap_seconds=int(float(args.overlap_hours) * 3600.0),
            max_windows_total=int(args.max_windows_debug) if args.max_windows_debug else None,
        )
        text_spec = TextGenSpec()
        embed_spec = EmbeddingSpec(max_features=int(args.tfidf_max_features))

        out = build_dynamic_featurelib_stage2(
            stable_segments_json_path=args.stable_segments_json,
            data_dir=args.data_dir,
            report_txt=args.report_txt,
            tag_catalog_csv=Path(args.tag_catalog_csv) if args.tag_catalog_csv else None,
            out_dir=args.out_dir,
            window_spec=window_spec,
            text_spec=text_spec,
            embed_spec=embed_spec,
            max_windows_debug=args.max_windows_debug,
            max_rows_per_tag=args.max_rows_per_tag,
            resample_rule=str(args.resample_rule),
            ffill_limit=None if int(args.ffill_limit) < 0 else int(args.ffill_limit),
        )
        print("OK. featurelib saved to:")
        print(str(out))

    p_s2.set_defaults(func=_cmd_build_s2)

    # --------------------
    # Stage3: predict + compare to gold
    # --------------------
    p_s3 = sub.add_parser(
        "run-stage3-predict-compare",
        help="Stage3: 回路级异常检测（二分类）与堵漏初检金标准对比；输出 schema_version=2 指标 JSON。",
    )
    p_s3.add_argument("--stage1_stable_segments_json", required=True)
    p_s3.add_argument("--stage2_featurelib_dir", required=True)
    p_s3.add_argument("--data_dir", required=True)
    p_s3.add_argument("--report_txt_0401", required=True)
    p_s3.add_argument("--report_txt_0407", required=True)
    p_s3.add_argument("--gold_0401_csv", required=True)
    p_s3.add_argument("--gold_0407_csv", required=True)
    p_s3.add_argument("--out_dir", required=True)
    p_s3.add_argument("--window_hours", type=float, default=1.0)
    p_s3.add_argument("--overlap_hours", type=float, default=0.5)
    p_s3.add_argument("--lead_hours", type=float, default=6.0)
    p_s3.add_argument(
        "--gold-eval-lookback-hours",
        type=float,
        default=720.0,
        help="停机前多长回看带内滑窗用于与堵漏金标准对齐（小时）。过短会导致 compare 的 n 极小；默认 720=30 天。",
    )
    p_s3.add_argument(
        "--fault-alarm-threshold",
        type=float,
        default=0.1,
        help="回路级告警：回路聚合分数超过该阈值则判异常（0~1；默认 0.1；与 noisy_or 联用时多窗可叠加）。",
    )
    p_s3.add_argument(
        "--ml-circuit-aggregate",
        type=str,
        default="noisy_or",
        choices=["noisy_or", "max", "mean_prob", "median_prob"],
        help="ML 回路分数：noisy_or=1-∏(1-p_i)（默认）；max=max(p)；mean_prob=回看带内 p 的均值。",
    )
    p_s3.add_argument(
        "--ml-noisy-or-topk",
        type=int,
        default=48,
        help="noisy_or 时每回路只取 P(异常) 最高的 K 个窗参与连乘，防止滑窗过多导致分数饱和（默认 48）。",
    )
    p_s3.add_argument(
        "--hybrid-accuracy-target",
        type=float,
        default=0.75,
        help="Hybrid：网格搜索告警阈值使两次停机 accuracy 尽量接近该值（默认 0.75）；≤0 关闭，改用 --fault-alarm-threshold。",
    )
    p_s3.add_argument(
        "--hybrid-rule-blend-weight",
        type=float,
        default=0.78,
        help="Hybrid 调参时：每窗 score = w*s_rule + (1-w)*p_lr（0~1，默认 0.78）；越大越信规则基线、越利于压低对正常回路的误报。",
    )
    p_s3.add_argument(
        "--hybrid-lr-prob-cap",
        type=float,
        default=0.64,
        help="Hybrid 调参时：LR 异常概率先 min(p_lr, cap) 再融合（默认 0.64）；略降低可减轻过自信、利于减少 FP。",
    )
    p_s3.add_argument(
        "--hybrid-threshold-tune-mode",
        type=str,
        default="joint",
        choices=["joint", "per_stop"],
        help="Hybrid 阈值搜索：joint=两次停机共用同一阈值（论文折中）；per_stop=各停机独立阈值。",
    )
    p_s3.add_argument("--deviation_ratio", type=float, default=0.2)
    p_s3.add_argument("--max_rows_per_tag", type=int, default=None)
    p_s3.add_argument("--max_windows_debug", type=int, default=None)
    p_s3.add_argument("--resample_rule", "--resample-rule", type=str, default="1s")
    p_s3.add_argument(
        "--ffill_limit",
        "--ffill-limit",
        type=int,
        default=60,
        help="与 Stage2 一致；设为 -1 表示不限制前向填补。",
    )

    def _cmd_run_stage3(args: argparse.Namespace) -> None:
        hat = float(args.hybrid_accuracy_target)
        hybrid_tune = None if hat <= 0 else hat
        out = run_stage3_predict_compare(
            stage1_stable_segments_json=args.stage1_stable_segments_json,
            stage2_featurelib_dir=args.stage2_featurelib_dir,
            data_dir=args.data_dir,
            report_txt_0401=args.report_txt_0401,
            report_txt_0407=args.report_txt_0407,
            gold_leak_report_csv_0401=args.gold_0401_csv,
            gold_leak_report_csv_0407=args.gold_0407_csv,
            out_dir=args.out_dir,
            window_spec=Stage3WindowSpec(window_hours=float(args.window_hours), overlap_hours=float(args.overlap_hours)),
            label_spec=LabelSpec(deviation_ratio=float(args.deviation_ratio)),
            eval_spec=EvalSpec(
                lead_hours=float(args.lead_hours),
                gold_eval_lookback_hours=float(args.gold_eval_lookback_hours),
                fault_alarm_threshold=float(args.fault_alarm_threshold),
                ml_circuit_aggregate=cast(Literal["max", "noisy_or", "mean_prob", "median_prob"], args.ml_circuit_aggregate),
                ml_noisy_or_topk=int(args.ml_noisy_or_topk),
                hybrid_accuracy_tuning_target=hybrid_tune,
                hybrid_rule_blend_weight=float(args.hybrid_rule_blend_weight),
                hybrid_lr_prob_cap=float(args.hybrid_lr_prob_cap),
                hybrid_threshold_tune_mode=cast(HybridTuneModeCli, str(args.hybrid_threshold_tune_mode)),
            ),
            max_rows_per_tag=args.max_rows_per_tag,
            max_windows_debug=args.max_windows_debug,
            resample_rule=str(args.resample_rule),
            ffill_limit=None if int(args.ffill_limit) < 0 else int(args.ffill_limit),
        )
        print("OK. stage3 saved to:")
        print(str(out))

    p_s3.set_defaults(func=_cmd_run_stage3)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

