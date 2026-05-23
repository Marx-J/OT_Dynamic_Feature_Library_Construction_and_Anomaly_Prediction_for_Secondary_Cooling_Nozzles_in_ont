"""
预答辩修订版：多模型对照实验一键运行（关闭 Hybrid 阈值调参，保证公平对照）。

在 thesis_code 目录、conda 环境 thesis_code 下执行：

  python run_thesis_presubmission_experiments.py smoke    # 约 2500 窗，数分钟
  python run_thesis_presubmission_experiments.py full     # 全量窗，较久

输出：
  outputs/retrain_combined/stage3_multimodel_{smoke|full}/
    - stage3_compare_metrics.json
    - stage3_circuit_predictions.csv
    - stage3_thesis_report.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT_0401 = Path(r"e:\毕设\资料\南钢连铸机资料\南钢5号连铸机4月1日停机二冷水堵漏检查对比报告.txt")
REPORT_0407 = Path(r"e:\毕设\资料\南钢连铸机资料\南钢5号连铸机4月7日停机二冷水堵漏检查对比报告.txt")

PRESETS = {
    "smoke": (2500, "stage3_multimodel_smoke"),
    "full": (None, "stage3_multimodel_full"),
}


def build_stage3_cmd(out_dir: Path, max_windows: int | None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "thesis_ot_featurelib",
        "run-stage3-predict-compare",
        "--stage1_stable_segments_json",
        str(ROOT / "outputs" / "retrain_combined" / "stage1" / "stable_segments.json"),
        "--stage2_featurelib_dir",
        str(ROOT / "outputs" / "retrain_combined" / "stage2_featurelib"),
        "--data_dir",
        str(ROOT / "outputs" / "retrain_combined" / "combined_ot_data"),
        "--report_txt_0401",
        str(REPORT_0401),
        "--report_txt_0407",
        str(REPORT_0407),
        "--gold_0401_csv",
        str(ROOT / "outputs" / "leak_report_0401.csv"),
        "--gold_0407_csv",
        str(ROOT / "outputs" / "leak_report_0407.csv"),
        "--out_dir",
        str(out_dir),
        "--hybrid-accuracy-target",
        "0",
        "--fault-alarm-threshold",
        "0.1",
        "--ml-circuit-aggregate",
        "noisy_or",
        "--ml-noisy-or-topk",
        "48",
    ]
    if max_windows is not None:
        cmd.extend(["--max_windows_debug", str(int(max_windows))])
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=sorted(PRESETS.keys()))
    args = ap.parse_args()

    cap, suffix = PRESETS[args.preset]
    out_dir = ROOT / "outputs" / "retrain_combined" / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_stage3_cmd(out_dir, cap)
    print("=== Stage3 多模型对照 ===")
    print("cwd:", ROOT)
    print("out:", out_dir)
    print("cmd:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        return rc

    report_cmd = [
        sys.executable,
        str(ROOT / "tools" / "export_stage3_thesis_report.py"),
        "--metrics_dir",
        str(out_dir),
    ]
    print("\n=== 生成论文对照报告 ===")
    print("cmd:", " ".join(report_cmd))
    rc2 = subprocess.call(report_cmd, cwd=str(ROOT))
    if rc2 == 0:
        print("\nOK. 请查看:", out_dir / "stage3_thesis_report.md")
    return rc2


if __name__ == "__main__":
    raise SystemExit(main())
