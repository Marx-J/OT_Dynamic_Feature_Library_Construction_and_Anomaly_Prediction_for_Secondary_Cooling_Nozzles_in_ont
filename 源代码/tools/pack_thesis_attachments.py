"""打包毕设附件到 e:\\毕设\\附件资料"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(r"e:\毕设")
DEST = ROOT / "附件资料"
THESIS = ROOT / "thesis_code"
OUT = THESIS / "outputs"
RC = OUT / "retrain_combined"


def robocopy(src: Path, dst: Path, *, exclude_dirs: list[str] | None = None) -> None:
    if not src.exists():
        print("SKIP missing:", src)
        return
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["robocopy", str(src), str(dst), "/E", "/MT:8", "/R:2", "/W:3", "/NFL", "/NDL", "/NJH", "/NJS", "/nc", "/ns", "/np"]
    for d in exclude_dirs or []:
        cmd.extend(["/XD", d])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode >= 8:
        raise RuntimeError(f"robocopy failed {src} -> {dst} code={r.returncode}")


def copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    for name in ("源代码", "数据集", "程序运行结果"):
        (DEST / name).mkdir(parents=True, exist_ok=True)

    print("[1/8] 源代码")
    robocopy(THESIS, DEST / "源代码" / "thesis_code", exclude_dirs=["outputs", "__pycache__", ".git", ".idea", ".pytest_cache"])

    print("[2/8] D1 报告")
    robocopy(ROOT / "资料" / "南钢连铸机资料", DEST / "数据集" / "D1_停机检查报告与二冷分析报告")

    print("[3/8] D2 POC (大文件)")
    data_root = ROOT / "数据集"
    if data_root.exists():
        for d in data_root.iterdir():
            if d.is_dir() and "POC" in d.name:
                robocopy(d, DEST / "数据集" / "D2_POC短周期OT原始测点" / d.name)

    print("[4/8] D3 折点导出")
    d3 = DEST / "数据集" / "D3_长周期图表折点OT导出"
    robocopy(OUT / "xlsx_chart_ot", d3 / "xlsx_chart_ot")
    copy_file(OUT / "xlsx_chart_tag_catalog.csv", d3 / "xlsx_chart_tag_catalog.csv")
    copy_file(OUT / "xlsx_chart_ot" / "xlsx_export_circuit_to_tag.json", d3 / "xlsx_export_circuit_to_tag.json")

    print("[5/8] D4 合并 OT (大文件)")
    d4 = DEST / "数据集" / "D4_合并统一OT数据集_593测点"
    robocopy(RC / "combined_ot_data", d4 / "combined_ot_data")
    copy_file(RC / "combined_tag_catalog.csv", d4 / "combined_tag_catalog.csv")

    res = DEST / "程序运行结果"
    print("[6/8] R1-R5 实验与图表")
    robocopy(RC / "stage1", res / "R1_Stage1_稳定浇铸片段")
    robocopy(RC / "stage2_featurelib", res / "R2_Stage2_动态特征库")
    robocopy(RC / "stage3_multimodel_full", res / "R3_Stage3_多模型全量实验")
    robocopy(RC / "stage3_multimodel_smoke", res / "R4_Stage3_烟雾测试")
    robocopy(RC / "thesis_figures_6models", res / "R5_论文图表_六模型对照")
    drawn = OUT / "retrain_combined" / "thesis_figures_drawn"
    if not drawn.exists():
        drawn = RC / "thesis_figures_drawn"
    if drawn.exists():
        robocopy(drawn, res / "R5_论文图表_技术路线与框架图")
    if (RC / "thesis_figures").exists():
        robocopy(RC / "thesis_figures", res / "R5_论文图表_早期对照图")

    print("[7/8] R6 写作文档")
    wdir = res / "R6_写作与说明文档"
    wdir.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.txt"):
        copy_file(f, wdir / f.name)
    for f in OUT.glob("*.md"):
        copy_file(f, wdir / f.name)
    copy_file(DEST / "目录说明.txt", wdir / "附件资料目录说明.txt")

    print("[8/8] R7 其他实验")
    other = res / "R7_其他Stage3实验记录"
    for sub in (
        "stage3_preset_full",
        "stage3_latest_run",
        "stage3_fp_reduce",
        "stage3_full_hybrid_objective",
        "stage3_full_hybrid_objective_v2",
    ):
        p = RC / sub
        if p.exists():
            robocopy(p, other / sub)

    print("DONE:", DEST.resolve())


if __name__ == "__main__":
    main()
