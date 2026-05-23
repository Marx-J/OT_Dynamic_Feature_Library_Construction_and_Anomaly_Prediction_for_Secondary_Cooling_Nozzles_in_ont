[README.md](https://github.com/user-attachments/files/28178200/README.md)
# 连铸二冷喷嘴 OT 动态特征库构建与异常预测研究
面向某钢厂连铸机二冷喷嘴回路，实现 **OT 数据治理 → 动态语义特征库 → 多模型回路级异常预测** 三阶段流水线，并在两次停机堵漏检查报告上对照评估。
本目录为附件包中的 **源代码**；同级的 `../数据集`、`../程序运行结果` 分别存放输入数据与 Stage1—Stage3 输出。

## 1. 附件目录结构
```
附件/
├── 源代码/                 ← 本 README 所在目录
├── 数据集/                 OT 测点 CSV（POC 原始 + 长周期折点合成，约 593 测点）
└── 程序运行结果/
    ├── Stage1_稳定浇铸片段/     stable_segments.json
    ├── Stage2_动态特征库/       windows_texts.csv、tfidf_* 等
    ├── Stage3_多模型全量实验/   论文主实验（80,481 窗）
    └── Stage3_烟雾测试/         快速验证（约 2,500 窗）
```

## 2. 实验流水线
| 阶段 | 功能 | 核心模块 | 主要输出 |
|------|------|----------|----------|
| 数据准备 | 停机报告解析、Excel 折线导出、多源 OT 合并 | `leak_report.py`、`xlsx_chart_export.py`、`combine_ot_sources.py` | 金标准 CSV、合并 OT 目录 |
| **Stage1** | 稳定浇铸片段抽取，剔除人工定阀等时段 | `stage1_segments.py` | `stable_segments.json` |
| **Stage2** | 1h 窗 / 0.5h 步长，工程语义文本 + TF-IDF(4096 维) | `stage2_dynamic_featurelib.py` | `windows_texts.csv`、`tfidf_vectors.npz` |
| **Stage3** | 五基线 + 提出模型，回路级二分类 vs 金标准 | `stage3_predict_compare.py` | `stage3_compare_metrics.json` 等 |
**弱监督训练标签**：窗内 `|ACT−TRG|/|TRG| > 0.2` 记为异常窗。  
**金标准验证**：2026-04-01、04-07 两次停机堵漏初检报告，回路级 n=17（Gold∩Pred 交集）。

## 3. Stage3 对照模型
| 模型键名 | 说明 | 公平对照 |
|----------|------|----------|
| `rule_rel_dev_max` | 相对偏差规则 + max 聚合 | 是（τ=0.1） |
| `lr_tabular_binary` | 数值特征 + L2 逻辑回归 | 是 |
| `rf_tabular_binary` | 数值特征 + 随机森林 | 是 |
| `hgb_tabular_binary` | 数值特征 + 直方图梯度提升 | 是 |
| `hybrid_tfidf_logreg_binary` | TF-IDF + 数值拼接 + 逻辑回归 | 是 |
| `proposed_conservative_rule_rf_blend` | 规则—RF 线性融合 + FP=0 约束标定（v2） | 否（允许结构/阈值搜索） |
公平基线统一：**noisy-or 聚合、top-K=48、回路告警阈值 0.1**。  
提出模型在 FP=0 可行域内优先最大化 TP，再优化 mean(MCC)。

## 4. 环境与依赖
- Python **3.10+**
- 安装依赖（在本目录执行）：
```bash
pip install -r requirements.txt
```
主要依赖：pandas、numpy、scikit-learn、scipy、matplotlib、joblib。

## 5. 目录说明（源代码）
```
源代码/
├── run_thesis_presubmission_experiments.py   # 一键运行 Stage3（smoke / full）
├── requirements.txt
├── configs/
│   └── stage1_segments_default.json        # Stage1 参数
├── thesis_ot_featurelib/                   # 核心 Python 包
│   ├── cli.py                              # 命令行入口
│   ├── stage1_segments.py
│   ├── stage2_dynamic_featurelib.py
│   ├── stage3_predict_compare.py
│   ├── combine_ot_sources.py
│   ├── xlsx_chart_export.py
│   ├── leak_report.py
│   └── io_nangang.py / tag_catalog.py / …
├── scripts/
│   ├── gen_thesis_figures_6models.py       # 六模型论文图/表
│   └── gen_thesis_figures_drawn.py         # 技术路线、框架示意图
└── tools/
    ├── export_stage3_thesis_report.py      # 生成 stage3_thesis_report.md
    └── pack_thesis_attachments.py          # 重新打包附件（可选）
```

## 6. 快速使用

### 6.1 仅查看已有结果
直接打开同级目录：
- `../程序运行结果/Stage3_多模型全量实验/stage3_thesis_report.md`
- `../程序运行结果/Stage3_多模型全量实验/stage3_compare_metrics.json`

### 6.2 重新运行 Stage3 多模型实验
代码默认在 **本目录下** 查找 `outputs/retrain_combined/` 中的 Stage1、Stage2 与合并 OT 数据。从附件包复现时，需先建立该目录结构（可将 `../程序运行结果` 与 `../数据集` 链入或复制）：
```text
源代码/outputs/retrain_combined/
├── stage1/stable_segments.json          ← 来自 ../程序运行结果/Stage1_稳定浇铸片段/
├── stage2_featurelib/                   ← 来自 ../程序运行结果/Stage2_动态特征库/
├── combined_ot_data/                    ← 来自 ../数据集/（全部测点 CSV）
├── leak_report_0401.csv                 ← 由停机报告解析得到
└── leak_report_0407.csv
```
在本目录执行：
```bash
python run_thesis_presubmission_experiments.py smoke    # 约 2,500 窗，数分钟
python run_thesis_presubmission_experiments.py full     # 全量 80,481 窗，耗时较长
```
输出：`outputs/retrain_combined/stage3_multimodel_{smoke|full}/`

> **注意**：`run_thesis_presubmission_experiments.py` 内停机报告路径默认为开发机绝对路径。复现前请改为本机 `../数据集/` 下报告 txt，或先运行 `parse-leak-report` 生成金标准 CSV。

### 6.3 分阶段 CLI
```bash
# 解析停机报告 → 金标准 CSV
python -m thesis_ot_featurelib parse-leak-report --report_txt <报告.txt> --out_csv outputs/leak_report_0401.csv
# Stage1：稳定片段
python -m thesis_ot_featurelib extract-stable-segments --data_dir ../数据集 --out_dir outputs/retrain_combined/stage1
# Stage2：动态特征库
python -m thesis_ot_featurelib build-dynamic-featurelib \
  --stable_segments_json outputs/retrain_combined/stage1/stable_segments.json \
  --data_dir ../数据集 --report_txt <二冷分析报告.txt> --out_dir outputs/retrain_combined/stage2_featurelib
# Stage3：多模型对照（或由 6.2 一键脚本调用）
python -m thesis_ot_featurelib run-stage3-predict-compare --help
```

### 6.4 生成论文图表
```bash
python scripts/gen_thesis_figures_6models.py --json outputs/retrain_combined/stage3_multimodel_full/stage3_compare_metrics.json
python scripts/gen_thesis_figures_drawn.py
python tools/export_stage3_thesis_report.py --metrics_dir outputs/retrain_combined/stage3_multimodel_full
```

## 7. 主实验结果摘要（全量）
| 指标 | 提出模型 v2 | 公平 RF 基线 |
|------|-------------|--------------|
| FP（两次停机合计） | 0 | 0 |
| TP（两次停机合计） | 5 | 4 |
| mean MCC | 0.272 | 0.236 |
| 4/7 停机 MCC | 0.342 | 0.270 |
详细数值见 `../程序运行结果/Stage3_多模型全量实验/`。
