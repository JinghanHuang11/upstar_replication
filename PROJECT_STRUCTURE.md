# 项目文件组织结构

> **最后更新**: 2026-03-30
> **代码规模**: 30 个 Python 源文件 | 5 个配置文件 | 8 个活跃脚本
> **论文对齐**: ✅ 工程清理完成（cv10-only，统一入口，论文对齐）
> **代码状态**: ✅ Ready for GPU experiments

---

## 目录结构

```
upstar/
├── configs/              # 配置文件 (5个) - 已收敛为最小集合
│   ├── tafeng_baseline.yaml       # ⭐ Baseline 训练配置
│   ├── tafeng_upstar.yaml         # ⭐ UPSTAR 训练配置
│   ├── item_repr.yaml             # Item 表示学习配置
│   ���── stb.yaml                   # STB 计算配置
│   └── tafeng_eval.yaml           # 评估配置
│
├── data/                 # 数据目录
│   ├── raw/
│   │   └── ta_feng.csv            # 原始 Tafeng 数据
│   ├── processed/
│   │   └── tafeng/                # Tafeng 处理结果
│   │       ├── metadata.pkl       # 词表
│   │       └── cv_splits/         # 10-fold CV 划分
│   └���─ cache/
│       └── tafeng/                # 图缓存
│
├── docs/                 # 详细文档
│   ├── SMOKE_TEST_GUIDE.md        # ⭐ Smoke test 指南
│   ├── CLEANUP_GUIDE.md            # 清理指南
│   ├── CLEANUP_SUMMARY.md          # 清理总结
│   ├── CONFIGS_CLEANUP_REPORT.md   # 配置清理报告
│   ├── SCRIPTS_CLEANUP_REPORT.md   # 脚本清理报告
│   ├── MODE_UNIFICATION_CLEANUP.md  # 模式统一报告
│   ├── RUN_FULL_EXPERIMENT_REFACTOR.md # 总控重构报告
│   └── DEPRECATED_FILES_CLEANUP.md # 废弃文件清理报告
│
├── scripts/              # 运行脚本 (8个)
│   ├── run_preprocess.sh            # ⭐ 数据预处理
│   ├── run_tafeng_baseline.sh      # ⭐ Baseline 训练
│   ├── run_tafeng_upstar.sh        # ⭐ UPSTAR 训练
│   ├── run_item_repr.sh            # Item 表示学习
│   ├── run_stb.sh                  # STB 计算
│   ├── compare_results.sh          # 结果比较
│   ├── smoke_test.sh               # ⭐ Smoke test 脚本
│   ├── clean_old_data.sh           # 清理工具
│   └── test_baseline_eval.sh       # 测试工具
│
├── src/                  # 源代码 (30个 Python 文件)
│   ├── data/             # 数据处理
│   │   ├── preprocess.py              # 数据预处理
│   │   ├── build_sequences.py         # 序列构建
│   │   └── dataloader.py               # 数据加载
│   ├── graphs/           # 图构建
│   │   ├── item_graph.py              # Item 图构建
│   │   ├── item_time_graph.py         # Item-Time 图构建
│   │   └── perturbation.py            # 图扰动
│   ├── models/           # 模型
│   │   ├── baseline_lstm.py           # Baseline LSTM
│   │   ├── upstar.py                  # UPSTAR 模型
│   │   ├── sequence_models.py        # 序列模型
│   │   ├── fusion.py                  # 融合模块
│   │   └── item_gnn.py                # Item-GNN
│   ├── training/         # 训练
│   │   ├── cross_validation_baseline.py # ⭐ Baseline CV10 实现
│   │   ├── train_baseline.py           # Baseline 薄封装
│   │   ├── train_upstar_cv.py          # ⭐ UPSTAR CV10 实现
│   │   ├── train_upstar.py             # UPSTAR 薄封装
│   │   ├── train_item_repr.py          # Item 表示训练
│   │   ├── train_stb.py                # STB 计算
│   │   └── losses.py                   # 损失函数
│   ├── evaluation/       # 评估
│   │   ├── metrics.py                  # 评估指标
│   │   ├── evaluator.py                # 评估器
│   │   ├── report.py                   # 报告生成
│   │   ├── comparison.py               # 结果比较
│   │   └── evaluate.py                 # 评估入口
│   ├── experiments/      # 实验管理（仅测试使用）
│   └── utils/            # 工具
│
├── outputs/              # 输出目录
│   ├── phase1_baseline/              # Baseline 结果
│   │   └── results/
│   │       └── cv_results.json       # ⭐ CV10 结果
│   ├── phase2_item_repr/             # Item 表示
│   │   └── checkpoints/
│   │       └── item_embeddings.pt    # ⭐ Embeddings
│   ├── phase3_stb/                   # STB 结果
│   │   └── checkpoints/
│   │       ├── stb_scores.npy        # ⭐ STB 分数
│   │       └── motivation_labels.npy # ⭐ 标签
│   ├── phase4_upstar/                # UPSTAR 结果
│   │   └── results/
│   │       └── cv_results.json       # ⭐ CV10 结果
│   └─��� comparison/                   # 比较结果
│
├── tests/                # 测试脚本
├── RUN_EXPERIMENTS.md    # ⭐ 运行指南
├── README.md             # 项目说明
├── PROJECT_STRUCTURE.md   # 本文件
├── PROJECT_ORGANIZATION.md
├── run_full_experiment.py # ⭐ 总控脚本
└── requirements.txt
```

---

## 核心文件说明

### 训练代码

| 文件 | 功能 | 状态 |
|------|------|------|
| `cross_validation_baseline.py` | Baseline 10-fold CV 实现 | ⭐ 唯一实现 |
| `train_baseline.py` | Baseline 薄封装 | 调用上方 |
| `train_upstar_cv.py` | UPSTAR 10-fold CV 实现 | ⭐ 唯一实现 |
| `train_upstar.py` | UPSTAR 薄封装 | 调用上方 |
| `train_item_repr.py` | Item 表示学习 | 独立实现 |
| `train_stb.py` | STB 计算 | 独立实现 |

### 配置文件

| 文件 | 用途 | 关键参数 |
|------|------|----------|
| `tafeng_baseline.yaml` | Baseline 训练 | `metric_for_best: 'NDCG@20'` |
| `tafeng_upstar.yaml` | UPSTAR 训练 | `metric_for_best: 'NDCG@20'` |
| `item_repr.yaml` | Item 表示 | `embed_dim: 128` |
| `stb.yaml` | STB 计算 | `num_time_nodes: 50` |
| `tafeng_eval.yaml` | 评估 | `k_values: [1,5,10,15,20,50]` |

### 运行脚本

| 脚本 | 功能 | 调用入口 |
|------|------|----------|
| `run_preprocess.sh` | 数据预处理 | `Preprocessor`, `SequenceBuilder` |
| `run_tafeng_baseline.sh` | Baseline 训练 | `cross_validation_baseline.run_cross_validation()` |
| `run_tafeng_upstar.sh` | UPSTAR 训练 | `train_upstar_cv.run_cross_validation()` |
| `run_item_repr.sh` | Item 表示 | `train_item_repr` |
| `run_stb.sh` | STB 计算 | `train_stb` |
| `compare_results.sh` | 结果比较 | `ModelComparator` |
| `smoke_test.sh` | Smoke test | 自动化主链路验证 |

---

## 运行入口

### Smoke Test（推荐先做）
```bash
bash scripts/smoke_test.sh
```

### 完整实验
```bash
python run_full_experiment.py
```

### 分步运行
```bash
bash scripts/run_preprocess.sh
bash scripts/run_tafeng_baseline.sh
bash scripts/run_item_repr.sh
bash scripts/run_stb.sh
bash scripts/run_tafeng_upstar.sh
bash scripts/compare_results.sh
```

---

## 关键特性

### ✅ CV10-ONLY 统一
- 所有训练统一使用 10-fold cross-validation
- 删除了 single/cv 双模式
- 删除了 leave_one_out 模式

### ✅ 统一入口
- 所有脚本调用新的 Python 训练入口
- 无旧实验框架依赖

### ✅ 最小配置
- 收敛为 5 个核心配置文件
- 统一使用 `metric_for_best: 'NDCG@20'`

---

**项目状态**: ✅ Ready for GPU experiments

**代码质量**: ⭐⭐⭐⭐⭐ (干净、一致、论文对齐)
