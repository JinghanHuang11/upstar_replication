# 项目文件组织结构

> **最后更新**: 2026-03-25
> **代码规模**: 40 个 Python 源文件 | 9 个配置文件 | 8 个脚本
> **论文对齐**: 已完成 5 个关键模块的对齐改进

---

## 目录结构

```
upstar/
├── configs/              # 配置文件 (9个) - 已与论文对齐
│   ├── baseline.yaml              # Baseline 测试配置
│   ├── tafeng_baseline.yaml       # Tafeng Baseline 配置
│   ├── tafeng_upstar.yaml         # Tafeng UPSTAR 配置（✓ 论文对齐）
│   ├── tafeng_eval.yaml           # Tafeng 评估配置
│   ├── test_baseline.yaml         # Baseline 测试
│   ├── item_repr.yaml             # Item 表示学习配置（✓ 论文对齐）
│   ├── stb.yaml                   # STB V1 配置（✓ 论文对齐）
│   ├── stb_advanced.yaml          # 高级 STB 配置
│   └── upstar_train.yaml           # UPSTAR 训练配置
│
├── data/                 # 数据目录
│   ├── raw/
│   │   ├── ta_feng.csv            # 原始 Tafeng 数据 (817K 条)
│   │   └── baseline/              # Baseline 测试数据
│   ├── processed/
│   │   ├── baseline/              # Baseline 处理结果
│   │   └── tafeng/                # Tafeng 处理结果（✓ 新格式：含 timestamp）
│   │       ├── metadata.pkl       # 词表 (26K 用户, 15K 商品)
│   │       ├── train_sequences.pkl     # [(item, timestamp), ...]
│   │       ├── val_sequences.pkl       # [(item, timestamp), ...]
│   │       └── test_sequences.pkl      # [(item, timestamp), ...]
│   └── cache/
│       └── tafeng/                # 图缓存（⚠️ 需重新生成）
│
├── docs/                 # 详细文档 (22个)
│   ├── phase1/           # Phase 1 文档 (6个)
│   ├── phase2/           # Phase 2 文档 (2个)
│   ├── phase3/           # Phase 3 文档 (2个)
│   ├── phase4/           # Phase 4 文档 (1个)
│   ├── phase5/           # Phase 5 文档 (2个)
│   ├── phase6/           # Phase 6 文档 (2个)
│   ├── evaluation/       # 评估文档 (1个)
│   ├── experiments/       # 实验文档 (1个)
│   ├── reference/        # 参考文档 (1个)
│   │   └── upstar_rec.md # 论文精读笔记
│   └── FULL_EXPERIMENT_README.md
│
├── scripts/              # 运行脚本 (8个)
│   ├── run_preprocess.sh          # 数据预处理（⚠️ 需重新运行）
│   ├── run_tafeng_baseline.sh     # Baseline 训练
│   ├── run_item_repr.sh           # Item 表示学习（⚠️ 需重新运行）
│   ├── run_stb.sh                 # STB 计算（⚠️ 需重新运行）
│   ├── run_stb_advanced.sh        # 高级 STB
│   ├── run_tafeng_upstar.sh       # UPSTAR 训练
│   ├── run_eval.sh                # 模型评估
│   └── compare_results.sh         # 结果对比
│
├── src/                  # 源代码 (40个 Python 文件) - 5个模块已更新
│   ├── data/             # 数据处理 (5个) - 1个已更新
│   ├── graphs/           # 图构建 (4个) - 2个已更新
│   ├── models/           # 模型 (8个) - 3个已更新
│   ├── training/         # 训练 (9个) - 1个已更新
│   ├── evaluation/       # 评估 (7个)
│   ├── experiments/      # 实验管理 (4个)
│   └── utils/            # 工具 (4个)
│
├── outputs/              # 输出目录（⚠️ 需清除旧数据）
│   ├── baselines/               # Baseline 实验
│   ├── upstar/                  # UPSTAR 实验
│   ├── checkpoints/             # 模型检查点（⚠️ 需清除）
│   ├── logs/                   # 日志
│   ├── predictions/             # 预测结果
│   └── results/                # 评估结果
│
├── tests/                # 测试脚本
│   ├── phase1/test_phase1.py
│   ├── test_phase5.py
│   ├── test_phase6.py
│   └── test_evaluation.py
│
├── QUICKSTART.md         # 快速开始（✓ 已更新）
├── PROJECT_STRUCTURE.md  # 本文件（✓ 已更新）
├── PROJECT_ORGANIZATION.md  # 项目组织详解（✓ 已更新）
├── README.md             # 项目说明（✓ 已更新）
├── FULL_EXPERIMENT_README.md # 完整实验指南
├── demo.py               # 环境验证脚本
├── run_full_experiment.py # 一键实验管理
└── requirements.txt      # 依赖列表
```

---

## 数据集状态

### Tafeng 数据集 ✅

| 统计项 | 数值 |
|--------|------|
| 原始记录 | 817,741 条 |
| 用户数 | 26,333 |
| 商品数 | 15,652 |
| 交互数 | 785,258 |
| 训练序列 | 26,327 |
| 验证序列 | 26,327 |
| 测试序列 | 26,327 |

**⚠️ 重要更新**：序列格式已改为 `[(item, timestamp), ...]`，需要重新预处理

---

## 源代码清单 (src/)

### src/data/ — 数据处理（1个已更新）

| 文件 | 功能 | 状态 |
|------|------|------|
| `preprocess.py` | 数据预处理器基类 + Tafeng 专用处理器 | - |
| `build_sequences.py` | 用户序列构建 | **✓ 已更新**：保留 timestamp |
| `dataset.py` | PyTorch Dataset / DataLoader | - |
| `dataloader.py` | DataLoader 封装 | - |
| `upstar_dataset.py` | UPSTAR 专用数据集 | - |

---

### src/graphs/ — 图构建（2个已更新）

| 文件 | 功能 | 状态 |
|------|------|------|
| `item_graph.py` | 会话内 + 跨会话商品共现图 | **✓ 已更新**：兼容新序列格式 |
| `item_time_graph.py` | 商品-时间二部图 | **✓ 已更新**：day-level 时间节点 |
| `perturbation.py` | V1 扰动（随机特征噪声 + 随机边扰动） | - |
| `perturbation_advanced.py` | 进阶扰动（PGD / 连续拓扑 / 互信息） | - |

---

### src/models/ — 模型定义（3个已更新）

| 文件 | 功能 | 状态 |
|------|------|------|
| `baseline_lstm.py` | LSTM Baseline | - |
| `item_gnn.py` | Item-GNN（GCN/GAT/SAGE 支持） | **✓ 已更新**：in/out 邻居分离 |
| `stb_encoder.py` | STB 编码器（V1） | **✓ 已更新**：MIEstimator |
| `stb_encoder_advanced.py` | 进阶 STB 编码器 | - |
| `sequence_models.py` | S/E/O 三分支序列编码器 | **✓ 已更新**：论文引用注释 |
| `fusion.py` | 可学习融合门 | - |
| `upstar.py` | UPSTAR 主模型 | **✓ 已更新**：修复 bug + 论文引用 |
| `load_item_embeddings.py` | 商品 Embedding 加载工具 | - |

---

### src/training/ — 训练脚本（1个已更新）

| 文件 | 功能 | 状态 |
|------|------|------|
| `train_baseline.py` | LSTM Baseline 单次训练 | - |
| `cross_validation_baseline.py` | LSTM Baseline 10 折交叉验证 | - |
| `train_item_repr.py` | Item-GNN 训练 | - |
| `train_stb.py` | STB V1 计算 | - |
| `train_stb_advanced.py` | 进阶 STB 计算 | - |
| `train_upstar_stage1.py` | UPSTAR Stage 1 训练 | - |
| `train_upstar.py` | UPSTAR 完整 4 阶段训练 | - |
| `losses.py` | 损失函数（global + branch + ortho + distillation） | **✓ 已确认**：完整实现 |

---

### src/evaluation/ — 评估

| 文件 | 功能 |
|------|------|
| `metrics.py` | HR@K, NDCG@K, MRR@K |
| `evaluator.py` | UPSTAR 统一评估器 |
| `evaluate_baseline.py` | Baseline 评估（排除 padding 干扰） |
| `cross_validation.py` | UPSTAR 交叉验证 |
| `report.py` | 报告生成（JSON/CSV/TXT/LaTeX/Markdown） |
| `comparison.py` | Baseline vs UPSTAR 性能对比 |
| `evaluate.py` | 评估主程序 |

---

### src/experiments/ — 实验管理

| 文件 | 功能 |
|------|------|
| `phase_base.py` | 实验阶段基类 |
| `phases.py` | 6 个 Phase 实现（Phase 0-5） |
| `experiment_manager.py` | 完整流水线编排 |
| `report_generator.py` | Markdown/HTML 综合报告生成 |

---

### src/utils/ — 工具

| 文件 | 功能 |
|------|------|
| `seed.py` | 随机种子设置 |
| `io.py` | pickle/torch 文件读写 |
| `logger.py` | 日志系统（TensorBoard 支持） |
| `registry.py` | 模型/组件注册器 |

---

## 配置文件清单 (configs/) - 全部已与论文对齐

| 文件 | 用途 | 对齐状态 |
|------|------|----------|
| `baseline.yaml` | Baseline 测试配置 | - |
| `tafeng_baseline.yaml` | Tafeng Baseline 训练 | - |
| `tafeng_upstar.yaml` | Tafeng UPSTAR 训练 | **✓ 论文对齐** |
| `tafeng_eval.yaml` | Tafeng 评估 | - |
| `test_baseline.yaml` | Baseline 功能测试 | - |
| `item_repr.yaml` | Item 表示学习 | **✓ 论文对齐** |
| `stb.yaml` | STB V1 计算 | **✓ 论文对齐** |
| `stb_advanced.yaml` | 高级 STB（4 种扰动方案） | - |
| `upstar_train.yaml` | UPSTAR 完整训练 | - |

---

## 运行脚本清单 (scripts/)

| 脚本 | 功能 | 需要重新运行？ |
|------|------|----------------|
| `run_preprocess.sh` | 数据预处理 | **⚠️ 是**（新格式） |
| `run_tafeng_baseline.sh [single\|cv]` | Baseline 单次/CV 训练 | - |
| `run_item_repr.sh` | Item 表示学习 | **⚠️ 是**（新架构） |
| `run_stb.sh` | STB V1 计算 | **⚠️ 是**（新时间节点） |
| `run_stb_advanced.sh` | 进阶 STB 计算 | - |
| `run_tafeng_upstar.sh` | UPSTAR 训练 | - |
| `run_eval.sh` | 模型评估 | - |
| `compare_results.sh` | Baseline vs UPSTAR 对比 | - |

---

## 文档清单 (docs/)

| 目录 | 文件 |
|------|------|
| `phase1/` | PHASE1_README.md, PHASE1_SUMMARY.md, PHASE1_TEST_REPORT.md, TAFENG_DATA_REPORT.md, TAFENG_QUICK_TEST_REPORT.md, PHASE1_COMPLETION_SUMMARY.md |
| `phase2/` | ITEM_REPR_README.md, PHASE2_SUMMARY.md |
| `phase3/` | STB_README.md, PHASE3_V1_SUMMARY.md |
| `phase4/` | UPSTAR_STAGE1_README.md |
| `phase5/` | UPSTAR_TRAIN_README.md, SUMMARY.md |
| `phase6/` | STB_ADVANCED_README.md, SUMMARY.md |
| `evaluation/` | EVALUATION_README.md |
| `experiments/` | TAFENG_EXPERIMENTS.md |
| `reference/` | upstar_rec.md（论文精读笔记） |
| 根目录 | FULL_EXPERIMENT_README.md |

---

## 论文对齐改进总结

### 已更新的 5 个关键模块

| # | 模块 | 对应文件 | 更新内容 |
|---|------|----------|----------|
| 1 | Item-Time Graph | `src/data/build_sequences.py`<br>`src/graphs/item_time_graph.py` | 真实 day-level 时间节点 |
| 2 | Item-GNN | `src/models/item_gnn.py` | 区分 in/out 邻居 |
| 3 | STB | `src/models/stb_encoder.py` | 互信息估计（MINE） |
| 4 | 序列模型 | `src/models/sequence_models.py`<br>`src/models/fusion.py`<br>`src/models/upstar.py` | 论文引用注释 + Bug 修复 |
| 5 | 超参数 | `configs/*.yaml` | 全部对齐 Section 7 |

---

## 快速使用

### 环境验证

```bash
conda create -n upstar python=3.10 -y
conda activate upstar
pip install torch torchvision torchaudio
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install -r requirements.txt

python -c "
import torch, yaml, numpy as np
from src.data.preprocess import Preprocessor
from src.data.build_sequences import SequenceBuilder
from src.graphs.item_graph import ItemGraphBuilder
from src.models.item_gnn import ItemGNN
from src.training.losses import UPSTARLoss
from src.evaluation.metrics import compute_all_metrics
print('PyTorch version:', torch.__version__)
print('All core modules imported successfully!')
"
```

---

### 完整实验（推荐）

```bash
# 一键运行（自动处理所有依赖）
python run_full_experiment.py --force-rerun --mode full
```

---

### 分步实验

```bash
# ⚠️ 需要按顺序重新运行以下步骤：

# 1. 清除旧数据和缓存
rm -rf data/processed/tafeng/*
rm -rf data/cache/tafeng/*
rm -rf outputs/checkpoints/item_repr/*
rm -rf outputs/checkpoints/stb/*

# 2. 重新预处理（包含 timestamp）
bash scripts/run_preprocess.sh

# 3. 重新训练 Item-GNN（新架构：in/out 聚合）
bash scripts/run_item_repr.sh

# 4. 重新计算 STB（新时间节点：day-level）
bash scripts/run_stb.sh

# 5. 重新训练 UPSTAR
bash scripts/run_tafeng_upstar.sh

# 6. 评估
bash scripts/run_eval.sh
```

---

**最后更新**: 2026-03-25
