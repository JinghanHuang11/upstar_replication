# UPSTAR Quick Start Guide

> **最后更新**: 2026-03-25
> **Conda 环境**: `conda create -n upstar python=3.10 -y && conda activate upstar`
> **论文对齐**: 已完成 5 个关键模块的对齐改进

---

## 项目概述

复现 UPSTAR 论文：**动机感知的序列推荐系统**

**论文**: UPSTAR - Uncovering Purchase Motivations for Sequential Recommendation

核心创新：
- **STB (Stable Transaction Bias)**：通过互信息估计（MINE）识别商品稳定性
- **三分支建模**：S-model (stable) / E-model (exploratory) / O-model (overall)
- **全局融合**：可学习融合门动态组合三分支预测
- **Dual Teacher-Student**：动态知识蒸馏
- **论文对齐**：5 个关键模块已完全对齐论文 Section 3 & 7

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n upstar python=3.10 -y
conda activate upstar
```

### 2. 安装 PyTorch (CPU)

```bash
pip install torch torchvision torchaudio
```

### 3. 安装 PyTorch Geometric (CPU)

```bash
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

> ⚠️ URL 中的版本号需匹配你的 PyTorch 版本。可先安装 PyTorch 后运行：
> `python -c "import torch; print(torch.__version__)"` 查看版本。

### 4. 安装其余依赖

```bash
pip install -r requirements.txt
```

### 5. 验证环境

```bash
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

## 快速开始

### ⚠️ 重要提示

由于完成了 5 个模块的论文对齐改进，需要**重新运行整个 Pipeline**：

**改进内容**：
1. ✅ Item-Time Graph：真实 day-level 时间节点
2. ✅ Item-GNN：区分 in/out 邻居
3. ✅ STB：互信息估计（MINE）
4. ✅ 序列模型：论文对齐 + Bug 修复
5. ✅ 超参数：完全对齐 Section 7

**需要重新运行**：预处理 → Item-GNN → STB → UPSTAR

---

### 方式一：一键运行（推荐）

```bash
# 完整实验（自动清除旧数据并重新训练）
python run_full_experiment.py --force-rerun --mode full

# 快速测试（小数据集，少量 epochs）
python run_full_experiment.py --force-rerun --mode quick
```

---

### 方式二：分步运行（手动控制）

#### 步骤 0：清除旧数据

```bash
# 清除处理后的数据（旧格式不含 timestamp）
rm -rf data/processed/tafeng/*

# 清除图缓存（旧时间节点）
rm -rf data/cache/tafeng/*

# 清除模型检查点（旧架构）
rm -rf outputs/checkpoints/item_repr/*
rm -rf outputs/checkpoints/stb/*
rm -rf outputs/checkpoints/upstar/*
```

#### 步骤 1：数据预处理（新格式：含 timestamp）

**选择预处理模式**：

| 模式 | 脚本 | 配置 | 说明 |
|------|------|------|------|
| Leave-One-Out | `bash scripts/run_preprocess.sh` | `configs/tafeng_baseline.yaml` | 快速验证，单次运行 |
| **10-Fold CV** | `bash scripts/run_preprocess_cv.sh` | `configs/tafeng_cv.yaml` | **论文对齐，推荐用于最终实验** |

```bash
# Leave-One-Out 模式（默认）
bash scripts/run_preprocess.sh

# 10-Fold CV 模式（论文推荐）
bash scripts/run_preprocess_cv.sh
```

**输出**：
- Leave-One-Out: `data/processed/tafeng/{train,val,test}_sequences.pkl`
- 10-Fold CV: `data/processed/tafeng/cv_splits/fold_{i}/{train,test}_sequences.pkl`

**新格式**：`[(item, timestamp), ...]`（保留原始 timestamp，支持 day-level item-time graph）

---

#### 步骤 2：训练 Baseline

```bash
# 单次训练（配合 leave-one-out 预处理）
bash scripts/run_tafeng_baseline.sh single

# 10 折交叉验证（配合 10-fold CV 预处理，论文对齐）
bash scripts/run_tafeng_baseline.sh cv
```

**输出**：`outputs/baselines/tafeng/*/results/results.json`

**注意**：训练模式必须与预处理模式匹配！

---

#### 步骤 3：Item 表示学习（新架构：in/out 聚合）

```bash
bash scripts/run_item_repr.sh
```

**输出**：`outputs/checkpoints/item_repr/item_embeddings.pt`

**新架构**：`h_n^{g(k)} = ReLU(W_in · Σ_in + W_out · Σ_out)`

---

#### 步骤 4：STB 计算（新时间节点：day-level）

```bash
# V1 基础版（推荐，使用 day-level 时间节点）
bash scripts/run_stb.sh

# 进阶版（可选，4 种 PGD 扰动方案）
bash scripts/run_stb_advanced.sh
```

**输出**：
- `outputs/checkpoints/stb/stb_scores.npy`
- `outputs/checkpoints/stb/motivation_labels.npy`

**新特性**：
- Day-level 时间节点（`datetime.fromtimestamp(ts).strftime('%Y-%m-%d')`）
- 可选 MI 估计（`MIEstimator`）

---

#### 步骤 5：UPSTAR 训练

```bash
bash scripts/run_tafeng_upstar.sh
```

**输出**：`outputs/checkpoints/upstar/tafeng/single_run/model_after_stage4.pt`

**4 阶段训练**：
1. Stage 1：Global loss
2. Stage 2：Branch loss
3. Stage 3：Orthogonality loss
4. Stage 4：Distillation loss

---

#### 步骤 6：评估与对比

```bash
# 评估模型
bash scripts/run_eval.sh

# Baseline vs UPSTAR 对比
bash scripts/compare_results.sh
```

**输出**：`outputs/results/comparison.json`

---

## 项目结构

```
upstar/
├── configs/              # 配置文件（已与论文对齐）
├── data/
│   ├── raw/            # 原始数据（ta_feng.csv）
│   ├── processed/      # 处理后数据（序列、词表）⚠️ 需重新生成
│   └── cache/          # 缓存（图、STB 分数）⚠️ 需重新生成
├── src/
│   ├── data/           # 数据处理（✓ 已更新：含 timestamp）
│   ├── graphs/         # 图构建（✓ 已更新：day-level 时间节点）
│   ├── models/         # 模型（✓ 已更新：in/out 聚合, MI 估计）
│   ├── training/       # 训练脚本（✓ 已确认：Dual Teacher-Student）
│   ├── evaluation/     # 评估（指标、评估器、报告）
│   ├── experiments/    # 实验管理（Phase 编排、报告生成）
│   └── utils/          # 工具（种子、日志、IO、注册器）
├── scripts/              # 运行脚本（每 Phase 一个）
├── outputs/              # 输出（checkpoints、logs、predictions）
├── docs/                 # 各 Phase 详细文档（含论文精读笔记）
├── demo.py               # 环境验证脚本
├── run_full_experiment.py  # 一键实验管理器
└── requirements.txt
```

---

## 实验阶段（Phase 0-5）

| Phase | 名称 | 脚本 | 预计耗时 (CPU) | 状态 |
|-------|------|------|---------------|------|
| 0 | 数据预处理 | `run_preprocess.sh` 或 `run_preprocess_cv.sh` | 2-5 分钟 | ⚠️ 需重新运行 |
| 1 | Baseline 训练 | `run_tafeng_baseline.sh [single\|cv]` | ~40 分钟 (single) / ~6 小时 (cv) | - |
| 2 | Item 表示学习 | `run_item_repr.sh` | 10-20 分钟 | ⚠️ 需重新运行 |
| 3 | STB 计算 | `run_stb.sh` | 20-40 分钟 | ⚠️ 需重新运行 |
| 4 | UPSTAR 训练 | `run_tafeng_upstar.sh` | 6-12 小时 | - |
| 5 | 评估对比 | `run_eval.sh` | 5-10 分钟 | - |

**数据划分模式说明**：
- **Leave-One-Out**: 单次快速验证，适合开发调试
- **10-Fold CV**: 论文对齐的实验设置，用于最终结果报告

---

## 核心模块说明

### src/data/ — 数据处理

| 文件 | 功能 | 更新状态 |
|------|------|----------|
| `preprocess.py` | 加载原始数据、构建词表、过滤低频用户/商品 | - |
| `build_sequences.py` | 构建用户购买序列，按时间切分 train/val/test | **✓ 已更新**：保留 timestamp |
| `dataset.py` | PyTorch Dataset（SequentialDataset + NegativeSamplingDataset） | - |
| `dataloader.py` | DataLoader 工厂函数 | - |
| `upstar_dataset.py` | UPSTAR 专用数据集，将序列切分为 Stable/Exploratory/Entire 子序列 | - |

---

### src/graphs/ — 图构建

| 文件 | 功能 | 更新状态 |
|------|------|----------|
| `item_graph.py` | 会话内 + 跨会话商品共现图 | **✓ 已更新**：兼容新序列格式 |
| `item_time_graph.py` | 商品-时间二部图 + STB 计算器 | **✓ 已更新**：day-level 时间节点 |
| `perturbation.py` | V1 扰动（随机特征噪声 + 随机边扰动） | - |
| `perturbation_advanced.py` | 进阶扰动（PGD、连续拓扑、互信息） | - |

---

### src/models/ — 模型定义

| 文件 | 功能 | 更新状态 |
|------|------|----------|
| `baseline_lstm.py` | LSTM 基线模型 | - |
| `item_gnn.py` | Item-GNN（**新**：in/out 邻居分离 + remember gate） | **✓ 已更新** |
| `stb_encoder.py` | STB 编码器（**新**：MIEstimator + cosine baseline） | **✓ 已更新** |
| `stb_encoder_advanced.py` | 进阶 STB 编码器 | - |
| `sequence_models.py` | S/E/O 三分支 LSTM（4-layer, hidden=128） | **✓ 已更新**：论文引用 |
| `fusion.py` | 可学习融合门 | - |
| `upstar.py` | UPSTAR 主模型（**修复**：类名 bug） | **✓ 已更新** |
| `load_item_embeddings.py` | 加载 Phase 2 商品 Embedding | - |

---

### src/training/ — 训练脚本

| 文件 | 功能 | 更新状态 |
|------|------|----------|
| `train_baseline.py` | LSTM Baseline 单次训练 | - |
| `cross_validation_baseline.py` | LSTM Baseline 10 折交叉验证 | - |
| `train_item_repr.py` | Item-GNN 训练 | - |
| `train_stb.py` | STB V1 计算 | - |
| `train_stb_advanced.py` | 进阶 STB 计算 | - |
| `train_upstar_stage1.py` | UPSTAR Stage 1 训练 | - |
| `train_upstar.py` | UPSTAR 完整 4 阶段训练 | - |
| `losses.py` | 损失函数（global + branch + ortho + distillation） | **✓ 已确认**：完整实现 |

---

## 配置说明

配置文件位于 `configs/` 目录（**已与论文 Section 7 对齐**）：

```yaml
# configs/tafeng_upstar.yaml 示例
dataset:
  name: tafeng
  data_dir: data/raw/
  processed_dir: data/processed/tafeng/

model:
  embed_dim: 64
  hidden_dim: 128          # Paper: hidden_size = 128
  num_layers: 4            # Paper: num_layers = 4

training:
  batch_size: 64           # Paper: batch_size = 64
  learning_rate: 0.0003    # Paper: lr = 3e-4

  # Distillation parameters (Section 3.3)
  tau_s: 0.5              # Paper: τ_s = 0.5
  tau_e: 0.5              # Paper: τ_e = 0.5
  lambda_distill: 0.7      # Paper: λ = 0.7
```

---

## 性能指标

### 预期性能（论文参考）

| 模型 | NDCG@10 | HR@10 | MRR@10 |
|------|---------|-------|--------|
| Baseline (LSTM) | ~0.097 | ~0.164 | ~0.040 |
| UPSTAR | ~0.103 | ~0.173 | ~0.043 |

**提升幅度**：+4-6% vs Baseline

---

## 常见问题

**Q: 为什么需要重新运行？**

A: 由于完成了论文对齐改进，数据格式和模型架构都有更新：
- 序列格式：从 `[item1, item2, ...]` 改为 `[(item, ts), ...]`（支持 day-level item-time graph）
- 数据划分：支持 leave-one-out 和 10-fold cross-validation（论文使用 CV）
- Item-Time Graph：从简化版改为真实 day-level 时间节点
- Item-GNN：从单一聚合改为 in/out 邻居分离
- STB：新增可选的 MI 估计
- 所有超参数已对齐论文 Section 7

**Q: 应该选择哪种数据划分模式？**

A:
- **开发调试阶段**: 使用 leave-one-out（`run_preprocess.sh`），速度快
- **最终实验报告**: 使用 10-fold CV（`run_preprocess_cv.sh`），符合论文设置
- **注意**: 训练模式必须与预处理模式一致！

**Q: CPU 环境训练太慢？**

A: 使用 `--mode quick` 快速测试：减少 epochs 和数据集规模

**Q: STB 计算需要 GPU 吗？**

A: 不需要，但有 GPU 会显著加速

**Q: 训练中断了怎么办？**

A: 使用 `python run_full_experiment.py --resume` 断点续跑

**Q: 如何验证论文对齐？**

A:
1. 查看各模块代码中的论文引用注释（如 `Section 3.2`）
2. 查看 `docs/reference/upstar_rec.md`（论文精读笔记）
3. 检查 `configs/*.yaml` 中的参数说明（已标注论文值）

**Q: demo.py 测试失败？**

A: demo.py 是过时的测试脚本，直接用上面的环境验证命令即可

---

## 论文对齐细节

### 5 个关键模块

| # | 模块 | 论文章节 | 更新内容 |
|---|------|----------|----------|
| 1 | Item-Time Graph | Section 7.1 | 真实 day-level 时间节点 |
| 2 | Item-GNN | Section 3.2 | 区分 in/out 邻居 |
| 3 | STB | Section 3.1.3 | 互信息估计（MINE） |
| 4 | 序列模型 | Section 3.1.4, 3.3 | 三路预测 + Dual Teacher-Student |
| 5 | 超参数 | Section 7 | 全部对齐 |

### 关键参数对照

| 参数 | 论文值 | 代码配置 | 状态 |
|------|--------|----------|------|
| ρ (时间节点) | 50 | `stb.yaml: num_time_nodes: 50` | ✅ |
| β (扰动轮数) | 40 | `stb.yaml: perturbation_rounds: 40` | ✅ |
| α (PGD 步长) | 0.4 | `stb.yaml: feature_step_size: 0.4` | ✅ |
| ε (扰动预算) | 0.1 | `stb.yaml: epsilon_x: 0.1` | ✅ |
| hidden_dim (LSTM) | 128 | `tafeng_upstar.yaml: hidden_dim: 128` | ✅ |
| num_layers (LSTM) | 4 | `tafeng_upstar.yaml: num_layers: 4` | ✅ |
| lr (UPSTAR) | 3e-4 | `tafeng_upstar.yaml: learning_rate: 0.0003` | ✅ |
| λ (distill) | 0.7 | `tafeng_upstar.yaml: lambda_distill: 0.7` | ✅ |
| τ_s | 0.5 | `tafeng_upstar.yaml: tau_s: 0.5` | ✅ |
| τ_e | 0.5 | `tafeng_upstar.yaml: tau_e: 0.5` | ✅ |

---

## 下一步

1. **确认环境可用**（运行验证命令）
2. **清除旧数据**（`rm -rf data/processed/tafeng/* data/cache/tafeng/*`）
3. **一键运行**（`python run_full_experiment.py --force-rerun --mode full`）
4. **查看详细文档**：
   - [README.md](README.md) - 项目概述
   - [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) - 项目组织详解
   - [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 文件结构详解
   - [docs/FULL_EXPERIMENT_README.md](docs/FULL_EXPERIMENT_README.md) - 完整实验指南
   - [docs/reference/upstar_rec.md](docs/reference/upstar_rec.md) - 论文精读笔记

---

**最后更新**: 2026-03-25
