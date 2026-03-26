# UPSTAR: Uncovering Purchase Motivations for Sequential Recommendation

Reproduction of UPSTAR: Motivation-aware Sequential Recommendation

**论文**: UPSTAR - Uncovering Purchase Motivations for Sequential Recommendation

**项目状态**: ✅ Phase 1-6 全部完成 | 论文对齐改进完成（2026-03-25）

---

## 论文核心贡献

### 1. STB (Stable Transaction Bias)

**Section 3.1**: 通过分析商品-时间图，识别购买动机

```
STB = min_S'∈B I(S'; e(S'))
```

- **实现**: 使用 MINE（Mutual Information Neural Estimation）估计互信息
- **时间粒度**: Day-level 时间节点（ρ=50）
- **扰动**: PGD 优化找最坏情况扰动

### 2. 三分支建模

**Section 3.1.4**: 根据动机分解序列

- **S-model**: Stable preference 子序列
- **E-model**: Exploratory intent 子序列
- **O-model**: Entire sequence（包括 uncategorized）

### 3. Global Fusion

**Section 3.1.4**: 可学习融合门

```
z_global = gate_stab · z_stab + gate_expl · z_expl + gate_other · z_other
```

### 4. Dual Teacher-Student

**Section 3.3**: 动态知识蒸馏

- Stable 样本：S teaches E
- Exploratory 样本：E teaches S

### 5. 正交约束

**Section 3.3**: 确保 O-model 学习独特特征

```
L_orth = τ_s · z_other^T · z_stab + τ_e · z_other^T · z_expl
```

---

## 项目结构

```
upstar/
├── configs/           # 配置文件（已与论文 Section 7 对��）
├── data/              # 数据存储（raw, processed, cache）
├── src/               # 源代码（40 个 Python 文件）
│   ├── data/          # 数据处理（✓ 已更新：含 timestamp）
│   ├── graphs/        # 图构建（✓ 已更新：day-level 时间节点）
│   ├── models/        # 模型定义（✓ 已更新：in/out 聚合, MI 估计）
│   ├── training/      # 训练脚本（✓ 已确认：Dual Teacher-Student）
│   ├── evaluation/    # 评估（HR@K, NDCG@K, MRR@K）
│   ├── experiments/   # 实验管理
│   └── utils/         # 工具
├── scripts/           # 运行脚本
├── outputs/           # 输出（checkpoints, logs, predictions）
├── docs/              # 详细文档（含论文精读笔记）
├── QUICKSTART.md      # 快速开始
├── PROJECT_STRUCTURE.md  # 文件结构
├── PROJECT_ORGANIZATION.md  # 项目组织详解
└── requirements.txt
```

---

## 核心改进（论文对齐）

### 5 个关键模块已对齐

| 模块 | 对齐内容 | 论文章节 |
|------|----------|----------|
| **Item-Time Graph** | 真实 day-level 时间节点 | Section 7.1 |
| **Item-GNN** | 区分 in/out 邻居的 Message Passing | Section 3.2 |
| **STB** | 互信息近似（MINE） | Section 3.1.3 |
| **序列模型** | 三路预测 + Dual Teacher-Student | Section 3.1.4, 3.3 |
| **超参数** | 配置对齐 Section 7 | Section 7 |

### 关键参数（已对齐论文 Section 7）

| 参数类别 | 参数 | 论文值 | 配置值 |
|----------|------|--------|--------|
| **STB** | hidden_dim | 512 | 512 ✅ |
| | num_layers | 1 | 1 ✅ |
| | lr | 1e-3 | 0.001 ✅ |
| | ρ | 50 | 50 ✅ |
| | β | 40 | 40 ✅ |
| | α | 0.4 | 0.4 ✅ |
| | ε, ε_x, ε_a | 0.1 | 0.1 ✅ |
| **Item-GNN** | embed_dim | 128 | 128 ✅ |
| | num_layers | 1 | 1 ✅ |
| **序列模型** | hidden_dim | 128 | 128 ✅ |
| | num_layers | 4 | 4 ✅ |
| **联合训练** | lr | 3e-4 | 0.0003 ✅ |
| | batch_size | 64 | 64 ✅ |
| | λ (distill) | 0.7 | 0.7 ✅ |
| | τ_s | 0.5 | 0.5 ✅ |
| | τ_e | 0.5 | 0.5 ✅ |

---

## Pipeline

### Phase 1: Data Preparation

1. **Preprocess** - 加载原始数据，构建用户序列（**保留 timestamp**）
2. **Item Graph** - 构建会话内 + 跨会话商品共现图
3. **Item-Time Graph** - 构建 day-level item-time 图
4. **STB Calculation** - 计算 STB 分数（**使用 MI 估计**）

### Phase 2: Model Training

1. **Item Representation** - 训练 Item-GNN（**in/out 聚合**）
2. **Sequence Models** - 训练 S/E/O 三分支 LSTM（4-layer, hidden=128）
3. **Fusion** - Global fusion 门
4. **Joint Training** - 4 阶段训练（global + branch + ortho + distill）

### Phase 3: Evaluation

1. **Metrics** - HR@K, NDCG@K, MRR
2. **10-Fold CV** - 与论文对齐的实验协议
3. **Ablation** - 消融研究

---

## Quick Start

### 1. 安装依赖

```bash
# 创建 Conda 环境
conda create -n upstar python=3.10 -y
conda activate upstar

# 安装 PyTorch (CPU)
pip install torch torchvision torchaudio

# 安装 PyTorch Geometric (CPU)
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 安装其余依赖
pip install -r requirements.txt
```

### 2. 验证环境

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

### 3. 完整实验

```bash
# 一键运行（推荐）
python run_full_experiment.py --force-rerun --mode full
```

### 4. 分步运行

```bash
# ⚠️ 需要重新运行以下步骤（由于论文对齐改进）：

# 清除旧数据
rm -rf data/processed/tafeng/*
rm -rf data/cache/tafeng/*
rm -rf outputs/checkpoints/item_repr/*
rm -rf outputs/checkpoints/stb/*

# 预处理（新格式：含 timestamp）
bash scripts/run_preprocess.sh

# Item 表示学习（新架构：in/out 聚合）
bash scripts/run_item_repr.sh

# STB 计算（新时间节点：day-level）
bash scripts/run_stb.sh

# UPSTAR 训练
bash scripts/run_tafeng_upstar.sh

# 评估
bash scripts/run_eval.sh
```

---

## 性能指标

### 预期性能（论文参考）

| 模型 | P@5 | P@20 | NDCG@5 | NDCG@20 | MRR@5 | MRR@20 |
|------|-----|------|--------|---------|-------|--------|
| LSTM Baseline | 8.23 | 16.48 | 5.75 | 8.13 | 4.92 | 5.77 |
| UPSTAR (Full) | 16.24 | 25.98 | 12.31 | 15.20 | 11.00 | 12.07 |

---

## 数据集

### Tafeng Dataset

- **26,333** 用户
- **15,652** 商品
- **785,258** 交互
- **29,142** sessions（按用户级长序列构造）

**论文处理方式**：将同一用户的所有购买拼接成一个 session，删除长度 < 3 的 session

---

## 论文实现要点

### 1. Motivation Identification (Section 3.1)

**核心思想**：STB 衡量商品对 timing 和 copurchase 上下文的鲁棒性

```
高 STB → Stable Preference（习惯性购买）
低 STB → Exploratory Intent（探索性购买）
```

### 2. Item Representation (Section 3.2)

**核心思想**：同时利用 in-session 和 cross-session 的商品关系

```
h_n^{g(k)} = ReLU(W_in · Σ_in + W_out · Σ_out)
```

### 3. Next Item Prediction (Section 3.1.4)

**核心思想**：三条路径 + 全局融合

- S-model: 稳定偏好子序列
- E-model: 探索意图子序列
- O-model: 完整序列

### 4. Dual Teacher-Student (Section 3.3)

**核心思想**：动态知识蒸馏

- Stable 样本：S teaches E（`L_{S→E} = D_KL(ŷ_S || ŷ_E)`）
- Exploratory 样本：E teaches S（`L_{E→S} = D_KL(ŷ_E || ŷ_S)`）

---

## 开发原则

- **Modular**: 每个模块独立可测
- **Cachable**: STB 分数可离线计算和缓存
- **Reproducible**: 固定随机种子
- **Configurable**: 所有参数通过 YAML 配置
- **Paper-Aligned**: 核心算法与论文一致

---

## 详细文档

- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 文件结构详解
- [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) - 项目组织详解
- [docs/FULL_EXPERIMENT_README.md](docs/FULL_EXPERIMENT_README.md) - 完整实验指南
- [docs/reference/upstar_rec.md](docs/reference/upstar_rec.md) - 论文精读笔记

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.0+ |
| 图神经网络 | torch-geometric |
| 互信息估计 | MINE (自实现) |
| 数据处理 | pandas, numpy |
| 评估 | HR@K, NDCG@K, MRR@K |
| 日志 | TensorBoard |
| 配置 | PyYAML |

---

## 常见问题

**Q: 为什么需要重新训练？**

A: 由于完成了 5 个模块的论文对齐改进，数据格式和模型架构都有更新：
- 序列现在包含 timestamp（`[(item, ts), ...]`）
- Item-Time Graph 使用真实 day-level 时间节点
- Item-GNN 区分 in/out 邻居
- STB 可选使用 MI 估计
- 需要清除旧缓存并重新生成

**Q: CPU 环境训练太慢？**

A: 使用 `--mode quick` 快速测试：减少 epochs 和数据集规模

**Q: 如何验证论文对齐？**

A: 查看 `docs/reference/upstar_rec.md` 和各模块代码中的论文引用注释

---

**最后更新**: 2026-03-25

**论文**: UPSTAR - Uncovering Purchase Motivations for Sequential Recommendation
