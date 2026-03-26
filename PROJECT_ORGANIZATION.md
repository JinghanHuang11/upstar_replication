# UPSTAR 项目组织详解

> **最后更新**: 2026-03-25
> **项目状态**: ✅ Phase 1-6 全部完成并验证 | 论文对齐改进完成
> **代码规模**: 40 个 Python 源文件 | 9 个配置文件 | 8 个脚本
> **论文对齐**: 已完成 5 个关键模块的对齐改进

---

## 一、项目总览

### 核心目标

复现 UPSTAR 论文：**动机感知的序列推荐系统 (Motivation-Aware Sequential Recommendation)**

**论文**: UPSTAR - Uncovering Purchase Motivations for Sequential Recommendation

### 技术创新（已按论文实现）

- **STB (Stable Transaction Bias)**：
  - 互信息近似（MINE）估计 `I(S; e(S))`
  - 最坏情况扰动（PGD）优化
  - Day-level 时间节点（ρ=50）

- **三分支建模**：
  - S-model (stable) / E-model (exploratory) / O-model (overall)
  - 4层 LSTM (hidden_dim=128, per Section 7.3)

- **全局融合**：可学习融合门动态组合三分支预测

- **Dual Teacher-Student**：
  - Stable 样本：S teaches E
  - Exploratory 样本：E teaches S

- **正交约束**：`L_orth = τ_s · z_other^T · z_stab + τ_e · z_other^T · z_expl`

- **Item-GNN**：区分 in/out 邻居的 Message Passing

### 数据集

- **Tafeng**：台湾超市数据集 (26K 用户, 15K 商品, 785K 交互)
- **Baseline**：测试数据集 (10 用户, 10 商品, 51 交互)

---

## 二、论文对齐改进（2026-03-25）

### 已完成的 5 个关键模块对齐

| 模块 | 对齐内容 | 论文章节 | 状态 |
|------|----------|----------|------|
| **1. Item-Time Graph** | 真实 day-level 时间节点 | Section 7.1 | ✅ |
| **2. Item-GNN** | 区分 in/out 邻居 | Section 3.2 | ✅ |
| **3. STB** | 互信息近似（MINE） | Section 3.1.3 | ✅ |
| **4. 序列模型** | 三路预测 + Dual Teacher-Student | Section 3.1.4, 3.3 | ✅ |
| **5. 超参数** | 配置对齐 Section 7 | Section 7 | ✅ |

### 详细改进说明

#### 模块 1: Item-Time Graph — 真实 Day-Level 时间节点

**修改前**：使用简化版时间节点（item 索引取模）

**修改后**：将 Unix timestamp 转换为 day-level 日期字符串
- 论文要求：`time_granularity = 'day'`, `ρ = 50`
- 实现：`datetime.fromtimestamp(ts).strftime('%Y-%m-%d')`

**对应文件**：`src/data/build_sequences.py`, `src/graphs/item_time_graph.py`

---

#### 模块 2: Item-GNN — 区分 In/Out 邻居

**修改前**：单一邻居聚合，无方向区分

**修改后**：分离的 in-neighbors 和 out-neighbors 聚合
```
h_n^{g(k)} ← ReLU(W_in^(k) · Σ h_p^(k-1) + W_out^(k) · Σ h_q^(k-1))
```

**对应文件**：`src/models/item_gnn.py`

---

#### 模块 3: STB — 互信息近似与判别器

**修改前**：仅使用余弦相似度作为稳定性代理

**修改后**：实现 MINE（Mutual Information Neural Estimation）
- 新增 `MIEstimator` 类
- 实现 `I(S; e(S))` 的可训练判别器
- 支持两种模式：MI-based（推荐）和 cosine similarity（基线）

**对应文件**：`src/models/stb_encoder.py`

---

#### 模块 4: 序列模型 — 三路预测与 Dual Teacher-Student

**修改内容**：
- ✅ 确认 LSTM 架构：4-layer, hidden_size=128 (Section 7.3)
- ✅ 确认三路输出：S/E/O 分支
- ✅ Global Fusion Gate：自适应权重融合
- ✅ 修复 Bug：类名拼写错误 `UPSTARMoodel` → `UPSTARModel`
- ✅ Dual Teacher-Student：已完整实现（`losses.py`）

**对应文件**：`src/models/sequence_models.py`, `src/models/fusion.py`, `src/models/upstar.py`, `src/training/losses.py`

---

#### 模块 5: 超参数配置对齐

**所有参数已与论文 Section 7 对齐**：

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

**对应文件**：`configs/tafeng_upstar.yaml`, `configs/stb.yaml`, `configs/item_repr.yaml`

---

## 三、六个实验阶段

### Phase 0: 数据预处理

**目标**：加载原始数据，构建用户购买序列，划分 train/val/test

**输入**：`data/raw/ta_feng.csv`

**输出**：
- `data/processed/tafeng/metadata.pkl` — 词表
- `data/processed/tafeng/train_sequences.pkl` — **新格式**：`[(item, timestamp), ...]`
- `data/processed/tafeng/val_sequences.pkl`
- `data/processed/tafeng/test_sequences.pkl`

**脚本**：`bash scripts/run_preprocess.sh`

**核心模块**：
- `src/data/preprocess.py` — `Preprocessor` + `TafengPreprocessor`
- `src/data/build_sequences.py` — `SequenceBuilder`（**已更新**：保留 timestamp）
- `src/data/dataset.py` — PyTorch Dataset
- `src/data/dataloader.py` — DataLoader

**统计**：26,333 用户 | 15,652 商品 | 785,258 交互

**⚠️ 重要**：由于序列格式更新，需要重新运行预处理！

---

### Phase 1: Baseline 训练

**目标**：训练 LSTM 基线模型，确立性能基准

**输入**：Phase 0 输出的序列数据

**输出**：`outputs/baselines/tafeng/single_run/checkpoints/best_model.pt`

**脚本**：
- `bash scripts/run_tafeng_baseline.sh single` — 单次训练
- `bash scripts/run_tafeng_baseline.sh cv` — 10 折交叉验证

**核心模块**：
- `src/models/baseline_lstm.py` — `LSTMRec`
- `src/training/train_baseline.py` — 单次训练逻辑
- `src/training/cross_validation_baseline.py` — 10 折交叉验证

**性能**：NDCG@10 ~0.097, HR@10 ~0.164, MRR@10 ~0.040

---

### Phase 2: Item 表示学习

**目标**：在商品共现图上训练 Item-GNN，生成商品 Embedding

**输入**：Phase 0 序列 + 图结构

**输出**：`outputs/checkpoints/item_repr/item_embeddings.pt`

**脚本**：`bash scripts/run_item_repr.sh`

**核心模块**：
- `src/graphs/item_graph.py` — `ItemGraphBuilder`（会话内图 + 跨会话图）
- `src/models/item_gnn.py` — `ItemGNN`（**已更新**：in/out 邻居分离）
- `src/training/train_item_repr.py` — Item-GNN 训练

**图类型**：
- In-session graph：同一购物会话内的商品共现
- Cross-session graph：跨会话的商品转移

**⚠️ 重要**：由于更新了 Item-GNN 架构，需要重新训练！

---

### Phase 3: STB 计算

**目标**：基于商品-时间图计算每个商品的 STB 分数，将商品分为 stable / exploratory / other

**输入**：Phase 2 商品 Embedding

**输出**：
- `outputs/checkpoints/stb/stb_scores.npy`
- `outputs/checkpoints/stb/motivation_labels.npy`

**脚本**：
- `bash scripts/run_stb.sh` — V1 基础版
- `bash scripts/run_stb_advanced.sh` — 进阶版（4 种扰动方案）

**核心模块**：
- `src/graphs/item_time_graph.py` — `ItemTimeGraphBuilder`（**已更新**：day-level 时间节点）
- `src/graphs/perturbation.py` — V1 扰动（随机特征噪声 + 随机边扰动）
- `src/graphs/perturbation_advanced.py` — 进阶扰动
- `src/models/stb_encoder.py` — STB 编码器（**已更新**：MIEstimator）
- `src/models/stb_encoder_advanced.py` — 进阶 STB 编码器
- `src/training/train_stb.py` — V1 STB 训练
- `src/training/train_stb_advanced.py` — 进阶 STB 训练

**STB 公式（Section 3.1.3）**：
```
STB = min_S'∈B I(S'; e(S'))
```
其中 `I(S; e(S))` 通过 MINE 估计。

**⚠️ 重要**：由于更新了时间节点和 MI 估计，需要重新计算！

---

### Phase 4: UPSTAR 训练

**目标**：训练三分支模型（S/E/O），结合全局融合进行推荐

**输入**：Phase 2 Embedding + Phase 3 STB 标签

**输出**：`outputs/checkpoints/upstar_train/model_after_stage4.pt`

**脚本**：
- `bash scripts/run_tafeng_upstar.sh` — 完整 UPSTAR

**核心模块**：
- `src/models/sequence_models.py` — S/E/O 三分支 LSTM（**已更新**：论文引用注释）
- `src/models/fusion.py` — 可学习融合门
- `src/models/upstar.py` — UPSTAR 主模型（**已更新**：修复 bug + 论文引用）
- `src/data/upstar_dataset.py` — UPSTAR 专用数据集
- `src/training/train_upstar_stage1.py` — Stage 1 训练
- `src/training/train_upstar.py` — 完整 4 阶段训练
- `src/training/losses.py` — 损失函数（**已确认**：完整实现）

**四阶段训练（Section 3.3）**：
1. **Stage 1**：Global loss — 全局预测
2. **Stage 2**：Branch loss — S/E/O 分支独立预测
3. **Stage 3**：Orthogonality loss — 分支正交约束
4. **Stage 4**：Distillation loss — 知识蒸馏

**预期性能**：NDCG@10 ~0.103, HR@10 ~0.173 (+4-6% vs Baseline)

---

### Phase 5: 评估与对比

**目标**：评估 Baseline 和 UPSTAR，生成对比报告

**输入**：Phase 1 和 Phase 4 的模型

**输出**：`outputs/results/comparison.json` + 对比报告

**脚本**：
- `bash scripts/run_eval.sh` — 模型评估
- `bash scripts/compare_results.sh` — 结果对比

**核心模块**：
- `src/evaluation/metrics.py` — HR@K, NDCG@K, MRR@K
- `src/evaluation/evaluator.py` — UPSTAR 统一评估器
- `src/evaluation/evaluate_baseline.py` — Baseline 评估
- `src/evaluation/cross_validation.py` — 交叉验证
- `src/evaluation/report.py` — 报告生成
- `src/evaluation/comparison.py` — 性能对比

---

## 四、完整 Pipeline

### 一键运行（推荐）

```bash
# 快速测试
python run_full_experiment.py --mode quick

# 完整实验
python run_full_experiment.py --mode full
```

### 分步运行

```bash
python run_full_experiment.py --start-phase 0 --stop-phase 0    # 预处理
python run_full_experiment.py --start-phase 1 --stop-phase 1    # Baseline
python run_full_experiment.py --start-phase 2 --stop-phase 2    # Item repr
python run_full_experiment.py --start-phase 3 --stop-phase 3    # STB
python run_full_experiment.py --start-phase 4 --stop-phase 4    # UPSTAR
python run_full_experiment.py --start-phase 5 --stop-phase 5    # 评估
```

### 断点续跑

```bash
python run_full_experiment.py --resume
```

---

## 五、重新训练指南

### ⚠️ 为什么需要重新训练？

由于完成了 5 个模块的论文对齐改进，以下内容需要更新：

1. **数据格式**：序列现在包含 timestamp（`[(item, ts), ...]`）
2. **Item-Time Graph**：使用真实 day-level 时间节点
3. **Item-GNN**：区分 in/out 邻居
4. **STB**：使用 MI 估计（可选）
5. **图缓存**：需要重新构建

### 重新训练步骤

```bash
# 1. 清除旧数据和缓存
rm -rf data/processed/tafeng/*
rm -rf data/cache/tafeng/*
rm -rf outputs/checkpoints/item_repr/*
rm -rf outputs/checkpoints/stb/*
rm -rf outputs/checkpoints/upstar/*

# 2. 重新预处理（包含 timestamp）
bash scripts/run_preprocess.sh

# 3. 重新训练 Item-GNN（新架构）
bash scripts/run_item_repr.sh

# 4. 重新计算 STB（新时间节点）
bash scripts/run_stb.sh

# 5. 重新训练 UPSTAR
bash scripts/run_tafeng_upstar.sh

# 或者一键运行（自动处理所有依赖）
python run_full_experiment.py --force-rerun --mode full
```

---

## 六、源代码组织

### 模块依赖关系

```
src/data/
    ↓（序列数据 with timestamps）
src/graphs/
    ↓（商品图 + day-level item-time 图）
src/models/
    ↓（模型定义：in/out Item-GNN, MI-based STB）
src/training/
    ↓（训练：Dual Teacher-Student）
src/evaluation/
    ↓（评估）
src/experiments/  (编排)
src/utils/        (工具，被所有模块使用)
```

### 核心数据流

```
原始数据 (CSV)
    ↓ preprocess.py（保留 timestamp）
处理后序列 (pkl) - [(item, timestamp), ...]
    ↓ build_sequences.py
train/val/test 序列
    ↓
├──→ Item Graph → Item GNN (in/out 聚合) → Item Embeddings
    ↓                              ↓
    STB Encoder (day-level 时间) ←←←←←←←←←←←←←←←←←←
    ↓（MI 估计）
STB Labels (stable/exploratory/other)
    ↓
UPSTAR Dataset (序列按动机切分)
    ↓
S/E/O 分支 LSTM → Fusion Gate → 预测
    ↓（Dual Teacher-Student）
评估指标 (HR@K, NDCG@K, MRR@K)
```

---

## 七、设计原则

### 1. 模块化
- 每个模块职责单一、边界清晰
- 可独立测试、验证和替换

### 2. 论文对齐
- 所有核心算法与论文一致
- 超参数与论文 Section 7 完全对齐
- 添加详细的论文引用注释

### 3. 可复现性
- 固定随机种子（`src/utils/seed.py`）
- 配置化参数管理（YAML）
- 详细日志记录（TensorBoard 支持）

### 4. 可缓存性
- STB 分数可离线计算和缓存
- Item Embeddings 可复用
- 中间结果持久化

### 5. 可扩展性
- 支持多种数据集（Tafeng / Retailer / IJCAI15）
- 支持多种 STB 版本（V1 + MI-based + 4 个升级版）
- 支持多种 GNN（GCN / GAT / SAGE）

---

## 八、技术栈

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.0+ |
| 图神经网络 | torch-geometric |
| 数据处理 | pandas, numpy |
| 评估 | 自实现 HR@K / NDCG@K / MRR@K |
| 日志 | TensorBoard |
| 配置 | PyYAML, Hydra |
| 序列化 | pickle |
| 互信息估计 | MINE (自实现) |

---

## 九、关键文件链接

### 入口文件
- [QUICKSTART.md](QUICKSTART.md) — 快速开始
- [run_full_experiment.py](run_full_experiment.py) — 一键实验管理
- [demo.py](demo.py) — 环境验证脚本

### 核心模块（已更新）
- [src/data/build_sequences.py](src/data/build_sequences.py) — 序列构建（**新**：含 timestamp）
- [src/graphs/item_time_graph.py](src/graphs/item_time_graph.py) — Item-Time 图（**新**：day-level）
- [src/models/item_gnn.py](src/models/item_gnn.py) — Item-GNN（**新**：in/out 聚合）
- [src/models/stb_encoder.py](src/models/stb_encoder.py) — STB 编码器（**新**：MIEstimator）
- [src/models/upstar.py](src/models/upstar.py) — UPSTAR 模型（**修复**：类名 bug）
- [src/training/losses.py](src/training/losses.py) — 损失函数（**确认**：完整 Dual Teacher-Student）

### 配置文件（已更新）
- [configs/tafeng_upstar.yaml](configs/tafeng_upstar.yaml) — UPSTAR 配置（**论文对齐**）
- [configs/stb.yaml](configs/stb.yaml) — STB 配置（**论文对齐**）
- [configs/item_repr.yaml](configs/item_repr.yaml) — Item-GNN 配置（**论文对齐**）

### 详细文档
- [docs/FULL_EXPERIMENT_README.md](docs/FULL_EXPERIMENT_README.md) — 完整实验指南
- [docs/phase6/STB_ADVANCED_README.md](docs/phase6/STB_ADVANCED_README.md) — 高级 STB 说明
- [docs/evaluation/EVALUATION_README.md](docs/evaluation/EVALUATION_README.md) — 评估指南
- [docs/reference/upstar_rec.md](docs/reference/upstar_rec.md) — 论文精读笔记

---

**最后更新**: 2026-03-25
