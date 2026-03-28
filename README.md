# UPSTAR: Uncovering Purchase Motivations for Sequential Recommendation

Reproduction of UPSTAR: Motivation-aware Sequential Recommendation

**论文**: UPSTAR - Uncovering Purchase Motivations for Sequential Recommendation

**项目状态**: ✅ Phase 1-6 全部完成 | **Phase 4/5 论文对齐改进完成（2026-03-26）**

---

## 论文核心贡献

### 1. STB (Stable Transaction Bias)

**Section 3.1.3**: 通过商品-时间图分析 + 最坏情况扰动识别购买动机

```
STB_{n,t} = inf_{S'∈B_t} [1 - Pr[h([e(S')]_n) ≠ Stable Preference]]
         ≤ (I(S'; e(S')) + log 2) / log|Y|
```

- **核心思想**: 高 STB → 稳定偏好；低 STB → 探索意图
- **实现**:
  - MINE 估计 `I(S; e(S))`（完整 MI 下界，非简化 proxy）
  - Worst-case aggregation（`min` over perturbation rounds，非 `mean`）
  - PGD 扰动（α=0.4, ε_x=0.1, ε_a=0.1, β=40）
- **图结构**: Attributed item-time bipartite graph（强制验证，无非法边）

### 2. 三分支建模

**Section 3.1.4**: 根据动机分解序列

- **S-model**: Stable preference 子序列（4-layer LSTM, hidden=128）
- **E-model**: Exploratory intent 子序列（4-layer LSTM, hidden=128）
- **O-model**: Entire sequence（包括 uncategorized）（4-layer LSTM, hidden=128）
- **三条路径各自独立**：各有独立的 Embedding / LSTM / Linear 层，无共享 backbone

### 3. Global Fusion — 逐维 Per-Dimension Gate

**Section 3.1.4**: 可学习逐维融合门

```
gate_logits_per_dim ∈ ℝ^{d×3}   # [batch, hidden, 3]
f_s = softmax(gate_logits_per_dim, dim=2)  # 每维三个权重归一
z_global = f_s[1] ⊙ z_stab + f_s[2] ⊙ z_expl + f_s[3] ⊙ z_other
y_hat_global = Linear(z_global)   # 不是加权 logit 平均！
```

### 4. 联合损失公式

**Section 3.3**:

```
L_total = L_global
        + λ · L_S&E&O
        + L_orth
        + L_distill

其中：
  L_global   = CE(y_hat_global, target)
  L_S&E&O   = CE(y_hat_stab) + CE(y_hat_expl) + CE(y_hat_other)
  L_orth    = τ_s · z_other^T z_stab + τ_e · z_other^T z_expl   （无 normalize，无 squared）
  L_distill = dual teacher-student KL（按 target motivation label 条件触发）
```

### 5. Dual Teacher-Student — 条件触发

**Section 3.3**: 动态知识蒸馏（按 next-item 动机标签）

| target item 动机 | teacher | student | uncategorized (label=2) |
|---|---|---|---|
| stable (label=1) | S-model | E-model | **不触发蒸馏** |
| exploratory (label=0) | E-model | S-model | **不触发蒸馏** |

- teacher `.detach()`，student 保留梯度
- `distill_lambda=0.7`，`temperature=3.0`

---

## 项目结构

```
upstar/
├── configs/           # 配置文件（已与论文 Section 7 对齐）
├── data/              # 数据存储（raw, processed, cache）
├── src/               # 源代码（40 个 Python 文件）
│   ├── data/          # 数据处理
│   ├── graphs/        # 图构建
│   ├── models/        # 模型定义（Phase 4：逐维 fusion gate + 独立三分支）
│   ├── training/      # 训练脚本（Phase 4：完整联合损失 + 条件双师生）
│   ├── evaluation/    # 评估（Phase 5：主指标 P@5/20, NDCG@5/20, MRR@5/20）
│   ├── experiments/    # 实验管理
│   └── utils/         # 工具
├── scripts/           # 运行脚本
├── outputs/           # 输出（checkpoints, logs, predictions）
├── docs/              # 详细文档（含论文精读笔记）
├── QUICKSTART.md
├── PROJECT_STRUCTURE.md
├── PROJECT_ORGANIZATION.md
└── requirements.txt
```

---

## Phase 4/5 论文对齐改进（2026-03-26）⭐

### Phase 4 — 模型与训练

| # | 模块 | 改进内容 | 论文章节 | 状态 |
|---|------|----------|----------|------|
| 1 | **FusionGate** | 标量 `[B,3]` → **逐维 `[B,128,3]`** gate；softmax 从 dim=1 → dim=2 | Section 3.1.4 | ✅ |
| 2 | **y_hat_global** | 移除加权 logit 平均；改为 **`Linear(z_global)`** | Section 3.1.4 | ✅ |
| 3 | **L_orth** | 去掉 `F.normalize`（cosine similarity）；改为论文原文 **raw dot product** `τ_s·z_oᵀz_s + τ_e·z_oᵀz_e`（无 squared） | Section 3.3 | ✅ |
| 4 | **L_distill** | 完整实现**条件双师生**：按 target motivation label 触发；teacher `.detach()`；student 保留梯度 | Section 3.3 | ✅ |
| 5 | **联合损失** | 统一 `total_loss = λg·Lg + λb·Lb + λo·Lo + λd·Ld` | Section 3.3 | ✅ |
| 6 | **超参数** | lr=3e-4, batch=64, hidden=128, layers=4, λ=0.7, τ_s=τ_e=0.5, optimizer=Adam | Section 7.3/7.4 | ✅ |

### Phase 5 — 评估

| # | 模块 | 改进内容 | 状态 |
|---|------|----------|------|
| 7 | **主指标** | 统一为 `P@5, P@20, NDCG@5, NDCG@20, MRR@5, MRR@20`（论文 Table 2/3 口径） | ✅ |
| 8 | **输出结构** | JSON/CSV 分段：`main_metrics` + `additional_metrics`；主表与附加分离 | ✅ |
| 9 | **命名规范** | 移除 `HR@K` 命名；单正样本用 `Precision@K`（论文口径） | ✅ |
| 10 | **配置驱动** | `k_values` 从 config 而非硬编码读取 | ✅ |

### 之前已对齐的 12 个模块

| # | 模块 | 对应文件 |
|---|------|----------|
| 11 | Item-Time Graph (day-level) | `src/graphs/item_time_graph.py` |
| 12 | Item-GNN (in/out 邻居) | `src/models/item_gnn.py` |
| 13 | STB MINE 估计 | `src/models/stb_encoder.py` |
| 14 | STB worst-case (min) aggregation | `src/models/stb_encoder.py` |
| 15 | MI 梯度真正可微分 | `src/graphs/perturbation_advanced.py` |
| 16 | Bipartite 约束 | `src/graphs/item_time_graph.py` |
| 17 | 分离 Graph Summary | `src/models/stb_encoder.py` |
| 18 | 拓扑扰动 bipartite edges | `src/graphs/perturbation_advanced.py` |
| 19 | Config 驱动分类阈值 | `configs/stb.yaml` |
| 20 | 三分支独立 LSTM | `src/models/sequence_models.py` |
| 21 | uncategorized 保留在 O-model | `src/data/upstar_dataset.py` |
| 22 | STB 标签条件蒸馏不误触发 | `src/training/losses.py` |

---

## 关键参数（已对齐论文 Section 7）

| 参数类别 | 参数 | 论文值 | 配置值 |
|----------|------|--------|--------|
| **STB (Section 3.1.3)** | hidden_dim | 512 | 512 ✅ |
| | num_layers | 1 | 1 ✅ |
| | lr | 1e-3 | 0.001 ✅ |
| | ρ (time nodes) | 50 | 50 ✅ |
| | β (perturbation rounds) | 40 | 40 ✅ |
| | α (PGD step size) | 0.4 | 0.4 ✅ |
| | ε, ε_x (feature budget) | 0.1 | 0.1 ✅ |
| | ε_a (topology budget) | 0.1 | 0.1 ✅ |
| **Item-GNN (Section 3.2)** | embed_dim | 128 | 128 ✅ |
| | num_layers | 1 | 1 ✅ |
| **序列模型 (Section 7.3)** | hidden_dim | 128 | 128 ✅ |
| | num_layers | 4 | 4 ✅ |
| **联合训练 (Section 7.4)** | lr | 3e-4 | 0.0003 ✅ |
| | batch_size | 64 | 64 ✅ |
| | λ (branch loss) | 0.7 | 0.7 ✅ |
| | λ (distill) | 0.7 | 0.7 ✅ |
| | τ_s | 0.5 | 0.5 ✅ |
| | τ_e | 0.5 | 0.5 ✅ |
| | temperature | - | 3.0 ✅ |
| | optimizer | Adam | Adam ✅ |

---

## 评估指标说明

**Phase 5 以论文 Table 2/3 口径为主输出**：

| 论文主表 | 说明 |
|----------|------|
| `Precision@5` | top-5 精确率（单正样本场景下与 HR@K 数值等价，但语义不同） |
| `Precision@20` | top-20 精确率 |
| `NDCG@5` | top-5 归一化折损累计增益 |
| `NDCG@20` | top-20 归一化折损累计增益 |
| `MRR@5` | top-5 平均倒数排名 |
| `MRR@20` | top-20 平均倒数排名 |

工程附加指标（`additional_metrics`）：`NDCG@1/10/15`、`MRR@1/10/15`、`Recall@K` 等，供调参参考。

**为什么用 `Precision@K` 而不是 `HR@K`**：
- 单正样本 next-item prediction 中两者数值碰巧相等
- 但语义不同：`Precision@K` 衡量"top-k 预测的精确程度"，`HR@K` 衡量"有多少用户命中"
- 论文主表用 `Precision@K` 是学术规范，评估目标为排序质量

---

## Pipeline

### Phase 0: Data Preprocessing

1. **Preprocess** - 加载原始数据，构建用户序列（**保留 timestamp**）
2. **Sequence Split** - 支持 leave_one_out 和 10-fold CV 两种模式
3. **Metadata** - 增强统计信息（avg/min/max/std sequence length）

**两种划分策略**：

| 模式 | 配置 | 划分方式 | 适用场景 |
|------|------|----------|----------|
| **Leave-One-Out** | `split_method: leave_one_out` | last→test, second_last→val, others→train | 快速验证 / 工程调试 |
| **10-Fold CV** | `split_method: cv10` | 用户级 10 折交叉验证 | 论文实验 / 最终结果 |

**输出格式**：
- 序列：`[(item_idx, timestamp), ...]` — 保留 timestamp 用于 Phase 2/3 图构建
- Leave-One-Out: `train_sequences.pkl`, `val_sequences.pkl`, `test_sequences.pkl`
- 10-Fold CV: `cv_splits/fold_{i}/train_sequences.pkl`, `test_sequences.pkl`

### Phase 1: Baseline LSTM Training

1. **Baseline Model** - 单路 LSTM（hidden=128, layers=4）
2. **Two Training Modes** - 支持 leave_one_out 和 cv10
3. **Metrics** - P@5/20, NDCG@5/20, MRR@5/20（论文主表口径）

**Baseline 设计**：
- 单路 LSTM（无 S/E/O 分支，无 fusion gate，无 STB）
- 论文对齐超参数：hidden_dim=128, num_layers=4
- 与 UPSTAR 相同的评估指标，确保公平对比

### Phase 2: Item Representation

1. **Item Graph** - 构建会话内 + 跨会话商品共现图
2. **Item-GNN** - 训练 in/out 邻居分离聚合（1-layer, embed=128）

### Phase 3: STB Calculation

1. **GNN Encoder** - 1-layer, hidden=512
2. **Worst-case Perturbation** - β=40 rounds, PGD
3. **MI Estimation** - 完整 MINE 下界（joint - marginal）
4. **Worst-case Aggregation** - `min` over rounds
5. **Classification** - top 50% stable, bottom 40% exploratory, middle 10% uncategorized

### Phase 4: UPSTAR Training

1. **Sequence Split** - 按 STB 标签将序列拆为 stable / exploratory / entire 三路
2. **S/E/O LSTM** - 各自独立 4-layer LSTM (hidden=128)
3. **FusionGate** - 逐维 softmax gate（`[B,128,3]`）
4. **Joint Loss** - 4 阶段课程：`L_global → +L_S&E&O → +L_orth → +L_distill`

### Phase 5: Evaluation

1. **Metrics** - P@5/20, NDCG@5/20, MRR@5/20（论文主表）
2. **10-Fold CV** - 与论文对齐的实验协议
3. **Comparison** - Baseline vs UPSTAR

---

## Quick Start

### 1. 安装依赖

```bash
conda create -n upstar python=3.10 -y
conda activate upstar
pip install torch torchvision torchaudio
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install -r requirements.txt
```

### 2. 验证环境

```bash
python -c "
from src.training.losses import UPSTARLoss, create_loss_from_config
from src.evaluation.metrics import compute_all_metrics
from src.models.fusion import FusionGate
from src.models.upstar import UPSTARModel
print('All Phase 4/5 modules imported successfully!')
"
```

### 3. 完整实验

```bash
# 一键运行
python run_full_experiment.py --force-rerun --mode full

# 分步运行
bash scripts/run_preprocess.sh
bash scripts/run_item_repr.sh
bash scripts/run_stb.sh
bash scripts/run_tafeng_upstar.sh
bash scripts/run_eval.sh
```

---

## 预期性能（论文参考）

| 模型 | P@5 | P@20 | NDCG@5 | NDCG@20 | MRR@5 | MRR@20 |
|------|-----|------|--------|---------|-------|--------|
| LSTM Baseline | 8.23 | 16.48 | 5.75 | 8.13 | 4.92 | 5.77 |
| UPSTAR (Full) | 16.24 | 25.98 | 12.31 | 15.20 | 11.00 | 12.07 |

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
| 评估 | **Precision@K, NDCG@K, MRR@K**（论文口径） |
| 日志 | TensorBoard |
| 配置 | PyYAML |

---

## 常见问题

**Q: 为什么需要重新训练？**

A: Phase 4/5 完成了以下关键对齐改进：
- FusionGate 从标量 `[B,3]` 改为逐维 `[B,128,3]` gate
- `L_orth` 去掉 cosine similarity，改为论文 raw dot product
- L_distill 完整实现条件双师生蒸馏
- 评估主指标统一为论文 Table 2/3 口径

**Q: 四阶段课程是论文规定的吗？**

A: 不是。四阶段课程是**工程实现**，目的是逐步开启损失项以稳定训练。论文直接用联合损失（Section 3.3）。文件内已明确标注"engineering curriculum, not paper modification"。

**Q: 如何验证 Phase 4/5 对齐？**

A: 各模块均有自检脚本：
```bash
python -m src.training.losses     # Stage 1-4 损失测试
python -m src.models.upstar        # 逐维 gate 测试
python -m src.evaluation.evaluator  # 主/附加指标分离测试
python -m src.evaluation.comparison # comparison table 更新测试
```

---

**最后更新**: 2026-03-28

**论文**: UPSTAR - Uncovering Purchase Motivations for Sequential Recommendation

---

## Phase 0/1 更新（2026-03-28）

### Phase 0 — 数据预处理改进

| # | 模块 | 改进内容 | 状态 |
|---|------|----------|------|
| 1 | **10-Fold CV** | 实现 10 折交叉验证划分（论文对齐） | ✅ |
| 2 | **Timestamp 保留** | 序列格式 `[(item, timestamp), ...]` | ✅ |
| 3 | **双模式支持** | leave_one_out（工程）+ cv10（论文） | ✅ |
| 4 | **Metadata 增强** | 添加统计信息（avg/min/max/std） | ✅ |
| 5 | **向后兼容** | 支持旧格式 `[item, ...]` 读取 | ✅ |

**新增文件**：
- `configs/tafeng_cv.yaml` — CV10 模式配置
- `scripts/run_preprocess_cv.sh` — CV 预处理脚本
- `docs/phase0_*.md` — Phase 0 详细文档

### Phase 1 — Baseline 训练改进

| # | 模块 | 改进内容 | 状态 |
|---|------|----------|------|
| 6 | **Baseline LSTM** | 保持单路（无分支），hidden=128, layers=4 | ✅ |
| 7 | **双模式训练** | leave_one_out + cv10 统一入口 | ✅ |
| 8 | **论文对齐指标** | P@5/20, NDCG@5/20, MRR@5/20 | ✅ |
| 9 | **训练脚本统一** | `train_baseline.py` 支持两种模式 | ✅ |
| 10 | **结果格式一致** | test_results.pkl / cv_results.pkl | ✅ |

**新增文件**：
- `docs/phase1_*.md` — Phase 1 详细文档
- `docs/phase1_self_check.md` — Phase 1 自检报告

### 关键变更

**Phase 0 数据格式**：
- 旧格式：`[item1, item2, ...]`
- 新格式：`[(item1, ts1), (item2, ts2), ...]` — 保留 timestamp 用于 Phase 2/3

**Phase 0 划分模式**：
- leave_one_out: 单次 train/val/test 划分
- cv10: 10 个 fold，每个 fold 包含 train/test

**Phase 1 训练模式**：
- 单次模式：train/val/test，early stopping on val
- CV 模式：10 个 fold 分别训练，输出 mean±std

**文档**：
- [docs/phase0_split_modes.md](docs/phase0_split_modes.md) — 划分模式说明
- [docs/phase0_sequence_format.md](docs/phase0_sequence_format.md) — 序列格式说明
- [docs/phase1_baseline_design.md](docs/phase1_baseline_design.md) — Baseline 设计
- [docs/phase1_training_modes.md](docs/phase1_training_modes.md) — 训练模式说明
- [docs/phase1_self_check.md](docs/phase1_self_check.md) — Phase 1 自检报告
