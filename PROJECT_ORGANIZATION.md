# UPSTAR 项目组织详解

> **最后更新**: 2026-03-26
> **项目状态**: ✅ Phase 1-6 全部完成 | Phase 4/5 论文对齐改进完成
> **代码规模**: 40 个 Python 源文件 | 9 个配置文件 | 8 个脚本
> **自检状态**: 12/12 核心检查项通过；待端到端训练验证

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

- **三分支建模**（Section 3.1.4）：
  - S-model (stable) / E-model (exploratory) / O-model (entire)
  - **三个独立 4-layer LSTM**（各有独立 Embedding / LSTM / Linear，无共享 backbone）
  - hidden_dim=128（Section 7.3）

- **全局融合**（Section 3.1.4）：
  - **逐维 Per-Dimension Gate** `[B,128,3]`，softmax 在 branch 维度
  - `z_global = gate_stab ⊙ z_stab + gate_expl ⊙ z_expl + gate_other ⊙ z_other`
  - `y_hat_global = Linear(z_global)`（不是加权 logit 平均）

- **联合损失**（Section 3.3）：
  - `L_total = L_global + 0.7·L_S&E&O + L_orth + L_distill`
  - `L_orth = τ_s·z_oᵀz_s + τ_e·z_oᵀz_expl`（论文原文，无 cosine similarity）
  - **条件双师生蒸馏**（按 next-item 动机 label 触发）

- **Dual Teacher-Student**（Section 3.3）：

  | target item 动机 | teacher | student | uncategorized (label=2) |
  |---|---|---|---|
  | stable (label=1) | S-model | E-model | **不触发** |
  | exploratory (label=0) | E-model | S-model | **不触发** |

- **Item-GNN**：区分 in/out 邻居的 Message Passing

### 数据集

- **Tafeng**：台湾超市数据集 (26K 用户, 15K 商品, 785K 交互)
- **Baseline**：测试数据集 (10 用户, 10 商品, 51 交互)

---

## 二、论文对齐改进（2026-03-26）

### Phase 4 最新改进（2026-03-26）⭐

| 模块 | 改进内容 | 论文章节 | 状态 |
|------|----------|----------|------|
| **逐维 FusionGate** | 标量 `[B,3]` → **逐维 `[B,128,3]`**；softmax 从 dim=1 → dim=2 | Section 3.1.4 | ✅ |
| **y_hat_global** | 移除加权 logit 平均；改为 **`Linear(z_global)`** | Section 3.1.4 | ✅ |
| **L_orth** | 去掉 `F.normalize`（cosine similarity）；改为论文原文 **raw dot product** `τ_s·z_oᵀz_s + τ_e·z_oᵀz_expl`（无 squared） | Section 3.3 | ✅ |
| **L_distill** | 完整实现**条件双师生蒸馏**：stable→S→E, exploratory→E→S, uncategorized→无蒸馏；teacher `.detach()`，student 保留梯度 | Section 3.3 | ✅ |
| **联合损失公式** | 统一 `total_loss = λg·Lg + λb·Lb + λo·Lo + λd·Ld` | Section 3.3 | ✅ |
| **Bug 修复** | `UPSTARMoodel` → `UPSTARModel`；`loss_loss_stab` 键名；Gate 日志 `[B,128]` 均值 | — | ✅ |

### Phase 5 最新改进（2026-03-26）⭐

| 模块 | 改进内容 | 状态 |
|------|----------|------|
| **主指标口径** | 统一为 `P@5, P@20, NDCG@5, NDCG@20, MRR@5, MRR@20`（论文 Table 2/3 格式） | ✅ |
| **输出结构** | JSON/CSV：`main_metrics` + `additional_metrics` 分段输出 | ✅ |
| **命名规范** | 移除 `HR@K` 默认命名；单正样本用 `Precision@K`（论文口径） | ✅ |
| **配置驱动** | `k_values` 从 `config['evaluation']['k_values']` 而非硬编码读取 | ✅ |

### Phase 3 改进（2026-03-26）

| 模块 | 改进内容 | 论文章节 | 状态 |
|------|----------|----------|------|
| **STB 计算** | 从 mean cosine 改为 **worst-case (min)** aggregation | Section 3.1.3 | ✅ |
| **MI 估计** | 从 simplified proxy 改为 **完整 MINE 下界**（`joint - marginal`） | Section 3.1.3 | ✅ |
| **MI 梯度** | 去除 `no_grad`，**真正可微分**的 adversarial perturbation | Section 3.1.3 | ✅ |
| **Bipartite 约束** | **强制验证** item-time 图结构（无非法边） | Section 3.1.3 | ✅ |
| **Graph Summary** | **分离 item/time summaries** + 度统计 | Section 3.1.3 | ✅ |
| **拓扑扰动** | **强制 bipartite edges**（item→time only） | Section 3.1.3 | ✅ |
| **分类阈值** | **Config 驱动** 50/40/10 划分（全局排序） | Section 3.1.4 | ✅ |

### 之前已对齐的 5 个关键模块

| 模块 | 对齐内容 | 论文章节 | 状态 |
|------|----------|----------|------|
| **1. Item-Time Graph** | 真实 day-level 时间节点 | Section 7.1 | ✅ |
| **2. Item-GNN** | 区分 in/out 邻居 | Section 3.2 | ✅ |
| **3. STB** | 互信息近似（MINE） | Section 3.1.3 | ✅ |
| **4. 序列模型** | 三路预测 + Dual Teacher-Student | Section 3.1.4, 3.3 | ✅ |
| **5. 超参数** | 配置对齐 Section 7 | Section 7 | ✅ |

---

## 三、超参数对齐（Section 7.3 / 7.4）

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
| **联合训练 (Section 7.4)** | optimizer | Adam | Adam ✅ |
| | lr | 3e-4 | 0.0003 ✅ |
| | batch_size | 64 | 64 ✅ |
| | λ (branch loss) | 0.7 | 0.7 ✅ |
| | λ (distill) | 0.7 | 0.7 ✅ |
| | τ_s | 0.5 | 0.5 ✅ |
| | τ_e | 0.5 | 0.5 ✅ |
| | temperature | - | 3.0 ✅ |

**对应文件**：`configs/tafeng_upstar.yaml`, `configs/stb.yaml`, `configs/item_repr.yaml`

---

## 四、Phase 4 详细改进说明

### Phase 4 改进 1: 逐维 FusionGate

**修改前**：标量 gate `[B, 3]`，所有维度共享同一组权重

**修改后**：`fusion.py:75-83`

```python
# 逐维 gate: Linear(384→384) → reshape → softmax
concat = torch.cat([z_stab, z_expl, z_other], dim=1)  # [B, 384]
gate_logits = self.gate_network(concat)                  # [B, 384]
gate_logits_per_dim = gate_logits.view(batch_size, H, 3)  # [B, 128, 3]
f_s = torch.softmax(gate_logits_per_dim, dim=2)          # softmax on branch dim
z_global = f_s[:,:,0]*z_stab + f_s[:,:,1]*z_expl + f_s[:,:,2]*z_other
```

### Phase 4 改进 2: y_hat_global

**修改前**：`y_hat_global = weighted_average(y_hat_stab, y_hat_expl, y_hat_other)`

**修改后**：`y_hat_global = Linear(z_global)`（`fusion.py:215`）

### Phase 4 改进 3: L_orth — 论文原文

**修改前**：`loss = self.tau_s * F.cosine_similarity(z_other, z_stab).mean() + ...`

**修改后**：`losses.py:205-209`

```python
# 论文原文: L_orth = τ_s * z_other^T z_stab + τ_e * z_other^T z_expl
# 无 normalize，无 squared，无 cosine
dot_s = (z_other * z_stab).sum(dim=1)    # [batch_size] per sample
dot_e = (z_other * z_expl).sum(dim=1)
loss = self.tau_s * dot_s.mean() + self.tau_e * dot_e.mean()
```

### Phase 4 改进 4: L_distill — 完整条件双师生

`losses.py:265-327`

```python
# 按 next-item 动机 label 条件触发
stable_mask = (motivation_labels == 1)   # S → E: S teaches E
expl_mask   = (motivation_labels == 0)   # E → S: E teaches S
# label == 2 (uncategorized): 两个分支均跳过，不触发蒸馏

if num_stable > 0:
    teacher = y_hat_stab[stable_mask].detach()   # teacher: detach
    student = y_hat_expl[stable_mask]             # student: keep grad
    kl_s_to_e = nn.functional.kl_div(log_softmax(student/T), softmax(teacher/T), reduction='batchmean')
    loss_s_to_e = self.distill_lambda * kl_s_to_e
```

### Phase 4 改进 5: 联合损失公式

`losses.py:394-399`

```python
total_loss = (
    self.lambda_global  * global_loss +          # λ_global = 1.0
    self.lambda_branch  * branch_losses['total_branch'] +  # λ_branch = 0.7
    self.lambda_ortho   * ortho_loss +            # λ_ortho = 1.0
    self.lambda_distill * distill_losses['loss_distill']   # λ_distill = 1.0
)
```

---

## 五、Phase 5 详细改进说明

### Phase 5 改进 1: 主指标口径

`evaluator.py:44`

```python
PAPER_MAIN_METRICS = ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']
```

`evaluate()` 返回结构化结果：
```python
{
    'main_metrics': {...},           # 论文主表六项
    'additional_metrics': {...},     # @1/@10/@15/Recall 等
    '_flat': {...}                   # 向后兼容：全部合并
}
```

### Phase 5 改进 2: JSON/CSV 分段输出

`evaluator.py:218-243`

**JSON**：
```json
{
  "main_metrics": {"Precision@5": 0.0523, ...},
  "additional_metrics": {"NDCG@10": 0.0967, ...}
}
```

**CSV**：
```
# Paper Main Table
Metric,Value (%)
Precision@5,5.23%
...

# Additional Engineering Metrics
Metric,Value (%)
NDCG@10,9.67%
...
```

### Phase 5 改进 3: 报告生成感知结构化输入

`report.py` 自动检测：
- **结构化输入**：`{'main_metrics': {...}, 'additional_metrics': {...}}` — 直接使用
- **扁平输入**：`{metric_name: value}` — 全部归入 main_metrics（向后兼容）

### Phase 5 改进 4: Comparison Table 更新

`comparison.py` — `generate_executive_summary()` 不再硬编码 `NDCG@10`/`Recall@20`，统一使用 `PAPER_MAIN_METRICS` 常量。

---

## 六、六个实验阶段

### Phase 0: 数据预处理

**目标**：加载原始数据，构建用户购买序列，划分 train/val/test

**输入**：`data/raw/ta_feng.csv`

**输出**：
- `data/processed/tafeng/metadata.pkl`
- `data/processed/tafeng/train_sequences.pkl` — **新格式**：`[(item, timestamp), ...]`
- `data/processed/tafeng/val_sequences.pkl`
- `data/processed/tafeng/test_sequences.pkl`

**脚本**：`bash scripts/run_preprocess.sh`

**⚠️ 重要**：由于序列格式更新，需要重新运行预处理！

---

### Phase 1: Baseline 训练

**目标**：训练 LSTM 基线模型，确立性能基准

**输入**：Phase 0 序列数据

**输出**：`outputs/baselines/tafeng/single_run/checkpoints/best_model.pt`

**脚本**：`bash scripts/run_tafeng_baseline.sh [single|cv]`

**性能**：NDCG@10 ~0.097, P@10 ~0.097, MRR@10 ~0.040

---

### Phase 2: Item 表示学习

**目标**：在商品共现图上训练 Item-GNN，生成商品 Embedding

**输入**：Phase 0 序列 + 图结构

**输出**：`outputs/checkpoints/item_repr/item_embeddings.pt`

**脚本**：`bash scripts/run_item_repr.sh`

**核心模块**：
- `src/graphs/item_graph.py` — 会话内图 + 跨会话图
- `src/models/item_gnn.py` — **in/out 邻居分离聚合**

**⚠️ 重要**：由于更新了 Item-GNN 架构，需要重新训练！

---

### Phase 3: STB 计算

**目标**：基于商品-时间图计算每个商品的 STB 分数

**输入**：Phase 2 商品 Embedding

**输出**：
- `outputs/phase3_stb/checkpoints/stb_scores.npy`
- `outputs/phase3_stb/checkpoints/motivation_labels.npy` — 0=exploratory, 1=stable, 2=uncategorized

**脚本**：`bash scripts/run_stb.sh` 或 `bash scripts/run_stb_advanced.sh`

**核心改进**（Phase 3 最新）：
- **Worst-case aggregation**：`min` over perturbation rounds（`stb_encoder.py`）
- **完整 MINE 下界**：`joint - marginal`（`stb_encoder.py`）
- **真正可微分 MI 梯度**：移除 `no_grad`（`perturbation_advanced.py`）
- **Bipartite 约束**：强制 item→time 边（`item_time_graph.py`）
- **Config 驱动阈值**：50/40/10 全局排序

---

### Phase 4: UPSTAR 训练

**目标**：训练三分支模型，结合全局融合进行推荐

**输入**：Phase 2 Embedding + Phase 3 STB 标签

**输出**：`outputs/phase4_upstar/checkpoints/model_after_stage4.pt`

**脚本**：`bash scripts/run_tafeng_upstar.sh`

**核心模块**（Phase 4 改进）：
- `src/models/fusion.py` — **逐维 FusionGate `[B,128,3]`**，`y_hat_global = Linear(z_global)`
- `src/models/sequence_models.py` — 三个**独立 4-layer LSTM**（无共享 backbone）
- `src/data/upstar_dataset.py` — **uncategorized 保留在 O-model**；空子序列 len=1+PAD
- `src/training/losses.py` — **L_orth 论文原文**；**完整条件双师生蒸馏**
- `src/training/train_upstar.py` — 联合损失；四阶段课程（engineering）
- `configs/tafeng_upstar.yaml` — **lr=3e-4, batch=64, Adam, λ=0.7, τ_s=τ_e=0.5**

**四阶段训练（Section 3.3）**：

> ⚠️ 四阶段课程是**工程实现**，目的是逐步开启损失项以稳定训练。
> 论文直接用联合损失（Section 3.3）。文件内已标注"engineering curriculum, not paper modification"。

| Stage | 启用损失 | 说明 |
|-------|----------|------|
| Stage 1 | L_global | 全局预测 warm-up |
| Stage 2 | + λ·L_S&E&O | 添加分支监督 |
| Stage 3 | + L_orth | 引入 O↔S/E 正交约束 |
| Stage 4 | + L_distill | 双师生知识蒸馏 |

---

### Phase 5: 评估与对比

**目标**：评估 Baseline 和 UPSTAR，生成对比报告

**输入**：Phase 1 和 Phase 4 的模型

**输出**：`outputs/results/comparison/comparison.json` + 对比报告

**脚本**：`bash scripts/run_eval.sh`

**核心模块**（Phase 5 改进）：
- `src/evaluation/evaluator.py` — **主/附加指标分离**；结构化返回
- `src/evaluation/report.py` — **主表 + 附加分段输出**
- `src/evaluation/comparison.py` — **主表六项对比**（移除硬编码 NDCG@10/Recall@20）

**主指标（论文 Table 2/3 格式）**：
- `P@5`, `P@20` — Precision@K（不是 HR@K）
- `NDCG@5`, `NDCG@20`
- `MRR@5`, `MRR@20`

---

## 七、完整 Pipeline

### 一键运行（推荐）

```bash
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

---

## 八、Phase 4/5 自检清单

| # | 检查项 | 对应文件 | 状态 |
|---|--------|----------|------|
| 1 | 三路序列正确拆为 stable/exploratory/entire | `upstar_dataset.py:106-116` | ✅ |
| 2 | uncategorized items 保留在 O-model 输入 | `upstar_dataset.py:103` | ✅ |
| 3 | S/E/O 是三个独立 4-layer LSTM | `sequence_models.py` + `upstar.py` | ✅ |
| 4 | hidden size = 128 | `tafeng_upstar.yaml:24,53` | ✅ |
| 5 | FusionGate 是逐维 softmax gate `[B,128,3]` | `fusion.py:82-83` | ✅ |
| 6 | z_global 由三路 gate-weighted 融合 | `fusion.py:98-102` | ✅ |
| 7 | L_global / L_S&E&O / L_orth / L_distill 分别存在并记录 | `losses.py` + `train_upstar.py` | ✅ |
| 8 | dual teacher-student 按 next-item motivation label 条件触发 | `losses.py:265-266` | ✅ |
| 9 | uncategorized next item 不误触发蒸馏 | `losses.py:265-266`（两个分支均跳过） | ✅ |
| 10 | 优化器超参数对齐论文 | `tafeng_upstar.yaml:64-84` | ✅ |
| 11 | 主输出为 P@5/20, NDCG@5/20, MRR@5/20 | `evaluator.py:44` | ✅ |
| 12 | 未修改 Phase 2/3 核心方法 | `git diff HEAD` | ✅ |

**12/12 全部通过** ✅

---

## 九、仍存在的工程差距

| 项目 | 类型 | 说明 |
|------|------|------|
| 四阶段课程 | **工程实现** | 论文直接用联合损失；课程是稳定性工程技巧，已在文件中明确标注 |
| L_orth batch 平均方式 | 自由度 | 论文未明确指定；对 batch 内所有样本 dot product 结果求均值 |
| Early stop 指标 | 工程近似 | 论文主表用 NDCG@20；配置用 NDCG@10 |
| 端到端训练验证 | **待完成** | 各模块自检通过，尚未运行完整 4-stage 训练流程 |

---

## 十、源代码组织

### 模块依赖关系

```
src/data/
    ↓（序列数据 with timestamps）
src/graphs/
    ↓（商品图 + day-level item-time 二部图）
src/models/
    ↓（模型定义：in/out Item-GNN, MI-based STB, 逐维 FusionGate）
src/training/
    ↓（训练：条件双师生蒸馏 + 联合损失）
src/evaluation/
    ↓（评估：P@5/20, NDCG@5/20, MRR@5/20）
src/experiments/
src/utils/
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
    ↓（MI 估计 + worst-case aggregation）
STB Labels (0=exploratory, 1=stable, 2=uncategorized)
    ↓
UPSTAR Dataset (序列按动机拆为 stable/exploratory/entire)
    ↓
S/E/O 独立 LSTM (各 4-layer, hidden=128) → 逐维 FusionGate [B,128,3]
    ↓
y_hat_global = Linear(z_global)
    ↓（L_total = L_global + 0.7·L_S&E&O + L_orth + L_distill）
评估指标 (P@5/20, NDCG@5/20, MRR@5/20)
```

---

**最后更新**: 2026-03-26
