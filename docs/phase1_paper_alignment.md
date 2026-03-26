# Phase 1 论文对齐修改说明

> **修改时间**: 2026-03-26
> **修改范围**: Phase 1 Baseline LSTM Training
> **修改原则**: 最小必要修改，仅对齐论文关键参数

---

## 一、论文对齐修改内容

### ✅ 修改 1: LSTM 层数对齐

**文件**: [configs/tafeng_baseline.yaml:23](../configs/tafeng_baseline.yaml#L23)

**修改前**:
```yaml
num_layers: 2
```

**修改后**:
```yaml
num_layers: 4  # Paper: num_layers = 4 (updated from 2)
```

**论文依据**: Section 7.3 (Hyperparameter Settings)
- 论文明确使用 4-layer LSTM
- Baseline 应与主模型架构对齐

**影响**:
- ✅ 模型容量与论文一致
- ✅ 可比性提升
- ⚠️ 训练时间略微增加

---

### ✅ 修改 2: Early Stopping 指标对齐

**文件**: [configs/tafeng_baseline.yaml:34](../configs/tafeng_baseline.yaml#L34)

**修改前**:
```yaml
metric_for_best: 'NDCG@10'
```

**修改后**:
```yaml
metric_for_best: 'NDCG@20'  # Paper-aligned: use NDCG@20 (main table metric)
```

**论文依据**: Table 2 & 3 (Main Results)
- 论文主表报告 NDCG@20
- Early stopping 应与主表指标一致

**影响**:
- ✅ 模型选择标准与论文一致
- ✅ 结果可比性提升

---

## 二、未修改项（保持现状）

### ✅ 评估指标（已正确）

**当前状态**:
- 主指标: Precision@5, Precision@20, NDCG@5, NDCG@20, MRR@5, MRR@20
- 附加指标: HR@K, Recall@K, NDCG@K (other K values)

**验证**:
```python
# src/evaluation/evaluator.py:38
PAPER_MAIN_METRICS = ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']
```

**结论**: ✅ 评估指标已正确，无需修改

---

### ✅ 数据切分协议（已支持）

**当前状态**:
- 默认: leave-one-out (`split_method: 'leave_one_out'`)
- 可选: 10-fold CV (`split_method: 'cv10'`)

**验证**:
```yaml
# configs/tafeng_baseline.yaml:14
split_method: 'leave_one_out'  # Options: 'leave_one_out', 'cv10', 'ratio'
```

**结论**: ✅ 数据切分已支持论文协议，无需修改

---

### ✅ 其他超参数（已对齐）

| 参数 | 论文值 | 当前值 | 状态 |
|------|--------|--------|------|
| `hidden_dim` | 128 | 128 | ✅ |
| `embed_dim` | - | 64 | ✅ (可接受) |
| `dropout` | - | 0.2 | ✅ (工程选择) |
| `learning_rate` | - | 0.001 | ✅ (工程选择) |
| `batch_size` | - | 256 | ✅ (工程选择) |

**结论**: ✅ 超参数合理，无需修改

---

## 三、修改前后对比

### 3.1 模型架构

| 组件 | 修改前 | 修改后 | 论文要求 |
|------|--------|--------|----------|
| **LSTM Layers** | 2 | **4** | 4 ✅ |
| **Hidden Dim** | 128 | 128 | 128 ✅ |
| **Embed Dim** | 64 | 64 | - |
| **Dropout** | 0.2 | 0.2 | - |

### 3.2 训练配置

| 配置项 | 修改前 | 修改后 | 论文要求 |
|--------|--------|--------|----------|
| **Early Stopping 指标** | NDCG@10 | **NDCG@20** | NDCG@20 ✅ |
| **Learning Rate** | 0.001 | 0.001 | - |
| **Batch Size** | 256 | 256 | - |

### 3.3 评估指标

| 指标 | 修改前 | 修改后 | 论文要求 |
|------|--------|--------|----------|
| **主指标** | Precision@5/20, NDCG@5/20, MRR@5/20 | Precision@5/20, NDCG@5/20, MRR@5/20 | ✅ |
| **附加指标** | HR@K, NDCG@K (other K) | HR@K, NDCG@K (other K) | ✅ |

---

## 四、兼容性影响分析

### 4.1 向后兼容性

**已训练模型**:
- ⚠️ 旧模型 (2-layer) 不能直接与新配置比较
- ✅ 但可作为额外的 ablation study

**训练脚本**:
- ✅ 无需修改，自动使用新配置

**评估脚本**:
- ✅ 无需修改，指标计算逻辑未变

### 4.2 性能影响

**训练时间**:
- 2-layer → 4-layer: 约 +30-50% 训练时间
- 参数量: 约 +2x

**预期性能**:
- 4-layer 模型容量更大，预期性能略优于 2-layer
- 与论文 baseline 更具可比性

---

## 五、验证检查清单

### 5.1 配置验证

```bash
# 检查配置文件
grep -E "num_layers|metric_for_best" configs/tafeng_baseline.yaml
# 应输出:
#   num_layers: 4
#   metric_for_best: 'NDCG@20'
```

### 5.2 模型验证

```bash
# 训练新模型
bash scripts/run_tafeng_baseline.sh single

# 验证模型层数
python <<EOF
import torch
model = torch.load('outputs/phase1_baseline/checkpoints/best_model.pt', weights_only=False)
print(f"Model layers: {model['model_state_dict']}")
# 应显示 4 个 LSTM 层的权重
EOF
```

### 5.3 评估验证

```bash
# 检查主指标
python <<EOF
import pickle
results = pickle.load(open('outputs/phase1_baseline/predictions/test_results.pkl', 'rb'))
main_metrics = ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']
for m in main_metrics:
    print(f"{m}: {results.get(m, 'N/A')}")
EOF
```

---

## 六、与论文的最终对齐状态

| 论文要求 | 当前状态 | 对齐程度 |
|----------|----------|----------|
| **4-layer LSTM** | ✅ num_layers = 4 | **完全对齐** |
| **hidden_size = 128** | ✅ hidden_dim = 128 | **完全对齐** |
| **Precision@5/20** | ✅ 支持 | **完全对齐** |
| **NDCG@5/20** | ✅ 支持 | **完全对齐** |
| **MRR@5/20** | ✅ 支持 | **完全对齐** |
| **10-fold CV** | ✅ 支持 | **完全对齐** |
| **Early stopping @ NDCG@20** | ✅ metric_for_best = NDCG@20 | **完全对齐** |

**总体评价**: ✅ Phase 1 Baseline 已与论文对齐

---

## 七、使用建议

### 7.1 新实验

**推荐做法**:
```bash
# 1. 清除旧模型（可选）
rm -rf outputs/phase1_baseline/checkpoints/*

# 2. 使用新配置训练
bash scripts/run_tafeng_baseline.sh single

# 3. 验证层数
# 检查日志确认模型参数量增加
```

### 7.2 与论文对比

**注意事项**:
- ✅ 使用相同数据集 (Tafeng)
- ✅ 使用相同评估指标 (Precision@5/20, NDCG@5/20, MRR@5/20)
- ✅ 使用相同 early stopping 指标 (NDCG@20)
- ✅ 模型架构对齐 (4-layer LSTM)

**预期结果**:
- Baseline 性能应接近论文 Table 2 的 Baseline 列
- 与 UPSTAR 的比较应具有可比性

---

## 八、常见问题

**Q: 为什么要改为 4-layer LSTM？**

A: 论文 Section 7.3 明确要求使用 4-layer LSTM。为保证可比性，baseline 必须使用相同架构。

**Q: 2-layer 模型还能用吗？**

A: 可以作为 ablation study，但不应作为论文对比的主要 baseline。

**Q: Early stopping 改为 NDCG@20 会影响训练吗？**

A: 可能略有影响，但更符合论文的模型选择标准，结果更具可比性。

**Q: 需要重新训练所有模型吗？**

A: 推荐重新训练 Phase 1 和 Phase 4，以确保 fair comparison。

---

**修改完成时间**: 2026-03-26
**修改者**: Phase 1 Baseline 模块
**审核状态**: ✅ 已通过论文对齐检查
