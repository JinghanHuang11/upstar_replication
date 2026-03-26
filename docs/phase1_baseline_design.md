# Phase 1 Baseline LSTM 设计说明

> **更新时间**: 2026-03-26
> **模型类型**: Plain LSTM Baseline
> **设计原则**: 保持简单，超参数与论文对齐

---

## 一、Baseline 模型架构

### 1.1 模型结构

```
Input Sequence: [item_1, item_2, ..., item_n]
        ↓
Item Embedding: [embed_dim = 64]
        ↓
   LSTM Encoder: [hidden_dim = 128, num_layers = 4]
        ↓
      Dropout: [rate = 0.2]
        ↓
Output Projection: Linear(hidden_dim, num_items)
        ↓
  Logits: [batch_size, num_items]
```

**关键特征**:
- ✅ **单路建模**: Single-path LSTM，无分支
- ✅ **直接预测**: LSTM hidden state → Linear → item logits
- ✅ **简单架构**: 无 fusion gate，无 STB，无 motivation

---

## 二、论文口径统一 vs Baseline 简化

### 2.1 论文对齐部分（超参数��

| 超参数 | Baseline 值 | UPSTAR 值 | 论文依据 | 对齐状态 |
|--------|-------------|-----------|----------|----------|
| **hidden_dim** | 128 | 128 | Section 7.3 | ✅ 完全对齐 |
| **num_layers** | 4 | 4 | Section 7.3 | ✅ 完全对齐 |
| **embed_dim** | 64 | 64 | Section 7.3 | ✅ 完全对齐 |
| **max_seq_length** | 50 | 50 | 工程选择 | ✅ 一致 |
| **dropout** | 0.2 | 0.2 | 工程选择 | ✅ 一致 |

**统一原因**: 确保公平比较，baseline 和 UPSTAR 的模型容量相近

---

### 2.2 Baseline 简化部分（架构）

| 组件 | Baseline | UPSTAR | 简化原因 |
|------|----------|--------|----------|
| **分支结构** | Single LSTM | S/E/O 3-branch LSTM | Baseline 无需动机建模 |
| **Fusion Gate** | ❌ 无 | ✅ 逐维 gate | Baseline 单路预测 |
| **STB 模块** | ❌ 无 | ✅ MI-based STB | Baseline 无需稳定性建模 |
| **Motivation Labels** | ❌ 无 | ✅ 使用 | Baseline 无需动机信息 |
| **Item Graph Embedding** | ❌ 无 | ✅ 预训练 + GNN | Baseline 从零训练 |
| **损失函数** | CrossEntropy | L_total = Lg + λb + Lo + Ld | Baseline 简单监督 |

**简化原因**:
1. **Baseline 定义**: Sequential recommendation 的标准基线
2. **公平对比**: UPSTAR 的改进应体现在性能提升上
3. **可复现性**: Plain LSTM 是广泛接受的 baseline

---

## 三、模型配置

### 3.1 当前配置

**文件**: [configs/tafeng_baseline.yaml](../configs/tafeng_baseline.yaml)

```yaml
# Model Configuration
model:
  # Plain LSTM Baseline (no S/E/O branches, no fusion gate, no STB)
  embed_dim: 64
  hidden_dim: 128            # Paper-aligned: Section 7.3
  num_layers: 4              # Paper-aligned: Section 7.3
  dropout: 0.2
  max_seq_length: 50
```

### 3.2 模型代码

**文件**: [src/models/baseline_lstm.py](../src/models/baseline_lstm.py)

**核心组件**:
```python
class LSTMRec(nn.Module):
    def __init__(self, num_items, embed_dim=64, hidden_dim=128, num_layers=4, ...):
        # Item embedding
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim)

        # Single-path LSTM (no branches)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers)

        # Direct output projection (no fusion gate)
        self.output_proj = nn.Linear(hidden_dim, num_items + 1)
```

**关键代码注释**:
- Line 16-22: 明确标注单路 LSTM 架构
- Line 24-42: 详细注释论文对齐参数 vs baseline 简化

---

## 四、输出接口

### 4.1 训练输出

**Checkpoint 格式**:
```python
{
    'epoch': int,
    'model_state_dict': dict,  # LSTMRec 模型参数
    'optimizer_state_dict': dict,
    'metrics': {
        'Precision@5': float,
        'Precision@20': float,
        'NDCG@5': float,
        'NDCG@20': float,
        'MRR@5': float,
        'MRR@20': float,
        # ... 其他指标
    }
}
```

**输出路径**:
- Best model: `outputs/phase1_baseline/checkpoints/best_model.pt`
- All checkpoints: `outputs/phase1_baseline/checkpoints/checkpoint_epoch_*.pt`

### 4.2 评估输出

**Test Results 格式**:
```python
{
    # 主指标（论文 Table 2/3 格式）
    'Precision@5': float,
    'Precision@20': float,
    'NDCG@5': float,
    'NDCG@20': float,
    'MRR@5': float,
    'MRR@20': float,

    # 附加指标（工程分析用）
    'HR@1': float,
    'HR@5': float,
    'HR@10': float,
    'HR@15': float,
    'HR@20': float,
    'HR@50': float,
    # ... 其他 K 值
}
```

**输出路径**:
- `outputs/phase1_baseline/predictions/test_results.pkl`

---

## 五、与 UPSTAR 的对比

### 5.1 模型复杂度对比

| 组件 | Baseline | UPSTAR | 复杂度比 |
|------|----------|--------|----------|
| **参数量** | ~1.2M | ~2.5M | 1 : 2.1 |
| **LSTM 数量** | 1 | 3 (S/E/O) | 1 : 3 |
| **Fusion Gate** | 0 | 1 (逐维) | - |
| **预训练模块** | 0 | 2 (Item GNN, STB) | - |

**结论**: Baseline 是简化的单路 LSTM，UPSTAR 通过复杂化提升性能

---

### 5.2 预期性能对比

**论文 Table 2** (Tafeng 数据集):
```
Baseline (LSTM):  NDCG@10 ≈ 0.097
UPSTAR:           NDCG@10 ≈ 0.103 (+6.2%)
```

**当前配置预期**:
- Baseline 应接近论文的 Baseline 性能
- 与 UPSTAR 的差距应体现论文的提升幅度

---

## 六、验证检查清单

### 6.1 架构验证

```python
# 验证模型是单路 LSTM
model = LSTMRec(num_items=1000, hidden_dim=128, num_layers=4)

# 检查组件
assert hasattr(model, 'lstm'), "Should have LSTM"
assert hasattr(model, 'output_proj'), "Should have output projection"
assert not hasattr(model, 'fusion_gate'), "Should NOT have fusion gate"
assert not hasattr(model, 'stb_encoder'), "Should NOT have STB"

# 检查 LSTM 配
assert model.lstm.num_layers == 4, "Should be 4-layer LSTM"
assert model.lstm.hidden_size == 128, "Should have hidden_size=128"

print("[OK] Baseline architecture verified")
```

### 6.2 超参数验证

```bash
# 检查配置文件
grep -E "hidden_dim|num_layers|embed_dim" configs/tafeng_baseline.yaml

# 应输出:
#   hidden_dim: 128
#   num_layers: 4
#   embed_dim: 64
```

### 6.3 输出验证

```bash
# 训练后检查
python <<EOF
import torch
checkpoint = torch.load('outputs/phase1_baseline/checkpoints/best_model.pt', weights_only=False)

# 检查模型结构
state_dict = checkpoint['model_state_dict']
print(f"LSTM layers: {len([k for k in state_dict.keys() if 'lstm' in k])}")

# 检查指标
metrics = checkpoint['metrics']
for k in ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20']:
    print(f"{k}: {metrics.get(k, 'N/A')}")
EOF
```

---

## 七、常见问题

**Q: 为什么 baseline 不用 S/E/O 三分支？**

A: Baseline 的定义就是 plain LSTM。三分支是 UPSTAR 的核心创新，如果 baseline 也用三分支，就无法体现 UPSTAR 的改进。

**Q: 为什么 baseline 不用预训练的 Item GNN embedding？**

A: 为确保公平对比，baseline 应从零训练。UPSTAR 的性能提升应来自整体架构，而非单个组件。

**Q: 为什么 hidden_dim 和 num_layers 要与 UPSTAR 一致？**

A: 确保模型容量相近，这样性能差异才能归因于架构创新（三分支 + STB + Fusion），而非模型大小。

**Q: Baseline 的 embed_dim 为什么是 64，UPSTAR 也是 64？**

A: 这是论文的配置。如果 embed_dim 不同，会影响模型容量和可比性。

**Q: 可以用其他 baseline 吗（如 GRU、Transformer）？**

A: 论文只对比了 LSTM baseline。如果要对比其他模型，应作为额外的 ablation study，而非替代 LSTM baseline。

---

## 八、总结

### ✅ 论文口径统一（超参数）

| 项目 | 统一内容 | 目的 |
|------|----------|------|
| **hidden_dim = 128** | Baseline 和 UPSTAR 相同 | 确保模型容量相近 |
| **num_layers = 4** | Baseline 和 UPSTAR 相同 | 确保模型深度相近 |
| **embed_dim = 64** | Baseline 和 UPSTAR 相同 | 确保 embedding 容量相近 |

### ✅ Baseline 保持简单（架构）

| 简化项 | 内容 | 原因 |
|--------|------|------|
| **单路 LSTM** | 无 S/E/O 三分支 | Baseline 定义 |
| **无 Fusion Gate** | 直接预测 | 简单架构 |
| **无 STB** | 不使用动机信息 | Plain LSTM |
| **无预训练** | 从零训练 | 公平对比 |

### 🎯 设计目标

1. **可比性**: 超参数与 UPSTAR 对齐
2. **简洁性**: 架构保持简单，不引入复杂组件
3. **可复现性**: 广泛接受的 LSTM baseline
4. **公平性**: 性能差异体现 UPSTAR 的创新价值

---

**设计完成时间**: 2026-03-26
**模型类型**: Plain LSTM Baseline
**论文对齐**: ✅ 超参数已对齐
**架构简化**: ✅ 保持 baseline 简单
