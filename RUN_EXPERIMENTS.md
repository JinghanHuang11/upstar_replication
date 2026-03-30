# UPSTAR 实验运行指南

**CV10-ONLY 模式：所有实验统一使用 10 折交叉验证**

## 🎯 核心原则

- **唯一运行模式**：10-fold cross-validation (cv10-only)
- **用户级别划分**：每个用户的完整序列只在一个 fold 中
- **内部验证集**：从训练数据划��� 10% 作为 early stopping 验证集
- **最终评估**：在独立的测试 fold 上评估

## 📋 完整实验流程

### 0. 数据预处理

```bash
bash scripts/run_preprocess.sh
```

**输出**：
- `data/processed/tafeng/metadata.pkl`
- `data/processed/tafeng/train_sequences.pkl`
- `data/processed/tafeng/test_sequences.pkl`

### 1. Item 表示学习（Item-GNN）

```bash
bash scripts/run_item_repr.sh
```

**输出**：
- `outputs/phase2_item_repr/checkpoints/item_embeddings.pt`

**论文对齐**：
- In/Out 邻居分离（Section 3.2）
- 隐藏维度：128（Section 7.2）

### 2. STB 计算

```bash
bash scripts/run_stb.sh
```

**输出**：
- `outputs/phase3_stb/checkpoints/stb_scores.npy`
- `outputs/phase3_stb/checkpoints/motivation_labels.npy`

**论文对齐**：
- Day-level 时间节点（Section 7.1：ρ=50）
- 互信息估计（Section 3.1.3）

### 3. Baseline 训练

```bash
bash scripts/run_tafeng_baseline.sh
```

**输出**：
- `outputs/phase1_baseline/results/cv_results.json`

**训练配置**：
- 10-fold cross-validation
- 每个 fold：2-4 小时
- 总计：20-40 小时

### 4. UPSTAR 训练

```bash
bash scripts/run_tafeng_upstar.sh
```

**输出**：
- `outputs/phase4_upstar/results/cv_results.json`

**训练配置**：
- 10-fold cross-validation
- 4 个训练阶段（Global → Branch → Orthogonality → Distillation）
- 每个 fold：8-16 小时
- 总计：80-160 小时

### 5. 结果比较

```bash
bash scripts/compare_results.sh
```

**输出**：
- `outputs/comparison/comparison.txt`
- `outputs/comparison/improvement_analysis.txt`

## 🔧 直接调用 Python 模块

如果你想直接调用 Python 而不是通过 shell 脚本：

### Baseline 训练

```bash
python -m src.training.train_baseline --config configs/tafeng_baseline.yaml
```

### UPSTAR 训练

```bash
python -m src.training.train_upstar_cv --config configs/tafeng_upstar.yaml
```

### Item 表示学习

```bash
python src/training/train_item_repr.py --config configs/item_repr.yaml
```

### STB 计算

```bash
python src/training/train_stb.py --config configs/stb.yaml
```

## 📁 结果目录结构

```
outputs/
├── phase1_baseline/          # Baseline 结果
│   └── results/
│       └── cv_results.json   # 10-fold CV 结果
├── phase2_item_repr/         # Item 表示
│   └── checkpoints/
│       └── item_embeddings.pt
├── phase3_stb/               # STB 结果
│   └── checkpoints/
│       ├── stb_scores.npy
│       └── motivation_labels.npy
├── phase4_upstar/            # UPSTAR 结果
│   └── results/
│       └── cv_results.json   # 10-fold CV 结果
└── comparison/               # 比较结果
    ├── comparison.txt
    └── improvement_analysis.txt
```

## ⚙️ 配置文件说明

| 配置文件 | 用途 | 核心参数 |
|---------|------|----------|
| `configs/tafeng_baseline.yaml` | Baseline 训练 | LSTM: 2层, hidden_dim=128 |
| `configs/tafeng_upstar.yaml` | UPSTAR 训练 | 3分支LSTM: 4层, hidden_dim=128 |
| `configs/item_repr.yaml` | Item-GNN | hidden_dim=128, layers=1 |
| `configs/stb.yaml` | STB 计算 | ρ=50, α=0.4, ε=0.1, β=40 |
| `configs/tafeng_eval.yaml` | 评估配置 | k_values: [1,5,10,15,20,50] |

## 🚀 快速开始

**首次运行完整实验**：

```bash
# 1. 数据预处理
bash scripts/run_preprocess.sh

# 2. Item 表示学习
bash scripts/run_item_repr.sh

# 3. STB 计算
bash scripts/run_stb.sh

# 4. Baseline 训练（20-40小时）
bash scripts/run_tafeng_baseline.sh

# 5. UPSTAR 训练（80-160小时）
bash scripts/run_tafeng_upstar.sh

# 6. 比较结果
bash scripts/compare_results.sh
```

## ⚠️ 重要注意事项

1. **CV10-ONLY**：所有实验必须使用 10-fold cross-validation
2. **时间估算**：
   - Baseline: 20-40 小时
   - UPSTAR: 80-160 小时
3. **GPU 要求**：建议使用 GPU 训练
4. **磁盘空间**：至少预留 50GB 空间
5. **中断恢复**：训练过程会保存 checkpoint，可以中断后恢复

## 🔍 验证安装

```bash
# 测试环境
bash scripts/test_baseline_eval.sh

# 清理旧数据（如需重新训练）
bash scripts/clean_old_data.sh
```

## 📊 主要评估指标

- **Precision@K**: K=1,5,10,15,20,50
- **NDCG@K**: K=1,5,10,15,20,50
- **MRR@K**: K=1,5,10,15,20,50
- **HitRate@K**: K=1,5,10,15,20,50

## 🎓 论文对齐检查

所有实验配置已与论文 Section 7 对齐：

- ✅ Module 1: Day-level item-time graph (Section 7.1: ρ=50)
- ✅ Module 2: Item-GNN in/out aggregation (Section 3.2)
- ✅ Module 3: STB mutual information (Section 3.1.3)
- ✅ Module 4: Dual teacher-student (Section 3.3)
- ✅ Module 5: Hyperparameters (Section 7)
