# UPSTAR Pipeline Steps

模块化的 UPSTAR 实验步骤，每个步骤可以独立运行或作为完整流程的一部分。

## 📋 步骤概述

| 步骤 | 名称 | 输出 | 预计时间 |
|------|------|------|----------|
| **Step 0** | 数据预处理 | `data/processed/tafeng/` | 10-20 分钟 |
| **Step 1** | Baseline 训练 | `outputs/phase1_baseline/` | 20-40 小时 |
| **Step 2** | Item 表示学习 | `outputs/phase2_item_repr/` | 1-2 小时 |
| **Step 3** | STB 计算 | `outputs/phase3_stb/` | 30-60 分钟 |
| **Step 4** | UPSTAR 训练 | `outputs/phase4_upstar/` | 80-160 小时 |
| **Step 5** | 评估和比较 | `outputs/comparison/` | 1-2 分钟 |

---

## 🚀 快速开始

### 方式 1: 运行完整流程

```bash
# 完整流程（所有步骤）
python run_pipeline.py

# 快速测试（1 fold，最少 epochs）
python run_pipeline.py --quick-test

# 从特定步骤开始
python run_pipeline.py --start-step 2

# 运行特定步骤范围
python run_pipeline.py --start-step 1 --stop-step 3
```

### 方式 2: 独立运行每个步骤

```bash
# Step 0: 数据预处理
python -m pipeline_steps.step0_preprocess

# Step 1: Baseline 训练
python -m pipeline_steps.step1_baseline

# Step 2: Item 表示学习
python -m pipeline_steps.step2_item_repr

# Step 3: STB 计算
python -m pipeline_steps.step3_stb

# Step 4: UPSTAR 训练
python -m pipeline_steps.step4_upstar

# Step 5: 评估和比较
python -m pipeline_steps.step5_evaluate
```

### 方式 3: 使用 Python 直接调用

```python
from pipeline_steps import Step0Preprocess, Step1Baseline, Step2ItemRepr

# Step 0: 预处理
step0 = Step0Preprocess()
step0.execute()

# Step 1: Baseline（快速测试）
step1 = Step1Baseline(num_folds=1, quick_test=True)
step1.execute()

# Step 2: Item 表示
step2 = Step2ItemRepr()
step2.execute()
```

---

## 📖 详细说明

### Step 0: 数据预处理

**功能**:
- 加载原始 Ta-Feng 数据
- 构建 CV10 划分
- 生成带时间戳的序列

**输出**:
```
data/processed/tafeng/
├── metadata.pkl           # 词表和元数据
├── cv_splits/            # 10-fold CV 划分
└── sequences.pkl         # 训练序列
```

**运行**:
```bash
# 默认配置
python -m pipeline_steps.step0_preprocess

# 自定义配置
python -m pipeline_steps.step0_preprocess --config configs/tafeng_baseline.yaml
```

---

### Step 1: Baseline 训练

**功能**:
- 训练 Baseline LSTM 模型
- 10-fold cross-validation
- 生成 CV 统计结果

**输出**:
```
outputs/phase1_baseline/
├── checkpoints/          # 每个 fold 的模型检查点
├── logs/                 # 训练日志
└── results/
    └── main_results.json   # CV10 统计结果
```

**运行**:
```bash
# 完整 10-fold CV
python -m pipeline_steps.step1_baseline

# 快速测试（1 fold, 3 epochs）
python -m pipeline_steps.step1_baseline --quick-test

# 2-fold CV
python -m pipeline_steps.step1_baseline --num-folds 2
```

---

### Step 2: Item 表示学习

**功能**:
- 构建 Item 图（in/out 邻居分离）
- 训练 Item-GNN
- 生成 item embeddings

**输出**:
```
outputs/phase2_item_repr/
├── checkpoints/
│   └── item_embeddings.pt  # Item embeddings
└── logs/                   # 训练日志
```

**运行**:
```bash
python -m pipeline_steps.step2_item_repr
```

---

### Step 3: STB 计算

**功能**:
- 构建日级别的 item-time 图
- 计算_STB_ 分数（互信息估计）
- 分类购买动机（stable/exploratory）

**输出**:
```
outputs/phase3_stb/
├── checkpoints/
│   ├── stb_scores.npy         # STB 分数
│   ├── motivation_labels.npy  # 动机标签
│   └── stb_stats.pkl          # 统计信息
└── logs/                       # 计算日志
```

**运行**:
```bash
python -m pipeline_steps.step3_stb
```

---

### Step 4: UPSTAR 训练

**功能**:
- 训练 UPSTAR 模型（三分支）
- 10-fold cross-validation
- Dual teacher-student 知识蒸馏

**输出**:
```
outputs/phase4_upstar/
├── checkpoints/          # 每个 fold 的模型检查点
├── logs/                 # 训练日志
├── predictions/          # 预测结果
└── results/
    └── main_results.json   # CV10 统计结果
```

**运行**:
```bash
# 完整 10-fold CV
python -m pipeline_steps.step4_upstar

# 快速测试（1 fold, 2 epochs per stage）
python -m pipeline_steps.step4_upstar --quick-test

# 2-fold CV
python -m pipeline_steps.step4_upstar --num-folds 2
```

---

### Step 5: 评估和比较

**功能**:
- 比较 Baseline 和 UPSTAR 结果
- 生成性能改善分析
- 输出可读性报告

**输出**:
```
outputs/comparison/
├── comparison.txt              # 可读性比较结果
└── improvement_analysis.txt   # 性能改善分析
```

**运行**:
```bash
python -m pipeline_steps.step5_evaluate
```

---

## 🔧 高级用法

### 运行特定步骤

```bash
# 只运行步骤 0, 1, 2
python run_pipeline.py --steps 0 1 2

# 只运行步骤 2, 3, 4
python run_pipeline.py --steps 2 3 4
```

### 跳过已存在的输出

```bash
# 自动跳过已完成的步骤
python run_pipeline.py --skip-existing
```

### 从失败处恢复

```bash
# 如果步骤 3 失败，直接从步骤 3 重新开始
python run_pipeline.py --start-step 3
```

### 自定义配置文件

```bash
# 使用自定义配置
python run_pipeline.py --baseline-config configs/my_baseline.yaml
python run_pipeline.py --upstar-config configs/my_upstar.yaml
```

---

## 📊 输出文件格式

### main_results.json 格式

```json
{
  "model_type": "baseline|upstar",
  "dataset": "tafeng",
  "num_folds": 10,
  "metrics": {
    "Precision@5": {
      "mean": 0.123,
      "std": 0.012
    },
    "NDCG@20": {
      "mean": 0.456,
      "std": 0.023
    }
  }
}
```

---

## 🎯 典型工作流程

### 首次运行

```bash
# 1. 先运行快速测试（30-60 分钟）
python run_pipeline.py --quick-test

# 2. 快速测试通过后，运行完整实验
python run_pipeline.py
```

### 分步运行

```bash
# 按顺序运行每个步骤
python -m pipeline_steps.step0_preprocess
python -m pipeline_steps.step1_baseline
python -m pipeline_steps.step2_item_repr
python -m pipeline_steps.step3_stb
python -m pipeline_steps.step4_upstar
python -m pipeline_steps.step5_evaluate
```

### 从中间恢复

```bash
# 如果在步骤 3 失败，从步骤 3 重新开始
python run_pipeline.py --start-step 3
```

---

## 🔍 故障排除

### 步骤失败

如果某个步骤失败：

1. 查看错误日志
2. 修复问题
3. 从失败的步骤重新开始

```bash
# 例如步骤 3 失败
python run_pipeline.py --start-step 3
```

### 清理并重新运行

```bash
# 清理特定阶段的输出
rm -rf outputs/phase1_baseline/
rm -rf outputs/phase2_item_repr/
rm -rf outputs/phase3_stb/
rm -rf outputs/phase4_upstar/

# 重新运行
python run_pipeline.py
```

---

## 📖 相关文档

- **[RUN_EXPERIMENTS.md](../RUN_EXPERIMENTS.md)** - 完整运行指南
- **[README.md](../README.md)** - 项目说明
- **[docs/SMOKE_TEST_GUIDE.md](../docs/SMOKE_TEST_GUIDE.md)** - Smoke test 指南

---

## ✅ 优势

相比 `run_full_experiment.py`，模块化步骤的优势：

1. **独立性**: 每个步骤可以独立运行和调试
2. **灵活性**: 可以选择运行特定步骤
3. **可恢复性**: 失败后可以从任意步骤恢复
4. **可测试性**: 每个步骤可以单独测试
5. **可维护性**: 代码更清晰，职责更单一

---

**项目状态**: ✅ Ready for GPU experiments

**代码质量**: ⭐⭐⭐⭐⭐ (模块化、清晰、易用)
