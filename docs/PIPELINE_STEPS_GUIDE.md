# Pipeline Steps 使用指南

## 📋 概述

我已经为你创建了一个完整的模块化 Pipeline 步骤系统，位于 `pipeline_steps/` 文件夹中。每个步骤都可以独立运行或作为完整流程的一部分。

## 🗂️ 文件结构

```
upstar/
├── pipeline_steps/               # ⭐ 新建：模块化步骤
│   ├── __init__.py              # 包初始化
│   ├── base_step.py             # 基础步骤类
│   ├── step0_preprocess.py      # Step 0: 数据预处理
│   ├── step1_baseline.py        # Step 1: Baseline 训练
│   ├── step2_item_repr.py       # Step 2: Item 表示学习
│   ├── step3_stb.py             # Step 3: STB 计算
│   ├── step4_upstar.py          # Step 4: UPSTAR 训练
│   ├── step5_evaluate.py        # Step 5: 评估和比较
│   └── README.md                # 详细说明文档
│
├── run_pipeline.py              # ⭐ 新建：模块化总控脚本
├── run_full_experiment.py       # 原有：Shell 脚本总控
│
└── docs/
    └── PIPELINE_STEPS_GUIDE.md  # 本文档
```

## 🚀 快速开始

### 方式 1: 使用新的模块化总控脚本（推荐）

```bash
# 运行完整流程
python run_pipeline.py

# 快速测试（30-60 分钟）
python run_pipeline.py --quick-test

# 从特定步骤开始
python run_pipeline.py --start-step 2

# 运行特定步骤
python run_pipeline.py --steps 0 1 2

# 跳过已存在的输出
python run_pipeline.py --skip-existing
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

### 方式 3: 使用 Python 代码调用

```python
from pipeline_steps import (
    Step0Preprocess,
    Step1Baseline,
    Step2ItemRepr,
    Step3Stb,
    Step4Upstar,
    Step5Evaluate
)

# Step 0: 预处理
step0 = Step0Preprocess()
step0.execute()

# Step 1: Baseline（快速测试）
step1 = Step1Baseline(num_folds=1, quick_test=True)
step1.execute()

# Step 2: Item 表示
step2 = Step2ItemRepr()
step2.execute()

# Step 3: STB
step3 = Step3Stb()
step3.execute()

# Step 4: UPSTAR（快速测试）
step4 = Step4Upstar(num_folds=1, quick_test=True)
step4.execute()

# Step 5: 评估
step5 = Step5Evaluate()
step5.execute()
```

## 📖 各步骤详解

### Step 0: 数据预处理

**功能**: 加载原始数据，构建 CV10 划分

**输出**:
- `data/processed/tafeng/metadata.pkl`
- `data/processed/tafeng/cv_splits/`

**运行**:
```bash
python -m pipeline_steps.step0_preprocess
```

---

### Step 1: Baseline 训练

**功能**: 训练 Baseline LSTM 模型（10-fold CV）

**输出**:
- `outputs/phase1_baseline/results/cv_results.json`
- `outputs/phase1_baseline/checkpoints/`

**运行**:
```bash
# 完整训练
python -m pipeline_steps.step1_baseline

# 快速测试
python -m pipeline_steps.step1_baseline --quick-test

# 自定义 fold 数量
python -m pipeline_steps.step1_baseline --num-folds 2
```

---

### Step 2: Item 表示学习

**功能**: 训练 Item-GNN，生成 item embeddings

**输出**:
- `outputs/phase2_item_repr/checkpoints/item_embeddings.pt`

**运行**:
```bash
python -m pipeline_steps.step2_item_repr
```

---

### Step 3: STB 计算

**功能**: 计算 STB 分数和动机标签

**输出**:
- `outputs/phase3_stb/checkpoints/stb_scores.npy`
- `outputs/phase3_stb/checkpoints/motivation_labels.npy`

**运行**:
```bash
python -m pipeline_steps.step3_stb
```

---

### Step 4: UPSTAR 训练

**功能**: 训练 UPSTAR 模型（10-fold CV）

**输出**:
- `outputs/phase4_upstar/results/cv_results.json`
- `outputs/phase4_upstar/checkpoints/`

**运行**:
```bash
# 完整训练
python -m pipeline_steps.step4_upstar

# 快速测试
python -m pipeline_steps.step4_upstar --quick-test

# 自定义 fold 数量
python -m pipeline_steps.step4_upstar --num-folds 2
```

---

### Step 5: 评估和比较

**功能**: 比较 Baseline 和 UPSTAR 结果

**输出**:
- `outputs/comparison/comparison.txt`
- `outputs/comparison/improvement_analysis.txt`

**运行**:
```bash
python -m pipeline_steps.step5_evaluate
```

## 🔧 高级用法

### 运行特定步骤

```bash
# 只运行步骤 0, 1, 2
python run_pipeline.py --steps 0 1 2

# 只运行步骤 2, 3, 4
python run_pipeline.py --steps 2 3 4
```

### 步骤范围

```bash
# 运行步骤 1-3
python run_pipeline.py --start-step 1 --stop-step 3

# 运行步骤 2-5
python run_pipeline.py --start-step 2 --stop-step 5
```

### 跳过已存在的输出

```bash
# 自动跳过已完成的步骤
python run_pipeline.py --skip-existing
```

### 自定义配置

```bash
# 使用自定义配置
python run_pipeline.py \
    --baseline-config configs/my_baseline.yaml \
    --upstar-config configs/my_upstar.yaml
```

## 🎯 典型使用场景

### 场景 1: 首次运行完整实验

```bash
# 1. 先运行快速测试
python run_pipeline.py --quick-test

# 2. 快速测试通过后，运行完整实验
python run_pipeline.py
```

### 场景 2: 从失败处恢复

```bash
# 如果步骤 3 失败，从步骤 3 重新开始
python run_pipeline.py --start-step 3
```

### 场景 3: 只运行部分步骤

```bash
# 只运行预处理和 Baseline
python run_pipeline.py --steps 0 1

# 只运行 UPSTAR 和评估
python run_pipeline.py --steps 4 5
```

### 场景 4: 调试单个步骤

```bash
# 独立运行步骤 3 并查看详细日志
python -m pipeline_steps.step3_stb
```

### 场景 5: Python 交互式调试

```python
# 在 Python 中调试步骤 2
from pipeline_steps import Step2ItemRepr

step = Step2ItemRepr()
step.check_prerequisites()  # 检查依赖
step.run()  # 运行
```

## 📊 两种总控脚本对比

| 特性 | run_full_experiment.py | run_pipeline.py |
|------|------------------------|-----------------|
| **实现方式** | 调用 Shell 脚本 | 直接调用 Python 类 |
| **模块化** | ❌ 否 | ✅ 是 |
| **独立性** | ❌ 步骤耦合 | ✅ 步骤独立 |
| **可调试性** | ⚠️ 中等 | ✅ 高 |
| **灵活性** | ⚠️ 中等 | ✅ 高 |
| **Python 调用** | ❌ 不支持 | ✅ 支持 |
| **代码复用** | ⚠️ 有限 | ✅ 完全复用 |

**推荐**: 使用新的 `run_pipeline.py`，它更灵活、更易维护。

## ✅ 验证

所有步骤都已验证可以正常导入和运行：

```bash
# 测试导入
python -c "from pipeline_steps import Step0Preprocess, Step1Baseline, Step2ItemRepr, Step3Stb, Step4Upstar, Step5Evaluate; print('✓ All steps imported successfully')"

# 测试帮助信息
python -m pipeline_steps.step0_preprocess --help
python -m pipeline_steps.step1_baseline --help
python -m pipeline_steps.step4_upstar --help
```

## 🎓 核心优势

### 1. 模块化
- 每个步骤是独立的类
- 单一职责，易于维护
- 可以单独测试和调试

### 2. 灵活性
- 可以选择运行特定步骤
- 可以从任意步骤恢复
- 支持多种运行方式

### 3. 可复用性
- 步骤类可以在其他脚本中导入使用
- 可以构建自定义流程
- 便于集成到其他系统

### 4. 可扩展性
- 添加新步骤很容易
- 继承 `BaseStep` 即可
- 统一的接口和日志

## 🔍 故障排除

### 导入错误

如果遇到导入错误：
```bash
# 确保在项目根目录
cd d:\jinghan_research\upstar

# 测试导入
python -c "from pipeline_steps import Step0Preprocess; print('OK')"
```

### 步骤失败

如果某个步骤失败：
1. 查看错误日志
2. 检查依赖是否满足
3. 修复问题后从失败步骤重新开始

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

## 📖 相关文档

- **[pipeline_steps/README.md](../pipeline_steps/README.md)** - 详细步骤说明
- **[RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md)** - 完整运行指南
- **[README.md](README.md)** - 项目说明

---

**创建日期**: 2026-03-30
**状态**: ✅ 已完成并测试
**推荐**: ⭐⭐⭐⭐⭐ 强烈推荐使用新的模块化步骤
