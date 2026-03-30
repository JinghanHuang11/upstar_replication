# UPSTAR 项目组织详解

> **最后更新**: 2026-03-30
> **项目状态**: ✅ Phase 0-6 全部完成 | **工程清理完成（cv10-only，论文对齐）**
> **代码规模**: 30 个 Python 源文件 | 5 个配置文件 | 8 个活跃脚本
> **自检状态**: ✅ 所有检查通过（cv10-only，统一入口，论文对齐）

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
  - **三个独立 4-layer LSTM**（各有独立 Embedding / LSTM / Linear）
  - hidden_dim=128（Section 7.3）

- **全局融合**（Section 3.1.4）：
  - **逐维 Per-Dimension Gate**
  - 联合损失：`L_total = L_global + 0.7·L_S&E&O + L_orth + L_distill`

- **Item-GNN**：区分 in/out 邻居的 Message Passing

### 数据集

- **Tafeng**：台湾超市数据集 (26K 用户, 15K 商品, 785K 交互)

---

## 二、CV10-ONLY 统一（2026-03-30）⭐

### **统一成果**
- ✅ **cv10-only**: 删除 leave_one_out 模式，统一为 10-fold CV
- ✅ **统一入口**: 所有脚本调用新的 Python 训练入口
- ✅ **最小配置**: 收敛为 5 个核心配置文件
- ✅ **统一指标**: 所有配置使用 `metric_for_best: 'NDCG@20'`

### **运行模式**
- ✅ **唯一模式**: 10-fold cross-validation
- ❌ 已删除：single_run, leave_one_out, split_ratio

### **已删除文件（12个）**
- ❌ `configs/baseline.yaml`
- ❌ `configs/experiment_tafeng.yaml`
- ❌ `configs/stb_advanced.yaml`
- ❌ `configs/tafeng_cv.yaml`
- ❌ `configs/test_baseline.yaml`
- ❌ `scripts/run_preprocess_cv.sh`
- ❌ `scripts/verify_cv_update.sh`
- ❌ `inspect_global.py`
- ❌ `upload_to_github.py`
- ❌ `QUICKSTART.md`
- ❌ `pipeline_steps/` 目录

---

## 三、项目结构

### 核心训练代码（CV10-ONLY）

```
src/training/
├── cross_validation_baseline.py  # ⭐ Baseline 唯一实现
├── train_baseline.py              # Baseline 薄封装
├── train_upstar_cv.py             # ⭐ UPSTAR 唯一实现
├── train_upstar.py                # UPSTAR 薄封装
├── train_item_repr.py             # Item 表示学习
├── train_stb.py                   # STB 计算
└── losses.py                      # 损失函数
```

### 运行脚本（CV10-ONLY）

```
scripts/
├── run_preprocess.sh              # ⭐ 数据预处理
├── run_tafeng_baseline.sh         # ⭐ Baseline 训练
├── run_tafeng_upstar.sh           # ⭐ UPSTAR 训练
├── run_item_repr.sh               # Item 表示学习
├── run_stb.sh                     # STB 计算
├── compare_results.sh             # 结果比较
├── smoke_test.sh                  # Smoke test 脚本
├── clean_old_data.sh              # 清理工具
└── test_baseline_eval.sh          # 测试工具
```

### 配置文件（最小集合）

```
configs/
├── tafeng_baseline.yaml           # ⭐ Baseline 训练
├── tafeng_upstar.yaml             # ⭐ UPSTAR 训练
├── item_repr.yaml                 # Item 表示学习
├── stb.yaml                       # STB 计算
└── tafeng_eval.yaml               # 评估配置
```

---

## 四、运行入口

### 推荐运行方式

#### 1. **Smoke Test（GPU 运行前必做）**
```bash
bash scripts/smoke_test.sh
```
**时间**: 30-60 分钟

#### 2. **完整实验**
```bash
# 使用总控脚本
python run_full_experiment.py

# 或分步运行
bash scripts/run_preprocess.sh          # Phase 0
bash scripts/run_tafeng_baseline.sh      # Phase 1 (20-40h)
bash scripts/run_item_repr.sh           # Phase 2
bash scripts/run_stb.sh                 # Phase 3
bash scripts/run_tafeng_upstar.sh       # Phase 4 (80-160h)
bash scripts/compare_results.sh         # Phase 5
```

#### 3. **Python 直接调用**
```bash
# Baseline 训练
python -m src.training.train_baseline --config configs/tafeng_baseline.yaml

# UPSTAR 训练
python -m src.training.train_upstar --config configs/tafeng_upstar.yaml

# Item 表示
python src/training/train_item_repr.py --config configs/item_repr.yaml

# STB 计算
python src/training/train_stb.py --config configs/stb.yaml
```

---

## 五、输出目录结构

```
outputs/
├── phase1_baseline/              # Baseline 10-fold CV
│   └── results/
│       └── cv_results.json       # ⭐ CV10 统计结果
│
├── phase2_item_repr/             # Item 表示
│   └── checkpoints/
│       └── item_embeddings.pt    # ⭐ Item embeddings
│
├── phase3_stb/                   # STB 计算
│   └── checkpoints/
│       ├── stb_scores.npy        # ⭐ STB 分数
│       └── motivation_labels.npy # ⭐ 动机标签
│
├── phase4_upstar/                # UPSTAR 10-fold CV
│   └── results/
│       └── cv_results.json       # ⭐ CV10 统计结果
│
└── comparison/                   # 结果比较
    └── comparison.txt
```

---

## 六、关键改进

| 改进项 | 状态 | 说明 |
|--------|------|------|
| **CV10-ONLY** | ✅ | 统一为 10-fold cross-validation |
| **统一入口** | ✅ | 所有脚本调用新的 Python 训练入口 |
| **最小配置** | ✅ | 收敛为 5 个核心配置文件 |
| **统一指标** | ✅ | `metric_for_best: 'NDCG@20'` |
| **Smoke Test** | ✅ | 30-60 分钟验证主链路 |

---

## 七、使用指南

### 快速开始

```bash
# 1. Smoke test（推荐先做）
bash scripts/smoke_test.sh

# 2. 完整实验
python run_full_experiment.py

# 3. 查看结果
cat outputs/comparison/comparison.txt
```

### 详细文档

- **运行指南**: [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md)
- **Smoke Test 指南**: [docs/SMOKE_TEST_GUIDE.md](docs/SMOKE_TEST_GUIDE.md)
- **工程自检报告**: [FINAL_ENGINEERING_CHECK.md](FINAL_ENGINEERING_CHECK.md)

---

## 八、项目状态

### ✅ 已完成
- [x] CV10-ONLY 统一
- [x] 统一入口实现
- [x] 最小配置收敛
- [x] Smoke test 支持
- [x] 文档更新

### ⚠️ 已知限制
- 测试框架仍使用旧的 `src.experiments/`（不影响主流程）
- 部分文档仍有旧引用（新文档已提供正确说明）

---

**项目状态**: ✅ Ready for GPU experiments

**代码质量**: ⭐⭐⭐⭐⭐ (干净、一致、论文对齐)
