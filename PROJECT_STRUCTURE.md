# 项目文件组织结构

> **最后更新**: 2026-03-28
> **代码规模**: 40+ 个 Python 源文件 | 10+ 个配置文件 | 9+ 个脚本
> **论文对齐**: 已完成 Phase 0/1/4/5 论文对齐改进（CV10 + Baseline + FusionGate + 联合损失 + 主指标口径）
> **代码状态**: Phase 0/1 自检通过，Phase 4/5 自检通过，待端到端训练验证

---

## 目录结构

```
upstar/
├── configs/              # 配置文件 (10+个) - 已与论文 Section 7 对齐
│   ├── baseline.yaml              # Baseline 测试配置
│   ├── tafeng_baseline.yaml       # Tafeng Baseline 配置（✓ Phase 1：hidden=128, layers=4）
│   ├── tafeng_cv.yaml             # Tafeng CV10 配置（✓ Phase 0：10-fold CV）
│   ├── tafeng_upstar.yaml         # Tafeng UPSTAR 配置（✓ 论文对齐：lr=3e-4, λ=0.7）
│   ├── tafeng_eval.yaml           # Tafeng 评估配置（✓ 已更新：主指标 P@5/20, NDCG@5/20, MRR@5/20）
│   ├── test_baseline.yaml         # Baseline 测试
│   ├── item_repr.yaml             # Item 表示学习配置（✓ 论文对齐）
│   ├── stb.yaml                   # STB V1 配置（✓ Phase 3 改进）
│   ├── stb_advanced.yaml          # 高级 STB 配置
│   └── upstar_train.yaml           # UPSTAR 训练配置

├── data/                 # 数据目录
│   ├── raw/
│   │   ├── ta_feng.csv            # 原始 Tafeng 数据 (817K 条)
│   │   └── baseline/              # Baseline 测试数据
│   ├── processed/
│   │   ├── baseline/              # Baseline 处理结果
│   │   └── tafeng/                # Tafeng 处理结果（✓ 新格式：含 timestamp）
│   │       ├── metadata.pkl       # 词表 (26K 用户, 15K 商品)
│   │       ├── train_sequences.pkl # [(item, timestamp), ...]
│   │       ├── val_sequences.pkl
│   │       └── test_sequences.pkl
│   └── cache/
│       └── tafeng/                # 图缓存

├── docs/                 # 详细文档 (30+个)
│   ├── phase0/           # Phase 0 文档 (4个) ✨ 新增
│   │   ├── phase0_split_modes.md      # 划分模式说明
│   │   ├── phase0_sequence_format.md  # 序列格式说明
│   │   ├── phase0_metadata_format.md  # Metadata 格式说明
│   │   └── phase0_self_check.md       # Phase 0 自检报告
│   ├── phase1/           # Phase 1 文档 (4个) ✨ 更新
│   │   ├── phase1_baseline_design.md  # Baseline 设计说明
│   │   ├── phase1_training_modes.md   # 训练模式说明
│   │   ├── phase1_evaluation_metrics.md # 评估指标说明
│   │   └── phase1_self_check.md       # Phase 1 自检报告
│   ├── phase2/           # Phase 2 文档 (2个)
│   ├── phase3/           # Phase 3 文档 (2个)
│   ├── phase4/           # Phase 4 文档 (1个)
│   ├── phase5/           # Phase 5 文档 (2个)
│   ├── phase6/           # Phase 6 文档 (2个)
│   ├── evaluation/       # 评估文档 (1个)
│   ├── experiments/       # 实验文档 (1个)
│   ├── reference/        # 参考文档 (1个)
│   │   └── upstar_rec.md # 论文精读笔记
│   └── FULL_EXPERIMENT_README.md

├── scripts/              # 运行脚本 (9+个)
│   ├── run_preprocess.sh          # 数据预处理（leave_one_out）
│   ├── run_preprocess_cv.sh       # 数据预处理（cv10）✨ 新增
│   ├── run_tafeng_baseline.sh     # Baseline 训练（✓ Phase 1：支持双模式）
│   ├── run_item_repr.sh           # Item 表示学习
│   ├── run_stb.sh                 # STB 计算
│   ├── run_stb_advanced.sh        # 高级 STB
│   ├── run_tafeng_upstar.sh       # UPSTAR 训练
│   ├── run_eval.sh                # 模型评估（✓ 已更新：主指标口径）
│   └── compare_results.sh         # 结果对比（✓ 已更新：Precision@K 主表）

├── src/                  # 源代码 (40+个 Python 文件)
│   ├── data/             # 数据处理 (5个) ✨ Phase 0 改进
│   ├── graphs/           # 图构建 (4个) - Phase 3 改进
│   ├── models/           # 模型 (8个) - Phase 4 改进
│   ├── training/         # 训练 (10+个) - Phase 1/4 改进 ✨
│   ├── evaluation/       # 评估 (7个) - Phase 5 改进
│   ├── experiments/      # 实验管理 (4个)
│   └── utils/            # 工具 (4个)

├── outputs/              # 输出目录
│   ├── baselines/               # Baseline 实验
│   ├── phase4_upstar/           # UPSTAR 实验（✓ 新结构）
│   ├── checkpoints/             # 模型检查点
│   ├── logs/                   # 日志
│   ├── predictions/             # 预测结果
│   └── results/                # 评估结果（✓ 结构化：main + additional）

├── tests/                # 测试脚本
├── QUICKSTART.md         # 快速开始
├── PROJECT_STRUCTURE.md  # 本文件
├── PROJECT_ORGANIZATION.md
├── README.md             # 项目说明
├── FULL_EXPERIMENT_README.md
├── demo.py               # 环境验证脚本
├── run_full_experiment.py
└── requirements.txt
```

---

## 源代码清单 (src/)

### src/models/ — 模型定义（Phase 1/4 改进）

| 文件 | 功能 | Phase 状态 |
|------|------|-------------|
| `baseline_lstm.py` | LSTM Baseline（✓ Phase 1：hidden=128, layers=4） | ✅ Phase 1 |
| `item_gnn.py` | Item-GNN（in/out 邻居分离） | ✅ Phase 3 |
| `stb_encoder.py` | STB 编码器 | ✅ Phase 3 |
| `stb_encoder_advanced.py` | 进阶 STB 编码器 | ✅ Phase 3 |
| `sequence_models.py` | S/E/O 三分支 LSTM | ✅ Phase 4 |
| `fusion.py` | **FusionGate 逐维 gate** | ✅ **Phase 4 核心改进** |
| `upstar.py` | UPSTAR 主模型 | ✅ Phase 4 |
| `load_item_embeddings.py` | Embedding 加载工具 | - |

**Phase 4 核心改进**（`fusion.py`）：
- Gate shape：`[B, 3]` → `[B, 128, 3]`（逐维 softmax）
- `y_hat_global = Linear(z_global)`（不是加权 logit 平均）

---

### src/training/ — 训练脚本（Phase 1/4 改进）

| 文件 | 功能 | Phase 状态 |
|------|------|-------------|
| `train_baseline.py` | LSTM Baseline 训练（✓ Phase 1：支持 leave_one_out + cv10） | ✅ **Phase 1 核心改进** |
| `cross_validation_baseline.py` | Baseline 10 折交叉验证 | ✅ Phase 1 |
| `train_item_repr.py` | Item-GNN 训练 | - |
| `train_stb.py` | STB V1 计算 | ✅ Phase 3 |
| `train_stb_advanced.py` | 进阶 STB 计算 | - |
| `train_upstar_stage1.py` | Stage 1 训练 | ✅ Bug 修复 |
| `train_upstar.py` | **UPSTAR 完整 4 阶段训练** | ✅ **Phase 4 核心改进** |
| `losses.py` | **联合损失（global + branch + ortho + distill）** | ✅ **Phase 4 核心改进** |

**Phase 4 核心改进**（`losses.py`）：
- `L_orth = τ_s·z_oᵀz_s + τ_e·z_oᵀz_expl`（无 normalize，无 squared）
- 完整条件双师生蒸馏（stable→S→E, exploratory→E→S, uncategorized→无蒸馏）
- `total_loss = λg·Lg + λb·Lb + λo·Lo + λd·Ld`

---

### src/evaluation/ — 评估（Phase 5 改进）

| 文件 | 功能 | Phase 5 状态 |
|------|------|-------------|
| `metrics.py` | Precision@K, NDCG@K, MRR@K | - |
| `evaluator.py` | **UPSTAR 统一评估器** | ✅ **Phase 5 核心改进** |
| `evaluate_baseline.py` | Baseline 评估 | - |
| `cross_validation.py` | UPSTAR 交叉验证（Bug 修复） | ✅ Bug 修复 |
| `report.py` | **报告生成（主/附加分离）** | ✅ **Phase 5 核心改进** |
| `comparison.py` | **Baseline vs UPSTAR 对比** | ✅ **Phase 5 核心改进** |
| `evaluate.py` | 评估主程序（k_values 配置驱动） | ✅ Phase 5 |

**Phase 5 核心改进**：
- `evaluator.py`：`PAPER_MAIN_METRICS` 常量（`P@5/20, NDCG@5/20, MRR@5/20`）
- `evaluator.py`：返回值结构化为 `{main_metrics, additional_metrics, _flat}`
- `report.py`：主表 + 附加指标分段输出（JSON/CSV/TXT/LaTeX/Markdown）
- `comparison.py`：`generate_executive_summary` 移除硬编码 `NDCG@10`/`Recall@20`
- `evaluate.py`：`k_values` 从 `config['evaluation']['k_values']` 读取

---

### src/experiments/ — 实验管理

| 文件 | 功能 |
|------|------|
| `phase_base.py` | 实验阶段基类 |
| `phases.py` | 6 个 Phase 实现（Phase 0-5） |
| `experiment_manager.py` | 完整流水线编排 |
| `report_generator.py` | Markdown/HTML 综合报告生成 |

---

### src/utils/ — 工具

| 文件 | 功能 |
|------|------|
| `seed.py` | 随机种子设置 |
| `io.py` | pickle/torch 文件读写 |
| `logger.py` | 日志系统 |
| `registry.py` | 模型/组件注册器 |

---

## 配置文件清单 (configs/)

| 文件 | 用途 | 对齐状态 |
|------|------|----------|
| `baseline.yaml` | Baseline 测试配置 | - |
| `tafeng_baseline.yaml` | Tafeng Baseline 训练 | **✓ Phase 1**：hidden=128, layers=4, split_method=leave_one_out |
| `tafeng_cv.yaml` | Tafeng CV10 配置 | **✓ Phase 0**：split_method=cv10, num_folds=10 |
| `tafeng_upstar.yaml` | **Tafeng UPSTAR 训练** | **✓ 论文对齐**：lr=3e-4, batch=64, hidden=128, layers=4, λ=0.7, τ_s=τ_e=0.5 |
| `tafeng_eval.yaml` | **Tafeng 评估** | **✓ 已更新**：主指标 P@5/20, NDCG@5/20, MRR@5/20 |
| `test_baseline.yaml` | Baseline 功能测试 | - |
| `item_repr.yaml` | Item 表示学习 | ✓ 论文对齐 |
| `stb.yaml` | STB V1 计算 | ✓ Phase 3 |
| `stb_advanced.yaml` | 高级 STB | ✓ Phase 3 |
| `upstar_train.yaml` | UPSTAR 完整训练 | - |

---

## Phase 4/5 论文对齐改进总结

### Phase 4 — 模型与训练

| # | 模块 | 对应文件 | 改进内容 |
|---|------|----------|----------|
| 1 | **逐维 FusionGate** | `src/models/fusion.py` | 标量 `[B,3]` → 逐维 `[B,128,3]`；softmax dim=2 |
| 2 | **y_hat_global** | `src/models/fusion.py` | 移除加权 logit 平均；`Linear(z_global)` |
| 3 | **L_orth** | `src/training/losses.py` | 去掉 `F.normalize`；论文 raw dot product `τ_s·z_oᵀz_s + τ_e·z_oᵀz_e` |
| 4 | **L_distill** | `src/training/losses.py` | 完整条件双师生：stable→S→E, exploratory→E→S, uncategorized→无蒸馏；teacher `.detach()` |
| 5 | **联合损失** | `src/training/losses.py` | `total_loss = λg·Lg + λb·Lb + λo·Lo + λd·Ld` |
| 6 | **超参数** | `configs/tafeng_upstar.yaml` | lr=3e-4, batch=64, Adam, λ=0.7, τ_s=τ_e=0.5 |
| 7 | **Bug 修复** | `train_upstar.py` | `loss_loss_stab` 键名 bug；Gate 日志 `[B,128]` 均值 |
| 8 | **拼写修复** | `evaluator/cross_validation/train_upstar_stage1` | `UPSTARMoodel` → `UPSTARModel` |

### Phase 5 — 评估

| # | 模块 | 对应文件 | 改进内容 |
|---|------|----------|----------|
| 9 | **主指标** | `evaluator.py` | `PAPER_MAIN_METRICS = [P@5, P@20, NDCG@5, NDCG@20, MRR@5, MRR@20]` |
| 10 | **输出结构** | `evaluator.py` | 返回 `{main_metrics, additional_metrics, _flat}`；JSON/CSV 分段 |
| 11 | **报告生成** | `report.py` | 主表 + 附加分段输出；支持结构化/扁平输入自动检测 |
| 12 | **对比报告** | `comparison.py` | `generate_executive_summary` 移除硬编码 NDCG@10/Recall@20 |
| 13 | **配置驱动** | `evaluate.py` | `k_values` 从 config 读取而非硬编码 |

---

## 快速使用

### 环境验证

```bash
conda create -n upstar python=3.10 -y
conda activate upstar
pip install torch torchvision torchaudio
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install -r requirements.txt

python -c "
from src.training.losses import UPSTARLoss, create_loss_from_config
from src.evaluation.metrics import compute_all_metrics
from src.models.fusion import FusionGate
from src.models.upstar import UPSTARModel
print('All Phase 4/5 modules verified!')
"
```

### 模块自检

```bash
python -m src.training.losses      # Stage 1-4 损失测试 ✅
python -m src.models.upstar        # 逐维 gate 测试 ✅
python -m src.evaluation.evaluator  # 主/附加指标分离测试 ✅
python -m src.evaluation.report    # 报告生成测试 ✅
python -m src.evaluation.comparison # 对比表测试 ✅
```

### 完整实验

```bash
# 完整运行（需重新训练）
python run_full_experiment.py --force-rerun --mode full

# 分步运行
bash scripts/run_preprocess.sh
bash scripts/run_item_repr.sh
bash scripts/run_stb.sh
bash scripts/run_tafeng_upstar.sh
bash scripts/run_eval.sh
```

---

## 论文对齐状态

| Phase | 模块 | 论文章节 | 状态 |
|-------|------|----------|------|
| Phase 3 | Item-Time Graph (day-level) | Section 7.1 | ✅ |
| Phase 3 | Item-GNN (in/out 邻居) | Section 3.2 | ✅ |
| Phase 3 | STB MINE 估计 | Section 3.1.3 | ✅ |
| Phase 3 | STB worst-case (min) aggregation | Section 3.1.3 | ✅ |
| Phase 3 | MI 梯度真正可微分 | Section 3.1.3 | ✅ |
| Phase 3 | Bipartite 约束 | Section 3.1.3 | ✅ |
| Phase 3 | 分离 Graph Summary | Section 3.1.3 | ✅ |
| Phase 3 | 拓扑扰动 bipartite edges | Section 3.1.3 | ✅ |
| Phase 3 | Config 驱动分类阈值 | Section 3.1.4 | ✅ |
| Phase 4 | 逐维 FusionGate | Section 3.1.4 | ✅ |
| Phase 4 | y_hat_global = Linear(z_global) | Section 3.1.4 | ✅ |
| Phase 4 | L_orth raw dot product | Section 3.3 | ✅ |
| Phase 4 | L_distill 条件双师生 | Section 3.3 | ✅ |
| Phase 4 | 联合损失公式 | Section 3.3 | ✅ |
| Phase 4 | 独立三分支 LSTM | Section 7.3 | ✅ |
| Phase 4 | uncategorized 保留 O-model | Section 3.1.4 | ✅ |
| Phase 4 | 超参数对齐 | Section 7.3/7.4 | ✅ |
| Phase 5 | 主指标口径 | Section 7.3 | ✅ |
| Phase 5 | Precision@K 命名（非 HR@K） | Section 7.3 | ✅ |
| Phase 5 | 主/附加指标分离输出 | 工程实现 | ✅ |

---

## 仍存在的工程差距

| 项目 | 状态 | 说明 |
|------|------|------|
| 四阶段课程 | 工程实现 | 论文直接用联合损失；课程是稳定性工程技巧 |
| L_orth batch 平均方式 | 自由度 | 论文未明确；对 batch 内所有样本求均值 |
| Early stop 指标 | NDCG@10 | 论文主表用 NDCG@20；工程近似 |
| 端到端训练验证 | **待完成** | 各模块自检通过，尚未运行完整 4-stage 训练 |

---

---

## Phase 0/1 改进总结（2026-03-28）

### Phase 0 — 数据预处理

| # | 模块 | 对应文件 | 改进内容 |
|---|------|----------|----------|
| 1 | **10-Fold CV** | `src/data/build_sequences.py` | 实现 10 折交叉验证划分（论文对齐） |
| 2 | **Timestamp 保留** | `src/data/build_sequences.py` | 序列格式 `[(item, timestamp), ...]` |
| 3 | **双模式支持** | `src/data/build_sequences.py` | leave_one_out + cv10 配置驱动 |
| 4 | **Metadata 增强** | `src/data/preprocess.py` | 添加统计信息（avg/min/max/std） |
| 5 | **向后兼容** | `src/data/build_sequences.py` | 支持旧格式 `[item, ...]` 读取 |

### Phase 1 — Baseline 训练

| # | 模块 | 对应文件 | 改进内容 |
|---|------|----------|----------|
| 6 | **Baseline LSTM** | `src/models/baseline_lstm.py` | num_layers: 2→4（论文对齐） |
| 7 | **双模式训练** | `src/training/train_baseline.py` | leave_one_out + cv10 统一入口 |
| 8 | **论文对齐指标** | `src/evaluation/metrics.py` | P@5/20, NDCG@5/20, MRR@5/20 |
| 9 | **训练脚本统一** | `src/training/train_baseline.py` | 自动检测 split_method，路由到对应训练流程 |
| 10 | **结果格式一致** | `src/training/train_baseline.py` | test_results.pkl / cv_results.pkl |

### 新增文档

**Phase 0**：
- `docs/phase0_split_modes.md` — 划分模式详细说明
- `docs/phase0_sequence_format.md` — 序列格式规范
- `docs/phase0_metadata_format.md` — Metadata v1.1 格式
- `docs/phase0_self_check.md` — Phase 0 自检报告

**Phase 1**：
- `docs/phase1_baseline_design.md` — Baseline 设计说明
- `docs/phase1_training_modes.md` — 训练模式说明
- `docs/phase1_evaluation_metrics.md` — 评估指标说明
- `docs/phase1_self_check.md` — Phase 1 自检报告

---

**最后更新**: 2026-03-28
