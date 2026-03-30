# UPSTAR Smoke Test 指南

## 🎯 目标

在 GPU 上运行完整实验前，先用最小 smoke test 验证主链路没有断。

## 🚀 快速开始

### 方式 1: 自动 Smoke Test（推荐）

```bash
bash scripts/smoke_test.sh
```

**预计时间**: 30-60 分钟

**覆盖范围**:
- ✅ Phase 0: CV10 数据划分
- ✅ Phase 1: Baseline 训练（1 fold, 3 epochs）
- ✅ Phase 2: Item 表示学习（3 epochs）
- ✅ Phase 3: STB 计算
- ✅ Phase 4: UPSTAR 训练（1 fold, 2 epochs, 1 stage）
- ✅ Phase 5: 结果评估

### 方式 2: 手动分步测试

#### Phase 0: 数据预处理（5分钟）
```bash
bash scripts/run_preprocess.sh
```

**验证**: `data/processed/tafeng/metadata.pkl` 存在

#### Phase 1: Baseline 快速测试（10分钟）
```bash
python -m src.training.train_baseline \
    --config configs/tafeng_baseline.yaml \
    --num-folds 1 \
    --quick-test
```

**验证**: `outputs/phase1_baseline/results/cv_results.json` 存在

#### Phase 2: Item 表示快速测试（10分钟）
```bash
# 创建快速测试配置
cat > configs/smoke_item_repr.yaml <<EOF
dataset:
  name: tafeng
  data_dir: data/raw/
  processed_dir: data/processed/tafeng/
  min_item_frequency: 2
  cache_dir: data/cache/tafeng/

model:
  embed_dim: 128
  num_layers: 1
  use_remember_gate: true
  dropout: 0.1

training:
  batch_size_edges: 10000
  learning_rate: 0.001
  weight_decay: 0.00001
  max_epochs: 3
  eval_interval: 1
  num_eval_edges: 1000

item_repr:
  time_threshold: 86400
  aggregation: sum
  use_edge_weight: false
  use_item_features: false

seed: 42
logging:
  log_dir: outputs/smoke_test/phase2_item_repr/logs/
  checkpoint_dir: outputs/smoke_test/phase2_item_repr/checkpoints/
EOF

python src/training/train_item_repr.py --config configs/smoke_item_repr.yaml
```

**验证**: `outputs/phase2_item_repr/checkpoints/item_embeddings.pt` 存在

#### Phase 3: STB 计算（5分钟）
```bash
python src/training/train_stb.py --config configs/stb.yaml
```

**验证**: `outputs/phase3_stb/checkpoints/motivation_labels.npy` 存在

#### Phase 4: UPSTAR 快速测试（20分钟）
```bash
python -m src.training.train_upstar \
    --config configs/tafeng_upstar.yaml \
    --num-folds 1 \
    --quick-test
```

**验证**: `outputs/phase4_upstar/results/cv_results.json` 存在

#### Phase 5: 结果评估（2分钟）
```bash
bash scripts/compare_results.sh
```

**验证**: `outputs/comparison/comparison.txt` 存在

## 📊 Smoke Test 产物路径

### 预期输出文件

```
outputs/
├── smoke_test/
│   ├── phase1_baseline/
│   │   └── results/
│   │       └── cv_results.json           # Baseline 1-fold 结果
│   ├── phase2_item_repr/
│   │   └── checkpoints/
│   │       └── item_embeddings.pt        # Item embeddings
│   ├── phase4_upstar/
│   │   └── results/
│   │       └── cv_results.json           # UPSTAR 1-fold 结果
│   └── comparison/
│       ├── comparison.txt
│       └── improvement_analysis.txt
│
├── phase1_baseline/ (copy for comparison)
│   └── results/
│       └── cv_results.json
│
├── phase4_upstar/ (copy for comparison)
│   └── results/
│       └── cv_results.json
│
├── phase2_item_repr/
│   └── checkpoints/
│       └── item_embeddings.pt
│
└── phase3_stb/
    └── checkpoints/
        ├── stb_scores.npy
        └── motivation_labels.npy
```

### 关键验证点

| 阶段 | 验证文件 | 检查命令 |
|------|----------|----------|
| Phase 0 | `data/processed/tafeng/metadata.pkl` | `ls -lh data/processed/tafeng/` |
| Phase 1 | `outputs/smoke_test/phase1_baseline/results/cv_results.json` | `cat outputs/smoke_test/phase1_baseline/results/cv_results.json` |
| Phase 2 | `outputs/phase2_item_repr/checkpoints/item_embeddings.pt` | `ls -lh outputs/phase2_item_repr/checkpoints/` |
| Phase 3 | `outputs/phase3_stb/checkpoints/motivation_labels.npy` | `ls -lh outputs/phase3_stb/checkpoints/` |
| Phase 4 | `outputs/smoke_test/phase4_upstar/results/cv_results.json` | `cat outputs/smoke_test/phase4_upstar/results/cv_results.json` |
| Phase 5 | `outputs/smoke_test/comparison/comparison.txt` | `cat outputs/smoke_test/comparison/comparison.txt` |

## 🔧 Quick Test 参数说明

### Baseline 训练

```bash
python -m src.training.train_baseline \
    --config configs/tafeng_baseline.yaml \
    --num-folds 1 \              # 只训练 1 fold（默认 10）
    --quick-test                 # 快速测试模式（3 epochs，默认 20）
```

### UPSTAR 训练

```bash
python -m src.training.train_upstar \
    --config configs/tafeng_upstar.yaml \
    --num-folds 1 \              # 只训练 1 fold（默认 10）
    --quick-test                 # 快速测试模式（2 epochs/stage，默认 10）
    --stage 1                    # 可选：只训练特定阶段
```

### Quick Test 模式效果

| 参数 | 正式模式 | Quick Test 模式 |
|------|----------|-----------------|
| **num_folds** | 10 | 1 |
| **max_epochs** | 20 | 3 |
| **max_epochs_per_stage** | 10 | 2 |
| **early_stop_patience** | 5 | 2 |
| **预计时间** | 20-40 小时 | 30-60 分钟 |

## ⚠️ 重要提示

### 1. Smoke Test 不是完整实验

Smoke test 只验证：
- ✅ 代码可以运行
- ✅ 数据格式正确
- ✅ 模型可以训练
- ✅ 结果可以保存

Smoke test 不保证：
- ❌ 模型收敛（epochs 太少）
- ❌ 性能达到论文水平（数据太少）
- ❌ 结果具有统计意义（folds 太少）

### 2. 清理 Smoke Test 产物

```bash
# 清理所有 smoke test 输出
rm -rf outputs/smoke_test/
rm -f configs/smoke_item_repr.yaml

# 保留有用的产物（可选）
# outputs/phase2_item_repr/checkpoints/item_embeddings.pt
# outputs/phase3_stb/checkpoints/*
```

### 3. 从 Smoke Test 到正式实验

Smoke test 通过后，可以运行正式实验：

```bash
# 完整实验
python run_full_experiment.py

# 或分步运行
bash scripts/run_tafeng_baseline.sh    # 20-40 小时
bash scripts/run_tafeng_upstar.sh      # 80-160 小时
bash scripts/compare_results.sh
```

## 🐛 常见问题

### Q1: Smoke test 失败怎么办？

**A**: 检查具体哪个阶段失败：

```bash
# 检查数据预处理
ls -lh data/processed/tafeng/

# 检查 item embeddings
ls -lh outputs/phase2_item_repr/checkpoints/

# 检查 STB 结果
ls -lh outputs/phase3_stb/checkpoints/

# 查看错误日志
tail -100 outputs/smoke_test/phase1_baseline/logs/cv.log
tail -100 outputs/smoke_test/phase4_upstar/logs/cv.log
```

### Q2: 可以在 CPU 上运行 smoke test 吗？

**A**: 可以，但会很慢（可能几小时）。建议使用 GPU：

```bash
# 检查 GPU 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 如果 GPU 不可用，确保安装了 GPU 版本的 PyTorch
```

### Q3: Smoke test 需要多少磁盘空间？

**A**: 大约 5-10 GB：

```
data/processed/tafeng/       ~1-2 GB
outputs/phase2_item_repr/    ~500 MB
outputs/phase3_stb/          ~100 MB
outputs/smoke_test/          ~2-5 GB
```

### Q4: 可以中断 smoke test 吗？

**A**: 可以，但下次运行需要从头开始：

```bash
# Ctrl+C 中断
# 清理后重新运行
rm -rf outputs/smoke_test/
bash scripts/smoke_test.sh
```

## ✅ Smoke Test 通过标准

Smoke test 成功的标准：

- [ ] 所有阶段都运行完成
- [ ] 所有关键输出文件都存在
- [ ] 没有异常或错误
- [ ] 结果文件格式正确（可以打开和读取）
- [ ] 日志显示训练过程正常

**如果以上都满足，就可以放心运行正式实验了！**

---

**Smoke Test 完成时间**: 30-60 分钟
**验证范围**: 完整训练主链路
**下一步**: 运行正式 10-fold CV 实验
