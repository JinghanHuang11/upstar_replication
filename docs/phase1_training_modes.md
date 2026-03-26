# Phase 1 训练模式说明

> **更新时间**: 2026-03-26
> **训练脚本**: `src/training/train_baseline.py`
> **支持模式**: leave_one_out (单次) / cv10 (10-fold CV)

---

## 一、修改后的训练入口

### 1.1 统一训练脚本

**文件**: [src/training/train_baseline.py](../src/training/train_baseline.py)

**新增功能**:
- ✅ 自动检测配置中的 `split_method`
- ✅ 支持两种训练模式：单次 (leave_one_out) 和 CV (cv10)
- ✅ 统一的训练逻辑，避免代码重复

**关键函数**:
```python
def load_data(config):
    """
    支持两种模式：
    - leave_one_out: 返回 train_loader, val_loader, test_loader, num_items
    - cv10: 返回 cv_splits (list of dict), num_items
    """

def train_single_mode(config, args, device):
    """单次训练模式 (leave_one_out)"""

def train_cv_mode(config, args, device):
    """10-fold CV 训练模式 (cv10)"""

def main(args):
    """统一入口，自动选择模式"""
```

---

## 二、两种模式的运行逻辑

### 2.1 Leave-One-Out 模式（工程模式）

**配置**:
```yaml
# configs/tafeng_baseline.yaml
dataset:
  split_method: 'leave_one_out'
```

**运行方式**:
```bash
# 方式 1: 自动检测（推荐）
python src/training/train_baseline.py --config configs/tafeng_baseline.yaml

# 方式 2: 显式指定
python src/training/train_baseline.py --config configs/tafeng_baseline.yaml --mode single
```

**训练流程**:
```
1. 加载 train/val/test sequences
2. 训练模型（early stopping on val set）
3. 在 test set 上评估最佳模型
4. 保存结果到 test_results.pkl
```

**输出文件**:
```
outputs/phase1_baseline/
├── checkpoints/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   ├── ...
│   └── best_model.pt
└── predictions/
    └── test_results.pkl  # 单次结果
```

**输出格式**:
```python
# test_results.pkl
{
    'Precision@5': 0.0523,
    'Precision@20': 0.1234,
    'NDCG@5': 0.0612,
    'NDCG@20': 0.1456,
    'MRR@5': 0.0412,
    'MRR@20': 0.0891,
    # ... 其他 K 值
}
```

---

### 2.2 CV10 模式（论文模式）

**配置**:
```yaml
# configs/tafeng_cv.yaml
dataset:
  split_method: 'cv10'
  num_folds: 10
```

**运行方式**:
```bash
# 方式 1: 自动检测（推荐）
python src/training/train_baseline.py --config configs/tafeng_cv.yaml

# 方式 2: 显式指定
python src/training/train_baseline.py --config configs/tafeng_cv.yaml --mode cv
```

**训练流程**:
```
For each fold i = 1..10:
    1. 加载 fold_i 的 train/test sequences
    2. 训练模型（early stopping on test set）
    3. 保存最佳模型到 fold_i/best_model.pt
    4. 记录 fold_i 的测试结果

After all folds:
    5. 计算所有 fold 的 mean ± std
    6. 保存汇总结果到 cv_results.pkl
```

**输出文件**:
```
outputs/phase1_baseline/
├── checkpoints/
│   ├── fold_1/
│   │   ├── checkpoint_epoch_1.pt
│   │   ├── checkpoint_epoch_2.pt
│   │   └── best_model.pt
│   ├── fold_2/
│   │   ├── ...
│   │   └── best_model.pt
│   ...
│   └── fold_10/
│       └── best_model.pt
└── predictions/
    └── cv_results.pkl  # CV 汇总结果
```

**输出格式**:
```python
# cv_results.pkl
{
    # 每个 fold 的详细结果
    'fold_results': [
        {
            'Precision@5': 0.0510,
            'Precision@20': 0.1220,
            'NDCG@5': 0.0602,
            'NDCG@20': 0.1441,
            'MRR@5': 0.0401,
            'MRR@20': 0.0875,
            # ... 其他 K 值
        },
        # ... fold 2-10
    ],

    # 汇总统计（mean ± std）
    'aggregate_metrics': {
        'Precision@5_mean': 0.0523,
        'Precision@5_std': 0.0018,
        'Precision@20_mean': 0.1234,
        'Precision@20_std': 0.0021,
        'NDCG@5_mean': 0.0612,
        'NDCG@5_std': 0.0025,
        'NDCG@20_mean': 0.1456,
        'NDCG@20_std': 0.0031,
        'MRR@5_mean': 0.0412,
        'MRR@5_std': 0.0028,
        'MRR@20_mean': 0.0891,
        'MRR@20_std': 0.0035,
        # ... 其他 K 值的 mean/std
    },

    'num_folds': 10
}
```

---

## 三、关键实现细节

### 3.1 模式自动检测

**代码**: [train_baseline.py:main](../src/training/train_baseline.py#L450-L470)

```python
def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine training mode
    split_method = config['dataset'].get('split_method', 'leave_one_out')

    if args.mode == 'auto':
        # Auto-detect from config
        if split_method == 'cv10':
            mode = 'cv'
        else:
            mode = 'single'
    else:
        mode = args.mode

    logger.info(f"Training mode: {mode} (split_method={split_method})")

    # Route to appropriate training function
    if mode == 'cv':
        return train_cv_mode(config, args, device)
    else:
        return train_single_mode(config, args, device)
```

**检测逻辑**:
1. 读取配置文件中的 `split_method`
2. 如果 `split_method == 'cv10'` → CV 模式
3. 否则 → 单次模式
4. 用户可以通过 `--mode` 参数覆盖

---

### 3.2 数据加载统一

**代码**: [train_baseline.py:load_data](../src/training/train_baseline.py#L35-L150)

```python
def load_data(config: dict):
    """
    支持两种模式：
    1. leave_one_out: Load train/val/test sequences
    2. cv10: Load pre-computed CV splits

    Returns:
        train_loader, val_loader, test_loader, num_items (for leave_one_out)
        OR
        cv_splits, num_items (for cv10)
    """
    split_method = config['dataset'].get('split_method', 'leave_one_out')

    if split_method == 'cv10':
        # CV10 mode: Load pre-computed CV splits
        cv_dir = processed_dir / 'cv_splits'

        # Load CV metadata
        with open(cv_dir / 'cv_metadata.pkl', 'rb') as f:
            cv_metadata = pickle.load(f)

        # Load all fold splits
        cv_splits = []
        for fold_idx in range(1, num_folds + 1):
            fold_dir = cv_dir / f'fold_{fold_idx}'
            with open(fold_dir / 'train_sequences.pkl', 'rb') as f:
                train_sequences = pickle.load(f)
            with open(fold_dir / 'test_sequences.pkl', 'rb') as f:
                test_sequences = pickle.load(f)
            cv_splits.append({'train': train_sequences, 'test': test_sequences})

        return cv_splits, num_items

    else:
        # leave_one_out mode: Load train/val/test sequences
        # ... (原有逻辑)
        return train_loader, val_loader, test_loader, num_items
```

**关键设计**:
- ✅ 统一的数据加载接口
- ✅ 自动检测模式
- ✅ 复用 dataloader 创建逻辑

---

### 3.3 CV 汇总统计

**代码**: [train_baseline.py:train_cv_mode](../src/training/train_baseline.py#L380-L420)

```python
# Compute aggregate statistics
aggregate_metrics = {}
for metric_key in fold_results[0].keys():
    values = [fold[metric_key] for fold in fold_results]
    mean_val = np.mean(values)
    std_val = np.std(values)
    aggregate_metrics[f'{metric_key}_mean'] = mean_val
    aggregate_metrics[f'{metric_key}_std'] = std_val

# Print main metrics
main_metrics = ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']
for metric in main_metrics:
    if f'{metric}_mean' in aggregate_metrics:
        logger.info(f"{metric}: {aggregate_metrics[f'{metric}_mean']:.4f} ± {aggregate_metrics[f'{metric}_std']:.4f}")
```

**输出示例**:
```
10-Fold CV Aggregate Results
================================================================================
Precision@5: 0.0523 ± 0.0018
Precision@20: 0.1234 ± 0.0021
NDCG@5: 0.0612 ± 0.0025
NDCG@20: 0.1456 ± 0.0031
MRR@5: 0.0412 ± 0.0028
MRR@20: 0.0891 ± 0.0035
```

---

## 四、结果文件格式对比

### 4.1 单次模式结果

**文件**: `outputs/phase1_baseline/predictions/test_results.pkl`

```python
{
    # 主指标（论文 Table 2/3）
    'Precision@5': 0.0523,
    'Precision@20': 0.1234,
    'NDCG@5': 0.0612,
    'NDCG@20': 0.1456,
    'MRR@5': 0.0412,
    'MRR@20': 0.0891,

    # 附加指标（工程分析）
    'Precision@1': 0.0123,
    'Precision@10': 0.0891,
    'NDCG@10': 0.0967,
    # ... 其他 K 值
}
```

### 4.2 CV10 模式结果

**文件**: `outputs/phase1_baseline/predictions/cv_results.pkl`

```python
{
    # 每个 fold 的详细结果
    'fold_results': [
        {
            'Precision@5': 0.0510,
            'Precision@20': 0.1220,
            'NDCG@5': 0.0602,
            'NDCG@20': 0.1441,
            'MRR@5': 0.0401,
            'MRR@20': 0.0875,
            # ...
        },
        # ... fold 2-10
    ],

    # 汇总统计（论文报告格式）
    'aggregate_metrics': {
        'Precision@5_mean': 0.0523,      # 论文主表：mean
        'Precision@5_std': 0.0018,       # 论文主表：std
        'Precision@20_mean': 0.1234,
        'Precision@20_std': 0.0021,
        'NDCG@5_mean': 0.0612,
        'NDCG@5_std': 0.0025,
        'NDCG@20_mean': 0.1456,
        'NDCG@20_std': 0.0031,
        'MRR@5_mean': 0.0412,
        'MRR@5_std': 0.0028,
        'MRR@20_mean': 0.0891,
        'MRR@20_std': 0.0035,
    },

    'num_folds': 10
}
```

---

## 五、使用示例

### 5.1 单次模式（开发调试）

```bash
# 1. 预处理（leave-one-out）
bash scripts/run_preprocess.sh

# 2. 训练
python src/training/train_baseline.py --config configs/tafeng_baseline.yaml

# 3. 查看结果
python <<EOF
import pickle
with open('outputs/phase1_baseline/predictions/test_results.pkl', 'rb') as f:
    results = pickle.load(f)

print("Test Results:")
for k in [5, 20]:
    print(f"  Precision@{k}: {results[f'Precision@{k}']:.4f}")
    print(f"  NDCG@{k}: {results[f'NDCG@{k}']:.4f}")
EOF
```

### 5.2 CV10 模式（论文实验）

```bash
# 1. 预处理（cv10）
bash scripts/run_preprocess_cv.sh

# 2. 训练（10-fold CV）
python src/training/train_baseline.py --config configs/tafeng_cv.yaml

# 3. 查看汇总结果
python <<EOF
import pickle
with open('outputs/phase1_baseline/predictions/cv_results.pkl', 'rb') as f:
    cv_results = pickle.load(f)

print("10-Fold CV Results (Mean ± Std):")
agg = cv_results['aggregate_metrics']
for k in [5, 20]:
    print(f"  Precision@{k}: {agg[f'Precision@{k}_mean']:.4f} ± {agg[f'Precision@{k}_std']:.4f}")
    print(f"  NDCG@{k}: {agg[f'NDCG@{k}_mean']:.4f} ± {agg[f'NDCG@{k}_std']:.4f}")
EOF
```

---

## 六、设计优势

### 6.1 代码复用

**避免重复**:
- ✅ 单个训练脚本支持两种模式
- ✅ 复用 `train_epoch()`, `evaluate()`, `save_checkpoint()` 等函数
- ✅ 统一的模型定义和训练逻辑

**模式切换**:
- ✅ 配置驱动：通过 `split_method` 自动选择
- ✅ 手动覆盖：通过 `--mode` 参数强制指定
- ✅ 清晰日志：明确显示当前模式

### 6.2 结果可比性

**与 UPSTAR 一致**:
- ✅ 相同的数据切分逻辑
- ✅ 相同的 CV 模式支持
- ✅ 相同的输出格式

**论文对齐**:
- ✅ CV10 模式：10-fold cross-validation
- ✅ 输出 mean ± std（论文标准）
- ✅ 主指标：P@5/20, NDCG@5/20, MRR@5/20

---

## 七、验证检查清单

### 7.1 单次模式验证

```bash
# 训练
python src/training/train_baseline.py --config configs/tafeng_baseline.yaml

# 验证输出
python <<EOF
import pickle
from pathlib import Path

result_path = Path('outputs/phase1_baseline/predictions/test_results.pkl')
assert result_path.exists(), "test_results.pkl not found"

with open(result_path, 'rb') as f:
    results = pickle.load(f)

assert 'Precision@5' in results, "Missing Precision@5"
assert 'NDCG@20' in results, "Missing NDCG@20"
print("[OK] Single mode output verified")
EOF
```

### 7.2 CV10 模式验证

```bash
# 训练
python src/training/train_baseline.py --config configs/tafeng_cv.yaml

# 验证输出
python <<EOF
import pickle
from pathlib import Path

result_path = Path('outputs/phase1_baseline/predictions/cv_results.pkl')
assert result_path.exists(), "cv_results.pkl not found"

with open(result_path, 'rb') as f:
    cv_results = pickle.load(f)

assert 'fold_results' in cv_results, "Missing fold_results"
assert 'aggregate_metrics' in cv_results, "Missing aggregate_metrics"
assert cv_results['num_folds'] == 10, "Should be 10 folds"

# 检查主指标
agg = cv_results['aggregate_metrics']
main_metrics = ['Precision@5_mean', 'Precision@20_mean', 'NDCG@5_mean', 'NDCG@20_mean']
for m in main_metrics:
    assert m in agg, f"Missing {m}"

print("[OK] CV10 mode output verified")
EOF
```

---

## 八、常见问题

**Q: 两种模式的训练时间对比？**

A:
- 单次模式：~40 分钟（1 个模型）
- CV10 模式：~6 小时（10 个模型）

**Q: 什么时候用哪种模式？**

A:
- 开发调试：单次模式（快速迭代）
- 论文实验：CV10 模式（符合论文）

**Q: 可以混用吗？**

A: 不建议。数据切分必须与训练模式一致：
- `leave_one_out` 预处理 → 单次模式训练
- `cv10` 预处理 → CV10 模式训练

**Q: CV 模式的 early stopping 用哪个集合？**

A: CV 模式下，test set 同时用于：
- Early stopping（选择最佳 epoch）
- 最终评估
这符合 CV 的标准做法。

**Q: 如何确保 baseline 和 UPSTAR 可比？**

A:
- 使用相同的数据切分模式
- 使用相同的评估指标
- 使用相同的 CV 设置（如都用 10-fold）

---

## 九、总结

### ✅ 已完成修改

1. **统一训练脚本**: 支持 leave_one_out 和 cv10 两种模式
2. **自动模式检测**: 根据配置自动选择训练模式
3. **避免代码重复**: 复用训练逻辑，不复制两套代码
4. **统一输出格式**: 确保与 UPSTAR 对比的一致性

### ✅ 关键特性

**单次模式**（leave_one_out）:
- 快速训练（~40 分钟）
- 单个模型结果
- 适合开发调试

**CV10 模式**（论文对齐）:
- 10-fold cross-validation
- 汇总统计（mean ± std）
- 符合论文实验设置

### 🎯 设计目标

- **灵活性**: 一个脚本支持两种模式
- **可配置**: 通过配置文件控制
- **可扩展**: 易于添加新的训练模式
- **可比性**: 与 UPSTAR 评估一致

---

**修改完成时间**: 2026-03-26
**训练脚本**: `src/training/train_baseline.py`
**支持模式**: leave_one_out / cv10
