# Phase 0 数据切分模式说明

> **更新时间**: 2026-03-26  
> **支持模式**: `leave_one_out` (默认) | `cv10` (10-fold CV)

---

## 一、两种切分模式对比

| 特性 | leave_one_out | cv10 |
|------|---------------|------|
| **用途** | 工程模式 / 快速验证 | 论文模式 / 最终实验 |
| **配置值** | `split_method: 'leave_one_out'` | `split_method: 'cv10'` |
| **切分策略** | 基于用户序列内部切分 | 用户级 10 折交叉验证 |
| **输出文件数** | 3 个文件 | 10 个 fold × 2 个文件 |
| **验证集** | 有 | 无（train/test 只有两部分） |
| **可复现性** | 确定性 | 确定性（固定 random_seed） |

---

## 二、leave_one_out 模式（默认/工程模式）

### 配置示例
```yaml
# configs/tafeng_baseline.yaml
dataset:
  split_method: 'leave_one_out'
```

### 切分策略
对每个用户序列：
- **Train**: 序列前 `n-2` 个 item（用于输入）
- **Val**: 序列第 `n-1` 个 item（验证目标）
- **Test**: 序列第 `n` 个 item（测试目标）

```
用户序列: [A, B, C, D, E, F, G]
├── Train: [A, B, C, D, E] → target: F (val)
├── Val:   [A, B, C, D, E, F] → target: G (test)
└── Test:  [A, B, C, D, E, F] → target: G (test)
```

### 输出文件结构
```
data/processed/tafeng/
├── metadata.pkl                 # 用户/商品词表
├── train_sequences.pkl          # 训练序列
├── val_sequences.pkl            # 验证序列
└── test_sequences.pkl           # 测试序列
```

### 文件格式
```python
# train_sequences.pkl / val_sequences.pkl / test_sequences.pkl
{
    user_idx_1: {
        'items': [(item_1, ts_1), (item_2, ts_2), ...],  # 输入序列（含 timestamps）
        'target': item_target                            # 目标 item（不含 timestamp）
    },
    user_idx_2: {...},
    ...
}
```

### 使用示例
```bash
# 预处理
bash scripts/run_preprocess.sh

# 训练（单次运行）
bash scripts/run_tafeng_baseline.sh single
```

---

## 三、cv10 模式（论文模式）

### 配置示例
```yaml
# configs/tafeng_cv.yaml
dataset:
  split_method: 'cv10'
  num_folds: 10
  cv_random_seed: 42
```

### 切分策略
1. 将所有用户随机 shuffle（固定 seed）
2. 平均分成 10 个 fold
3. 对每个 fold i（i = 1..10）：
   - **Test**: fold i 中的所有用户
   - **Train**: 其余 9 个 fold 的所有用户
4. 每个用户的**完整序列**用于 train/test（不再内部切分）

```
所有用户: [U1, U2, U3, ..., U10000]
随机 shuffle 后分为 10 fold:
├── Fold 1:  [U1, U1002, ...]   → Test: Fold 1, Train: Fold 2-10
├── Fold 2:  [U5, U1020, ...]   → Test: Fold 2, Train: Fold 1,3-10
├── ...
└── Fold 10: [U8, U2005, ...]   → Test: Fold 10, Train: Fold 1-9
```

**每个用户的数据构造**（用于 train 或 test）：
```python
# 用户序列: [A, B, C, D, E, F, G]
{
    'items': [(A, ts1), (B, ts2), (C, ts3), (D, ts4), (E, ts5), (F, ts6)],  # 输入
    'target': G  # 目标
}
```

### 输出文件结构
```
data/processed/tafeng/
├── metadata.pkl                 # 用户/商品词表
└── cv_splits/                   # CV 划分目录
    ├── cv_metadata.pkl          # 元数据（fold 大小等）
    ├── fold_1/
    │   ├── train_sequences.pkl
    │   └── test_sequences.pkl
    ├── fold_2/
    │   ├── train_sequences.pkl
    │   └── test_sequences.pkl
    ...
    └── fold_10/
        ├── train_sequences.pkl
        └── test_sequences.pkl
```

### 文件格式

#### cv_metadata.pkl
```python
{
    'num_folds': 10,
    'split_method': 'cv10',
    'folds': [
        {'train_size': 11700, 'test_size': 1300},  # Fold 1
        {'train_size': 11700, 'test_size': 1300},  # Fold 2
        ...
    ]
}
```

#### fold_i/train_sequences.pkl & test_sequences.pkl
```python
{
    user_idx_1: {
        'items': [(item_1, ts_1), (item_2, ts_2), ...],  # 输入序列（含 timestamps）
        'target': item_target                            # 目标 item
    },
    user_idx_2: {...},
    ...
}
```

### 使用示例
```bash
# 预处理
bash scripts/run_preprocess_cv.sh

# 训练（10-fold CV）
python src/training/cross_validation_baseline.py --config configs/tafeng_cv.yaml
# 或
bash scripts/run_tafeng_baseline.sh cv
```

---

## 四、兼容性保证

### 1. 序列格式完全一致
**两种模式输出相同的序列格式**：
```python
{
    'items': [(item, timestamp), ...],  # 保留 timestamps
    'target': item                      # 目标 item
}
```

这确保了：
- ✅ Phase 2 (Item Graph) 可以直接使用
- ✅ Phase 3 (Item-Time Graph) 可以提取 timestamps 构造 day-level 时间节点
- ✅ Phase 4 (UPSTAR) 无需修改数据加载逻辑

### 2. 接口兼容性

#### leave_one_out 模式加载（现有代码无需修改）
```python
from src.data.build_sequences import SequenceBuilder

# 现有代码继续工作
with open('data/processed/tafeng/train_sequences.pkl', 'rb') as f:
    train_sequences = pickle.load(f)
with open('data/processed/tafeng/val_sequences.pkl', 'rb') as f:
    val_sequences = pickle.load(f)
with open('data/processed/tafeng/test_sequences.pkl', 'rb') as f:
    test_sequences = pickle.load(f)
```

#### cv10 模式加载（新增简单的循环逻辑）
```python
from pathlib import Path
import pickle

cv_dir = Path('data/processed/tafeng/cv_splits')

for fold_idx in range(1, 11):
    fold_dir = cv_dir / f'fold_{fold_idx}'
    
    with open(fold_dir / 'train_sequences.pkl', 'rb') as f:
        train_sequences = pickle.load(f)
    with open(fold_dir / 'test_sequences.pkl', 'rb') as f:
        test_sequences = pickle.load(f)
    
    # 训练和评估 fold i
    train_and_evaluate_fold(train_sequences, test_sequences, fold_idx)
```

### 3. 向后兼容
- ✅ 现有 `leave_one_out` 模式**完全不受影响**
- ✅ 默认行为保持不变（`split_method` 默认为 `'leave_one_out'`）
- ✅ 已有的训练脚本无需修改
- ✅ 新增 `cv10` 模式作为可选功能

---

## 五、配置文件速查表

| 模式 | 配置文件 | split_mode 值 | 输出位置 |
|------|----------|---------------|----------|
| Leave-One-Out | `configs/tafeng_baseline.yaml` | `'leave_one_out'` | `data/processed/tafeng/*.pkl` |
| 10-Fold CV | `configs/tafeng_cv.yaml` | `'cv10'` | `data/processed/tafeng/cv_splits/fold_*/` |

---

## 六、常见问题

**Q: 两种模式的序列格式有什么区别？**

A: **没有区别**。两种模式都输出 `[(item, timestamp), ...]` 格式，完全兼容后续阶段。

**Q: cv10 模式为什么没有验证集？**

A: 10-fold CV 本身就是通过 10 次训练来评估模型泛化性能，不需要额外的验证集。每次训练时可以从 train 中划分一部分用于验证。

**Q: 如何切换模式？**

A: 只需修改配置文件中的 `split_method` 值，然后重新运行预处理脚本。

**Q: cv10 模式训练时间会更长吗？**

A: 是的，cv10 需要训练 10 个模型，大约是单次训练的 10 倍时间。建议：
- 开发调试阶段使用 `leave_one_out`
- 最终实验报告使用 `cv10`（论文对齐）

**Q: 现有的 baseline 和 UPSTAR 训练脚本需要修改吗？**

A: 不需要。两种模式输出的序列格式完全一致，现有代码可直接使用。

---

## 七、代码实现关键点

### build_sequences.py 核心逻辑
```python
class SequenceBuilder:
    def __init__(self, config: Dict):
        # 支持 'leave_one_out' (默认) 或 'cv10'
        self.split_method = config['dataset'].get('split_method', 'leave_one_out')
        
        # 向后兼容：支持旧配置 '10fold_cv'
        if self.split_method == '10fold_cv':
            self.split_method = 'cv10'
    
    def run(self, df: pd.DataFrame):
        user_sequences = self.build_user_sequences(df)
        
        if self.split_method == 'cv10':
            # 10-fold CV 模式
            train_list, test_list = self.split_10fold_cv(user_sequences)
            self.save_sequences(train_sequences_list=train_list,
                              test_sequences_list=test_list)
        else:  # leave_one_out 或 ratio
            # 单次划分模式
            train, val, test = self.split_leave_one_out(user_sequences)
            self.save_sequences(train, val, test)
```

---

**最后更新**: 2026-03-26  
**维护者**: Phase 0 数据预处理模块
