# Phase 0 元数据格式说明

> **更新时间**: 2026-03-26
> **格式版本**: v1.1
> **向后兼容**: Yes (v1.0 完全兼容)

---

## 一、元数据文件总览

### 1.1 metadata.pkl（主元数据）

**文件位置**: `data/processed/tafeng/metadata.pkl`

**数据结构**:
```python
{
    # ===== 现有字段 (v1.0, 向后兼容) =====
    'num_users': int,              # 用户数量
    'num_items': int,              # 商品数量
    'num_interactions': int,       # 总交互数
    'user2idx': Dict,              # user_id → user_idx 映射
    'idx2user': Dict,              # user_idx → user_id 映射
    'item2idx': Dict,              # item_id → item_idx 映射
    'idx2item': Dict,              # item_idx → item_id 映射

    # ===== 新增字段 (v1.1) =====
    'statistics': Dict,            # 数据集统计信息���可选）
    'version': str,                # 元数据格式版本
}
```

### 1.2 cv_metadata.pkl（CV 专用元数据）

**文件位置**: `data/processed/tafeng/cv_splits/cv_metadata.pkl`

**数据结构**:
```python
{
    # ===== 现有字段 (v1.0, 向后兼容) =====
    'num_folds': int,              # Fold 数量（通常为 10）
    'split_method': str,           # 划分方法（'cv10'）
    'folds': List[Dict],           # 每个 fold 的统计信息

    # ===== 新增字段 (v1.1) =====
    'random_seed': int,            # 随机种子（可复现性）
    'total_users': int,            # 总用户数
    'version': str,                # CV 元数据格式版本
}
```

---

## 二、详细字段说明

### 2.1 metadata.pkl 字段详解

#### **现有字段 (v1.0)**

| 字段 | 类型 | 说明 | 使用者 |
|------|------|------|--------|
| `num_users` | int | 用户总数 | Phase 1/4/5 |
| `num_items` | int | 商品总数 | **所有阶段** |
| `num_interactions` | int | 总交互数 | 日志/分析 |
| `user2idx` | Dict | user_id → user_idx | 数据加载 |
| `idx2user` | Dict | user_idx → user_id | 结果解释 |
| `item2idx` | Dict | item_id → item_idx | **所有阶段** |
| `idx2item` | Dict | item_idx → item_id | 结果解释 |

#### **新增字段 (v1.1)**

| 字段 | 类型 | 说明 | 使用者 | 必需性 |
|------|------|------|--------|--------|
| `statistics` | Dict | 数据集统计信息 | Phase 0 分析 | 可选 |
| `statistics.avg_sequence_length` | float | 平均序列长度 | 分析/调试 | 可选 |
| `statistics.min_sequence_length` | int | 最短序列长度 | 分析/调试 | 可选 |
| `statistics.max_sequence_length` | int | 最长序列长度 | 分析/调试 | 可选 |
| `statistics.std_sequence_length` | float | 序列长度标准差 | 分析/调试 | 可选 |
| `version` | str | 元数据版本号 | 版本控制 | 可选 |

### 2.2 cv_metadata.pkl 字段详解

#### **现有字段 (v1.0)**

| 字段 | 类型 | 说明 | 使用者 |
|------|------|------|--------|
| `num_folds` | int | Fold 数量 | Phase 1/4 CV 训练 |
| `split_method` | str | 划分方法标识 | 调试/日志 |
| `folds` | List[Dict] | 每个 fold 的统计 | CV 训练循环 |

**folds 列表结构**:
```python
[
    {'train_size': 11700, 'test_size': 1300},  # Fold 1
    {'train_size': 11700, 'test_size': 1300},  # Fold 2
    ...
]
```

#### **新增字段 (v1.1)**

| 字段 | 类型 | 说明 | 使用者 | 必需性 |
|------|------|------|--------|--------|
| `random_seed` | int | 随机种子 | 可复现性 | 可选 |
| `total_users` | int | 总用户数 | 验证完整性 | 可选 |
| `version` | str | 元数据版本号 | 版本控制 | 可选 |

---

## 三、字段用途映射表

### 3.1 按使用阶段划分

| Phase | 需要的字段 | 用途 |
|-------|-----------|------|
| **Phase 0** | `statistics`, `num_interactions` | 数据分析、质量检查 |
| **Phase 1** | `num_items`, `num_users` | 模型初始化 |
| **Phase 2** | `num_items`, `item2idx` | 图构建、节点索引 |
| **Phase 3** | `num_items` | Bipartite 图构建 |
| **Phase 4** | `num_items`, `num_users` | 模型初始化 |
| **Phase 5** | `num_items`, `idx2item` | 指标计算、结果解释 |

### 3.2 按使用频率划分

| 字段 | 使用频率 | 关键程度 | 备注 |
|------|----------|----------|------|
| `num_items` | **极高** | **必需** | 所有阶段都使用 |
| `item2idx` | 高 | 必需 | 图构建需要 |
| `num_users` | 中 | 必需 | 模型初始化 |
| `statistics` | 低 | 可选 | 仅分析用途 |
| `version` | 低 | 可选 | 版本控制 |

---

## 四、向后兼容性保证

### 4.1 兼容性策略

**核心原则**: 只新增字段，不修改或删除现有字段

```python
# v1.0 格式（旧）
{
    'num_items': 500,
    'num_users': 1000,
    ...
}

# v1.1 格式（新）
{
    'num_items': 500,        # 保留
    'num_users': 1000,       # 保留
    'statistics': {...},     # 新增
    'version': '1.1'         # 新增
}
```

### 4.2 现有代码验证

**所有现有读取逻辑保持不变**:

```python
# Phase 1: train_baseline.py:40-43
with open(processed_dir / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

num_users = metadata['num_users']  # ✅ 仍然有效
num_items = metadata['num_items']  # ✅ 仍然有效
```

**新增字段的优雅处理**:

```python
# 推荐做法：检查字段是否存在
if 'statistics' in metadata:
    avg_len = metadata['statistics']['avg_sequence_length']
else:
    avg_len = None  # 旧格式没有此字段

# 或者使用 get() 方法
stats = metadata.get('statistics', {})
avg_len = stats.get('avg_sequence_length')
```

---

## 五、为什么这样更稳

### 5.1 只增加，不修改

**优势**:
- ✅ **零破坏性**: 现有代码无需修改
- ✅ **渐进式升级**: 可选择性使用新字段
- ✅ **易于回滚**: 出问题可立即忽略新字段

**示例**:
```python
# 旧代码（不知道新字段）
num_items = metadata['num_items']  # ✅ 正常工作

# 新代码（使用新字段）
stats = metadata.get('statistics', {})  # ✅ 优雅降级
if stats:
    print(f"Avg sequence length: {stats['avg_sequence_length']}")
```

### 5.2 可选字段设计

**原则**: 新增字段都是可选的，不影响核心功能

| 字段类型 | 核心功能 | 可选功能 |
|----------|----------|----------|
| **必需字段** | `num_items`, `item2idx` | - |
| **可选字段** | - | `statistics`, `version` |

**好处**:
- ✅ 即使新字段缺失，核心功能仍正常
- ✅ 支持部分数据集（无统计信息）
- ✅ 降低维护成本

### 5.3 版本号管理

**目的**: 便于未来格式升级

```python
metadata = {
    'version': '1.1',  # 当前版本
    ...
}

# 未来检查版本
if metadata.get('version', '1.0') >= '1.1':
    # 使用新功能
    stats = metadata['statistics']
else:
    # 降级处理
    stats = {}
```

---

## 六、实际使用示例

### 6.1 Phase 2: Item Graph 构建

```python
from pathlib import Path
import pickle

# 加载元数据
with open('data/processed/tafeng/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# 使用必需字段
num_items = metadata['num_items']        # ✅ 核心功能
item2idx = metadata['item2idx']          # ✅ 核心功能

# 使用可选字段（如果有）
stats = metadata.get('statistics', {})
if stats:
    logger.info(f"Avg sequence length: {stats.get('avg_sequence_length', 'N/A')}")
```

### 6.2 Phase 1: CV 训练

```python
# 加载 CV 元数据
with open('data/processed/tafeng/cv_splits/cv_metadata.pkl', 'rb') as f:
    cv_metadata = pickle.load(f)

# 使用必需字段
num_folds = cv_metadata['num_folds']      # ✅ 核心功能
fold_info = cv_metadata['folds']          # ✅ 核心功能

# 使用可选字段
seed = cv_metadata.get('random_seed', 42)  # ✅ 有默认值
logger.info(f"Using random seed: {seed}")
```

---

## 七、验证清单

### 7.1 元数据完整性检查

```bash
# 运行预处理
bash scripts/run_preprocess.sh

# 验证元数据文件
python <<EOF
import pickle
from pathlib import Path

# 检查主元数据
with open('data/processed/tafeng/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("=== Main Metadata ===")
print(f"num_items: {metadata['num_items']}")
print(f"num_users: {metadata['num_users']}")
print(f"num_interactions: {metadata['num_interactions']}")
print(f"version: {metadata.get('version', 'N/A')}")

if 'statistics' in metadata:
    stats = metadata['statistics']
    print(f"\nStatistics:")
    print(f"  avg_sequence_length: {stats.get('avg_sequence_length', 'N/A')}")
    print(f"  min_sequence_length: {stats.get('min_sequence_length', 'N/A')}")
    print(f"  max_sequence_length: {stats.get('max_sequence_length', 'N/A')}")

# 检查 CV 元数据（如果存在）
cv_path = Path('data/processed/tafeng/cv_splits/cv_metadata.pkl')
if cv_path.exists():
    with open(cv_path, 'rb') as f:
        cv_metadata = pickle.load(f)

    print(f"\n=== CV Metadata ===")
    print(f"num_folds: {cv_metadata['num_folds']}")
    print(f"split_method: {cv_metadata['split_method']}")
    print(f"random_seed: {cv_metadata.get('random_seed', 'N/A')}")
    print(f"total_users: {cv_metadata.get('total_users', 'N/A')}")
    print(f"version: {cv_metadata.get('version', 'N/A')}")

print("\n[OK] Metadata validation complete")
EOF
```

### 7.2 向后兼容性测试

```python
# 模拟现有代码
with open('data/processed/tafeng/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# 测试所有现有读取逻辑
num_items = metadata['num_items']           # ✅
num_users = metadata['num_users']           # ✅
item2idx = metadata['item2idx']             # ✅
idx2item = metadata['idx2item']             # ✅

print("[OK] All existing access patterns work")
```

---

## 八、常见问题

**Q: 为什么不添加 timestamp normalization 信息？**

A: Timestamps 直接存储在序列数据中 `[(item, ts), ...]`，无需额外映射。如果需要 day-level 节点，Phase 3 会动态构建。

**Q: 统计信息会增加多少存储空间？**

A: 可忽略不计：
- 基础 metadata: ~100KB
- 统计信息: ~1KB
- 总增加: ~1%

**Q: 旧代码需要修改吗？**

A: 不需要。所有现有字段保持不变，新增字段是可选的。

**Q: 如何处理旧版本的元数据？**

A: 代码会自动检测：
```python
version = metadata.get('version', '1.0')  # 默认为 1.0
if version >= '1.1':
    # 使用新功能
    stats = metadata['statistics']
```

**Q: CV metadata 中的 random_seed 有什么用？**

A: 用于确保可复现性。如果需要重新生成相同的 fold 划分，可以使用相同的 random_seed。

---

## 九、总结

### ✅ 设计原则
1. **向后兼容**: 只新增字段，不修改现有字段
2. **可选优先**: 新增字段都不影响核心功能
3. **渐进式**: 支持选择性使用新功能
4. **可维护**: 版本号便于未来升级

### ✅ 实现保证
- **零破坏**: 现有代码无需修改
- **优雅降级**: 缺失新字段时自动使用默认值
- **完整性**: 包含所有后续阶段需要的信息
- **可扩展**: 为未来增强留出空间

### ✅ 使用建议
- **新实验**: 可选择使用新增的统计信息
- **旧代码**: 继续使用现有字段，无需修改
- **版本检查**: 建议使用 `metadata.get('version', '1.0')` 检测版本

---

**最后更新**: 2026-03-26
**维护者**: Phase 0 数据预处理模块
