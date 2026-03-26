# Phase 0 序列输出格式说明

> **更新时间**: 2026-03-26
> **格式版本**: Paper-Aligned Format v1.0
> **向后兼容**: Yes (支持旧格式)

---

## 一、序列格式总览

### 1.1 Paper-Aligned Format（新格式，推荐）

**数据结构**:
```python
{
    user_idx: {
        'items': [(item_1, ts_1), (item_2, ts_2), ...],  # (item_id, timestamp) 元组
        'target': item_target                              # 预测目标 item_id
    }
}
```

**关键特性**:
- ✅ **保留 timestamps**: 支持 Phase 2 session 构造和 Phase 3 day-level item-time graph
- ✅ **论文对齐**: 符合 UPSTAR 论文 Section 7.1 的时间节点要求
- ✅ **向后兼容**: 现有训练代码自动检测并兼容两种格式

### 1.2 Legacy Format（旧格式，仅兼容）

**数据结构**:
```python
{
    user_idx: {
        'items': [item_1, item_2, ...],  # 仅 item_id，无 timestamp
        'target': item_target
    }
}
```

**局限性**:
- ❌ 无法构造 session（需要时间间隔）
- ❌ 无法构造 day-level item-time graph（需要日期信息）
- ⚠️ 仅用于向后兼容已有代码

---

## 二、示例样本

### 2.1 Leave-One-Out 模式输出

#### **train_sequences.pkl**
```python
{
    42: {
        'items': [
            (101, 975678500),  # (item_id, Unix_timestamp)
            (205, 975687100),
            (308, 975695800),
            (412, 983491200),  # 跨度多天的购买记录
            (515, 983499900)
        ],
        'target': 618  # 预测目标（仅 item_id）
    },
    123: {
        'items': [
            (77, 975654300),
            (88, 975662900)
        ],
        'target': 99
    }
}
```

#### **val_sequences.pkl** & **test_sequences.pkl**
```python
{
    42: {
        'items': [
            (101, 975678500),
            (205, 975687100),
            (308, 975695800),
            (412, 983491200),
            (515, 983499900),
            (618, 992265600)  # val 时加入
        ],
        'target': 721  # test 目标
    }
}
```

### 2.2 CV10 模式输出

#### **cv_splits/fold_1/train_sequences.pkl**
```python
{
    42: {
        'items': [
            (101, 975678500),
            (205, 975687100),
            ...,
            (999, 992271100)  # 用户完整购买历史（除最后一个）
        ],
        'target': 1001  # 最后一个 item 作为目标
    },
    # 约 11700 个用户
}
```

#### **cv_splits/fold_1/test_sequences.pkl**
```python
{
    157: {  # fold 1 的测试用户
        'items': [(5, 975612000), ..., (888, 992260000)],
        'target': 889
    },
    # 约 1300 个用户
}
```

---

## 三、格式用途对比

| 组件 | 需要 timestamps | 使用字段 | 论文对齐 |
|------|----------------|----------|----------|
| **Phase 0 输出** | ✅ | `'items': [(item, ts), ...]` | ✅ Section 7.1 |
| **Phase 2 (Item Graph)** | ✅ | 提取 timestamps 构造 session | ✅ Section 3.2 |
| **Phase 3 (Item-Time Graph)** | ✅ | 提取 timestamps 构造 day-level 节点 | ✅ Section 7.1 |
| **Phase 4 (Training)** | ❌ | 仅使用 item_id | ✅ Section 3.1.4 |
| **Phase 5 (Evaluation)** | ❌ | 仅使用 item_id | ✅ Section 7.5 |

---

## 四、后续模块使用方式

### 4.1 Phase 2: Item Graph（需要 timestamps）

**代码位置**: `src/graphs/item_graph.py:49-100`

**Session 构造逻辑**:
```python
def build_session_sequences(self, user_sequences):
    for user_idx, data in user_sequences.items():
        raw_items = data['items']  # [(item, ts), ...]

        # 检测是否有 timestamps
        has_timestamps = isinstance(raw_items[0], tuple)

        if has_timestamps:
            # 按时间间隔分 session
            sorted_items = sorted(raw_items, key=lambda x: x[1])
            for k in range(1, len(sorted_items)):
                if (sorted_items[k][1] - sorted_items[k-1][1]) > threshold:
                    # 新 session 开始
                    ...
        else:
            # 向后兼容：整个序列作为一个 session
            user_sessions[user_idx] = [list(raw_items)]
```

### 4.2 Phase 3: Item-Time Graph（需要 timestamps）

**代码位置**: `src/graphs/item_time_graph.py:65-95`

**Day-level 时间节点构造**:
```python
def build_time_nodes(self, user_sequences):
    all_days = set()
    for user_idx, data in user_sequences.items():
        items_with_ts = data['items']  # [(item, ts), ...]
        for item, ts in items_with_ts:
            # Unix timestamp → 日期字符串
            day_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            all_days.add(day_str)

    # 每个唯一日期成为时间节点
    time2idx = {day: i for i, day in enumerate(sorted(all_days))}
    return time2idx
```

### 4.3 Phase 4: Training（不需要 timestamps）

**代码位置**: `src/data/dataset.py:61-96`

**自动格式检测与提取**:
```python
def __getitem__(self, idx):
    items = sample['items']

    # **PAPER-ALIGNED FORMAT HANDLING**
    if len(items) > 0 and isinstance(items[0], (tuple, list)):
        # 新格式: [(item, timestamp), ...] → 提取 items
        items = [item[0] for item in items]
    # 旧格式: [item1, item2, ...] → 直接使用

    # 训练只需要 item_id
    return {
        'items': torch.tensor(items),
        'target': torch.tensor(target)
    }
```

---

## 五、论文对齐 vs 工程兼容

### 5.1 论文对齐部分（Paper-Aligned）

| 组件 | 实现 | 论文依据 |
|------|------|----------|
| **Timestamps 保留** | `'items': [(item, ts), ...]` | Section 7.1: "day-level time nodes" |
| **Day-level 节点** | `datetime.fromtimestamp(ts).strftime('%Y-%m-%d')` | Section 7.1: "ρ = 50 time nodes" |
| **Session 构造** | 时间间隔 > threshold → 新 session | Section 3.2: "in-session/cross-session graphs" |
| **Item-Time 边** | 每个 (item, ts) 创建边到对应日期节点 | Section 3.1.3: "bipartite item-time graph" |

### 5.2 工程兼容部分（Backward Compatibility）

| 组件 | 实现 | 用途 |
|------|------|------|
| **旧格式支持** | 检测 `isinstance(items[0], tuple)` | 允许旧代码继续工作 |
| **自动降级** | 无 timestamps 时整个序列作为单 session | 防止崩溃，优雅降级 |
| **训练代码兼容** | 自动提取 item_id，忽略 timestamps | 训练阶段不需要时间信息 |

---

## 六、验证清单

### 6.1 Phase 0 输出验证

```bash
# 运行预处理
bash scripts/run_preprocess.sh

# 验证输出格式
python - <<EOF
import pickle
from pathlib import Path

processed_dir = Path('data/processed/tafeng')
with open(processed_dir / 'train_sequences.pkl', 'rb') as f:
    train = pickle.load(f)

# 检查第一个样本
user_id = list(train.keys())[0]
sample = train[user_id]

print("User ID:", user_id)
print("Items format:", type(sample['items'][0]))
print("Sample items:", sample['items'][:3])
print("Target:", sample['target'])

# 验证 timestamps
if isinstance(sample['items'][0], tuple):
    print("✅ Paper-aligned format: (item, timestamp)")
else:
    print("⚠️ Legacy format: [item]")
EOF
```

### 6.2 Phase 2 验证

```python
from src.graphs.item_graph import ItemGraphBuilder

builder = ItemGraphBuilder(config)
train_seqs = builder.load_sequences('train')
sessions = builder.build_session_sequences(train_seqs)

# 检查 session 构造结果
user_id = list(sessions.keys())[0]
print(f"User {user_id} has {len(sessions[user_id])} sessions")
```

### 6.3 Phase 3 验证

```python
from src.graphs.item_time_graph import ItemTimeGraphBuilder

builder = ItemTimeGraphBuilder(config)
train_seqs = builder.load_sequences('train')
time2idx = builder.build_time_nodes(train_seqs)

print(f"Built {len(time2idx)} day-level time nodes")
print("Sample time nodes:", list(time2idx.keys())[:5])
```

---

## 七、常见问题

**Q: 为什么 target 不包含 timestamp？**

A: Target 是预测目标，只需要 item_id。Timestamps 只在输入序列中保留，用于：
- Session 构造（Phase 2）
- Item-Time 图构建（Phase 3）

**Q: 如何处理旧格式数据？**

A: 代码自动检测并兼容：
- 新格式: `[(item, ts), ...]` → 提取 timestamps
- 旧格式: `[item, ...]` → 忽略 timestamps 功能

**Q: 可以重新生成新格式吗？**

A: 可以，只需重新运行预处理：
```bash
# 清除旧数据
rm -rf data/processed/tafeng/*.pkl

# 重新预处理（自动生成新格式）
bash scripts/run_preprocess.sh
```

**Q: 新格式会增加存储空间吗？**

A: 影响很小：
- 旧格式: `4 bytes` per item (int32)
- 新格式: `8 bytes` per item (int32 + int32)
- 对于 Tafeng 数据集 (~800K 交互): 约 +3MB

---

## 八、总结

### ✅ 论文对齐（必须保留）
- **Timestamps**: 支持 day-level item-time graph (Section 7.1)
- **Session 构造**: 支持时间间隔分 session (Section 3.2)
- **输出格式**: `[(item, timestamp), ...]`

### ✅ 工程兼容（向后兼容）
- **旧格式支持**: 自动检测并处理 `[item, ...]` 格式
- **训练代码**: 自动提取 item_id，忽略 timestamps
- **优雅降级**: 无 timestamps 时仍可工作（功能受限）

### 🎯 使用建议
- **新实验**: 始终使用新格式（`run_preprocess.sh`）
- **已有结果**: 可继续使用旧格式（功能受限）
- **论文实验**: 必须使用新格式（完整功能）

---

**最后更新**: 2026-03-26
**维护者**: Phase 0 数据预处理模块
