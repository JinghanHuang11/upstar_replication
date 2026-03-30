# UPSTAR Quick Reference - Entry Points

## 🎯 Recommended Main Entry Points (唯一主入口清单)

### **Phase 1: Data Processing**
```bash
# Build sequences (cv10-only)
python -m src.data.build_sequences --config configs/dataset_tafeng.yaml
```

### **Phase 2: Item Representation Learning**
```bash
# Train Item GNN
python -m src.training.train_item_repr --config configs/item_repr.yaml
```

### **Phase 3: STB Computation**
```bash
# Compute STB (paper-aligned MI - default)
python -m src.training.train_stb --config configs/stb.yaml

# With explicit mode
python -m src.training.train_stb --config configs/stb.yaml --mode=paper     # MI-based
python -m src.training.train_stb --config configs/stb.yaml --mode=baseline  # Cosine similarity
python -m src.training.train_stb --config configs/stb.yaml --mode=auto      # Auto-detect
```

### **Phase 4: Model Training**

#### Baseline LSTM
```bash
# 10-fold CV with internal validation
python -m src.training.train_baseline --config configs/tafeng_baseline.yaml
```

#### UPSTAR
```bash
# 10-fold CV with 4-stage curriculum
python -m src.training.train_upstar --config configs/tafeng_upstar.yaml
```

### **Phase 5: Evaluation**
```bash
# Single model evaluation
python -m src.evaluation.evaluate --config configs/tafeng_eval.yaml --mode eval --checkpoint <path>

# Cross-validation evaluation
python -m src.evaluation.evaluate --config configs/tafeng_eval.yaml --mode cv
```

---

## ❌ Deprecated Files (已废弃文件清单)

### **Training**
- `src/training/train_upstar_stage1.py` → Use `train_upstar.py` instead
- `src/training/train_stb_advanced.py` → Use `train_stb.py --mode <paper|baseline>` instead

### **Evaluation**
- `src/evaluation/evaluate_baseline.py` → Use `src.evaluation.metrics` instead
- `src/evaluation/cross_validation.py` → Use `src.training.train_upstar_cv` instead

### **Scripts**
- `scripts/run_stb_advanced.sh` → Use `scripts/run_stb.sh` instead

---

## 📊 Module Structure

### **Active (Recommended)**
```
src/training/
├── train_item_repr.py           # Phase 2: Item GNN
├── train_stb.py                 # Phase 3: STB (MAIN)
├─��� train_baseline.py            # Phase 4: Baseline wrapper
├── cross_validation_baseline.py # Phase 4: Baseline impl
├── train_upstar.py              # Phase 4: UPSTAR wrapper
└── train_upstar_cv.py           # Phase 4: UPSTAR impl

src/evaluation/
├── metrics.py                   # Unified metrics (CORE)
├── evaluator.py                 # UPSTAR evaluator
└── evaluate.py                  # Evaluation CLI
```

### **Deprecated (Not Recommended)**
```
src/training/
├── train_upstar_stage1.py      # ❌ Old 3-stage training
└── train_stb_advanced.py       # ❌ Development comparison

src/evaluation/
├── evaluate_baseline.py        # ❌ Duplicate of metrics.py
└── cross_validation.py         # ❌ CV in wrong location

scripts/
└── run_stb_advanced.sh         # ❌ Wraps deprecated script
```

---

## 🔧 Common Tasks

### **Train Baseline**
```bash
python -m src.training.train_baseline --config configs/tafeng_baseline.yaml
# or
bash scripts/run_tafeng_baseline.sh
```

### **Train UPSTAR**
```bash
python -m src.training.train_upstar --config configs/tafeng_upstar.yaml
# or
bash scripts/run_tafeng_upstar.sh
```

### **Compute STB**
```bash
python -m src.training.train_stb --config configs/stb.yaml
# or
bash scripts/run_stb.sh
```

### **Evaluate Model**
```bash
python -m src.evaluation.evaluate \
  --config configs/tafeng_eval.yaml \
  --mode eval \
  --checkpoint outputs/phase4_upstar/checkpoints/best_model.pt
```

---

## 📝 Notes

1. **All entry points use `-m` flag**: `python -m src.module.name`
2. **Scripts are wrappers**: `bash scripts/run_xxx.sh` calls the module
3. **Deprecated files show warnings**: They will print migration instructions
4. **Metrics are unified**: Use `src.evaluation.metrics` for all evaluation

---

**Full documentation**: See [CLEANUP_GUIDE.md](CLEANUP_GUIDE.md)
