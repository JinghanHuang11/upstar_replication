# UPSTAR Outputs Directory

This directory contains the outputs from UPSTAR experiments using the CV10-ONLY mode.

## Directory Structure (CV10-ONLY)

```
outputs/
├── phase1_baseline/              # Baseline 10-fold CV results
│   ├── checkpoints/              # Model checkpoints for each fold
│   ├── logs/                     # Training logs
│   └── results/
│       └── cv_results.json       # ⭐ Aggregated 10-fold CV results
│
├── phase2_item_repr/             # Item representation learning
│   ├── checkpoints/
│   │   └── item_embeddings.pt    # ⭐ Learned item embeddings
│   ├── logs/                     # Training logs
│   └── predictions/              # Item graph predictions
│
├── phase3_stb/                   # STB computation results
│   ├── checkpoints/
│   │   ├── stb_scores.npy        # ⭐ STB scores for each item
│   │   ├── motivation_labels.npy # ⭐ Motivation labels (0=exploratory, 1=stable, 2=uncategorized)
│   │   └── stb_stats.pkl         # STB computation statistics
│   ├── logs/                     # Computation logs
│   └── predictions/              # STB predictions
│
├── phase4_upstar/                # UPSTAR 10-fold CV results
│   ├── checkpoints/              # Model checkpoints for each fold
│   ├── logs/                     # Training logs
│   └── results/
│       └── cv_results.json       # ⭐ Aggregated 10-fold CV results
│
├── comparison/                   # Baseline vs UPSTAR comparison
│   ├── comparison.txt            # Human-readable comparison
│   └── improvement_analysis.txt  # Performance improvement analysis
│
└── smoke_test/                   # Smoke test outputs (for validation)
    ├── phase1_baseline/
    ├── phase2_item_repr/
    ├── phase4_upstar/
    └── comparison/
```

## Key Result Files

### Baseline Results
- **File**: `phase1_baseline/results/cv_results.json`
- **Content**: 10-fold CV aggregated results (mean ± std for each metric)
- **Metrics**: Precision@K, NDCG@K, MRR@K, HitRate@K

### UPSTAR Results
- **File**: `phase4_upstar/results/cv_results.json`
- **Content**: 10-fold CV aggregated results (mean ± std for each metric)
- **Metrics**: Precision@K, NDCG@K, MRR@K, HitRate@K

### Item Embeddings
- **File**: `phase2_item_repr/checkpoints/item_embeddings.pt`
- **Content**: Learned item embeddings from Item-GNN
- **Dimension**: [num_items, embed_dim=128]

### STB Scores and Labels
- **Files**:
  - `phase3_stb/checkpoints/stb_scores.npy`
  - `phase3_stb/checkpoints/motivation_labels.npy`
- **Content**: STB scores and motivation classification
- **Labels**: 0=exploratory, 1=stable, 2=uncategorized

## Result File Format

### cv_results.json Format
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
    },
    ...
  }
}
```

## Cleanup

To clean old results and free up disk space:

```bash
# Clean specific phases
rm -rf outputs/phase1_baseline/
rm -rf outputs/phase2_item_repr/
rm -rf outputs/phase3_stb/
rm -rf outputs/phase4_upstar/
rm -rf outputs/comparison/
rm -rf outputs/smoke_test/

# Or use the provided script
bash scripts/clean_old_data.sh
```

## Notes

- All experiments use 10-fold cross-validation (CV10-ONLY mode)
- Results are reported as mean ± standard deviation across 10 folds
- Internal validation split (10% of training data) is used for early stopping
- Final evaluation is performed on held-out test fold
