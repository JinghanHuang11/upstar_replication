#!/bin/bash
# Minimal Smoke Test for UPSTAR Pipeline
#
# Validates the complete pipeline before running full GPU experiments:
#   - CV10 splits can be loaded
#   - Item embeddings can be generated
#   - STB scores and labels can be computed
#   - UPSTAR can train (1 fold, 2 epochs)
#   - Results can be evaluated
#
# Usage:
#   bash scripts/smoke_test.sh

set -e

echo "============================================================"
echo "UPSTAR Pipeline Smoke Test"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Preprocess data (CV10 splits)"
echo "  2. Train Item-GNN (minimal epochs)"
echo "  3. Compute STB scores"
echo "  4. Train Baseline (1 fold, 3 epochs)"
echo "  5. Train UPSTAR (1 fold, 2 epochs, 1 stage)"
echo "  6. Evaluate results"
echo ""
echo "Estimated time: 30-60 minutes"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "data/raw/ta_feng.csv" ]; then
    echo "ERROR: Raw data not found: data/raw/ta_feng.csv"
    echo "Please download Ta-Feng dataset first."
    exit 1
fi
echo "  ✓ Raw data found"

echo ""
echo "All prerequisites satisfied!"
echo ""

# Clean previous smoke test results
echo "Cleaning previous smoke test results..."
rm -rf outputs/smoke_test/
mkdir -p outputs/smoke_test
echo "  ✓ Cleaned"
echo ""

# Phase 0: Preprocess (CV10 splits)
echo "============================================================"
echo "Phase 0: Data Preprocessing (CV10 Splits)"
echo "============================================================"
echo ""

bash scripts/run_preprocess.sh

if [ ! -f "data/processed/tafeng/metadata.pkl" ]; then
    echo "ERROR: Preprocessing failed - metadata not found"
    exit 1
fi
echo "✓ Phase 0 complete"
echo ""

# Phase 2: Item Representations (minimal training)
echo "============================================================"
echo "Phase 2: Item Representation Learning (Quick Test)"
echo "============================================================"
echo ""

# Create quick test config for item repr
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
  max_epochs: 3         # Quick test: 3 epochs
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

if [ ! -f "outputs/smoke_test/phase2_item_repr/checkpoints/item_embeddings.pt" ]; then
    echo "ERROR: Item repr failed - embeddings not found"
    exit 1
fi
echo "✓ Phase 2 complete"
echo ""

# Phase 3: STB Computation
echo "============================================================"
echo "Phase 3: STB Computation (Quick Test)"
echo "============================================================"
echo ""

# Copy embeddings to expected location
mkdir -p outputs/phase2_item_repr/checkpoints/
cp outputs/smoke_test/phase2_item_repr/checkpoints/item_embeddings.pt outputs/phase2_item_repr/checkpoints/

python src/training/train_stb.py --config configs/stb.yaml

if [ ! -f "outputs/phase3_stb/checkpoints/motivation_labels.npy" ]; then
    echo "ERROR: STB failed - labels not found"
    exit 1
fi
echo "✓ Phase 3 complete"
echo ""

# Phase 1: Baseline Training (1 fold, 3 epochs)
echo "============================================================"
echo "Phase 1: Baseline Training (1 Fold, 3 Epochs)"
echo "============================================================"
echo ""

python - <<PYTHON_SCRIPT
import sys
sys.path.append('.')

from src.training.cross_validation_baseline import run_cross_validation

# Run 1-fold CV with 3 epochs (quick test)
cv_results = run_cross_validation(
    config_path="configs/tafeng_baseline.yaml",
    num_folds=1,  # Quick test: 1 fold
    output_dir="outputs/smoke_test/phase1_baseline",
    quick_test=True  # Use quick test mode
)

print("\n✓ Phase 1 complete")
print(f"Results: {cv_results}")
PYTHON_SCRIPT

if [ ! -f "outputs/smoke_test/phase1_baseline/results/cv_results.json" ]; then
    echo "ERROR: Baseline training failed - results not found"
    exit 1
fi
echo ""

# Phase 4: UPSTAR Training (1 fold, 2 epochs, 1 stage)
echo "============================================================"
echo "Phase 4: UPSTAR Training (1 Fold, 2 Epochs, 1 Stage)"
echo "============================================================"
echo ""

python - <<PYTHON_SCRIPT
import sys
sys.path.append('.')

from src.training.train_upstar_cv import run_cross_validation

# Run 1-fold CV with 2 epochs per stage (quick test)
cv_results = run_cross_validation(
    config_path="configs/tafeng_upstar.yaml",
    num_folds=1,  # Quick test: 1 fold
    output_dir="outputs/smoke_test/phase4_upstar",
    quick_test=True  # Use quick test mode
)

print("\n✓ Phase 4 complete")
print(f"Results: {cv_results}")
PYTHON_SCRIPT

if [ ! -f "outputs/smoke_test/phase4_upstar/results/cv_results.json" ]; then
    echo "ERROR: UPSTAR training failed - results not found"
    exit 1
fi
echo ""

# Phase 5: Evaluation
echo "============================================================"
echo "Phase 5: Results Evaluation"
echo "============================================================"
echo ""

# Copy results to expected locations for comparison
mkdir -p outputs/phase1_baseline/results/
mkdir -p outputs/phase4_upstar/results/
cp outputs/smoke_test/phase1_baseline/results/cv_results.json outputs/phase1_baseline/results/
cp outputs/smoke_test/phase4_upstar/results/cv_results.json outputs/phase4_upstar/results/

python - <<PYTHON_SCRIPT
import sys
import json
sys.path.append('.')

from src.evaluation.comparison import ModelComparator

# Compare results
comparator = ModelComparator(
    baseline_results_dir="outputs/phase1_baseline/results",
    upstar_results_dir="outputs/phase4_upstar/results",
    output_dir="outputs/smoke_test/comparison"
)

comparator.save_comparison()
print("\n" + comparator.generate_executive_summary())
PYTHON_SCRIPT

echo ""
echo "✓ Phase 5 complete"
echo ""

# Summary
echo "============================================================"
echo "✅ SMOKE TEST PASSED!"
echo "============================================================"
echo ""
echo "All phases completed successfully:"
echo "  ✓ Phase 0: CV10 splits generated"
echo "  ✓ Phase 1: Baseline trained (1 fold, 3 epochs)"
echo "  ✓ Phase 2: Item embeddings generated"
echo "  ✓ Phase 3: STB scores and labels computed"
echo "  ✓ Phase 4: UPSTAR trained (1 fold, 2 epochs, 1 stage)"
echo "  ✓ Phase 5: Results evaluated"
echo ""
echo "Smoke test outputs:"
echo "  - outputs/smoke_test/phase1_baseline/"
echo "  - outputs/smoke_test/phase2_item_repr/"
echo "  - outputs/smoke_test/phase4_upstar/"
echo "  - outputs/smoke_test/comparison/"
echo ""
echo "Next steps:"
echo "  1. Review smoke test results to verify pipeline works"
echo "  2. Run full experiment: bash scripts/run_tafeng_baseline.sh"
echo "  3. Or run complete pipeline: python run_full_experiment.py"
echo ""
echo "Clean up smoke test:"
echo "  rm -rf outputs/smoke_test configs/smoke_item_repr.yaml"
