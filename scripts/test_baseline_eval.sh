#!/bin/bash
# Quick smoke test: Baseline CV10 training (minimal folds)
# Verifies CV10 training pipeline works correctly

set -e

echo "============================================================"
echo "Baseline CV10 Smoke Test"
echo "============================================================"

echo ""
echo "This will:"
echo "  1. Run 2-fold CV (instead of 10) for quick testing"
echo "  2. Train baseline with minimal epochs"
echo "  3. Verify CV10 pipeline works end-to-end"
echo ""

# Check prerequisites
if [ ! -f "data/processed/tafeng/metadata.pkl" ]; then
    echo "ERROR: Data not found. Run: bash scripts/run_preprocess.sh"
    exit 1
fi

echo "✓ Data found"
echo ""

# Create smoke test config
cat > configs/smoke_test.yaml <<EOF
# Smoke test config (2-fold CV, minimal epochs)
dataset:
  name: tafeng
  data_dir: data/raw/
  processed_dir: data/processed/tafeng/
  cache_dir: data/cache/tafeng/
  min_item_frequency: 2
  split_method: 'cv10'  # CV10-ONLY mode
  num_folds: 2          # Smoke test: only 2 folds
  cv_random_seed: 42

model:
  embed_dim: 64
  hidden_dim: 128
  num_layers: 2         # Smoke test: 2 layers (instead of 4)
  dropout: 0.2
  max_seq_length: 50

training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.00001
  max_epochs: 3         # Smoke test: only 3 epochs
  early_stop_patience: 2
  metric_for_best: 'NDCG@20'
  num_workers: 4

  # Internal validation split
  val_ratio: 0.1
  val_random_seed: 42

evaluation:
  k_values: [5, 10, 20]
  batch_size: 256

logging:
  log_dir: outputs/smoke_test/logs/
  checkpoint_dir: outputs/smoke_test/checkpoints/
  prediction_dir: outputs/smoke_test/predictions/

results:
  output_dir: outputs/smoke_test/results/
  model_type: baseline

seed: 42
deterministic: true
EOF

echo "Running 2-fold CV smoke test..."
echo "------------------------------------------------------------"

python - <<PYTHON_SCRIPT
import sys
sys.path.append('.')

from src.training.cross_validation_baseline import run_cross_validation
from src.evaluation.report import ReportGenerator

# Run 2-fold cross-validation (smoke test)
cv_results = run_cross_validation(
    config_path="configs/smoke_test.yaml",
    num_folds=2,
    output_dir="outputs/smoke_test"
)

# Generate reports
cv_stats_pct = {}
for metric_name, stats in cv_results.items():
    cv_stats_pct[metric_name] = {
        'mean': stats['mean'] * 100,
        'std': stats['std'] * 100
    }

generator = ReportGenerator({}, cv_stats_pct)
generator.save_all_formats("outputs/smoke_test/results/")

print("\n" + "=" * 80)
print("Smoke Test Complete!")
print("=" * 80)
print("\n✓ CV10 pipeline works correctly")
print("\n2-Fold CV Results:")
print(generator.generate_main_table(format='text'))
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Smoke test passed!"
    echo "============================================================"
    echo ""
    echo "CV10 pipeline is working correctly."
    echo "You can now run the full 10-fold experiment:"
    echo "  bash scripts/run_tafeng_baseline.sh"
    echo ""
    echo "Clean up smoke test outputs:"
    echo "  rm -rf outputs/smoke_test configs/smoke_test.yaml"
else
    echo ""
    echo "============================================================"
    echo "✗ Smoke test failed!"
    echo "============================================================"
    exit 1
fi
