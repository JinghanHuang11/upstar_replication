#!/bin/bash
# Run Baseline Experiment on Ta-Feng Dataset
#
# Note: Baseline does not require paper-aligned retraining
# as it uses standard LSTM without the specialized modules.
#
# Usage:
#   bash scripts/run_tafeng_baseline.sh single    # Single evaluation
#   bash scripts/run_tafeng_baseline.sh cv        # 10-fold cross-validation

set -e

echo "============================================================"
echo "Baseline Experiment on Ta-Feng Dataset"
echo "============================================================"

MODE="${1:-single}"
CONFIG="${2:-configs/tafeng_baseline.yaml}"

echo ""
echo "Configuration:"
echo "  Mode: $MODE"
echo "  Config: $CONFIG"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check config
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi
echo "  [OK] Config: $CONFIG"

# Check Phase 1: Data preprocessing
if [ ! -f "data/processed/tafeng/metadata.pkl" ]; then
    echo "  [FAIL] Phase 1 data not found"
    echo "        Run: bash scripts/run_preprocess.sh"
    exit 1
fi
echo "  [OK] Phase 1: Data preprocessing complete"

echo ""
echo "All prerequisites satisfied!"
echo ""

# Create output directories
OUTPUT_BASE="outputs/phase1_baseline"
if [ "$MODE" = "single" ]; then
    OUTPUT_DIR="$OUTPUT_BASE/single_run"
    echo "Running single evaluation..."
    echo "Output directory: $OUTPUT_DIR"
else
    OUTPUT_DIR="$OUTPUT_BASE/cross_validation"
    echo "Running 10-fold cross-validation..."
    echo "Output directory: $OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/results"

# Run experiment
if [ "$MODE" = "single" ]; then
    # Single evaluation
    echo ""
    echo "Step 1: Training baseline model..."
    echo "------------------------------------------------------------"

    python src/training/train_baseline.py --config "$CONFIG"

    if [ $? -eq 0 ]; then
        echo ""
        echo "Step 2: Evaluating model..."
        echo "------------------------------------------------------------"

        # Find best model
        BEST_MODEL=$(find "$OUTPUT_DIR/checkpoints" -name "best_model.pt" -type f)

        if [ -z "$BEST_MODEL" ]; then
            echo "ERROR: Best model not found"
            exit 1
        fi

        # Evaluate
        python - <<PYTHON_SCRIPT
import sys
sys.path.append('.')

from src.evaluation.evaluator import evaluate_from_checkpoint
from src.evaluation.report import ReportGenerator

# Evaluate
metrics = evaluate_from_checkpoint(
    checkpoint_path="$BEST_MODEL",
    config_path="$CONFIG",
    split="test",
    k_values=[1, 5, 10, 15, 20, 50],
    output_dir="$OUTPUT_DIR/results/"
)

# Generate reports
generator = ReportGenerator(metrics)
generator.save_all_formats("$OUTPUT_DIR/results/")

# Save with model type label
import json
with open("$OUTPUT_DIR/results/main_results.json", 'r') as f:
    data = json.load(f)
data['model_type'] = 'baseline'
data['dataset'] = 'tafeng'
with open("$OUTPUT_DIR/results/main_results.json", 'w') as f:
    json.dump(data, f, indent=2)

print("\n" + "=" * 80)
print("Baseline Experiment Complete!")
print("=" * 80)
print("\nResults saved to: $OUTPUT_DIR/results/")
print("\nMain Results:")
print(generator.generate_main_table(format='text'))
PYTHON_SCRIPT

    else
        echo ""
        echo "============================================================"
        echo "Baseline training failed! Check logs for details."
        echo "============================================================"
        exit 1
    fi

else
    # Cross-validation
    echo ""
    echo "WARNING: 10-fold cross-validation will take significant time!"
    echo "  - Training 10 baseline models"
    echo "  - Each model: ~2-4 hours"
    echo "  - Total: ~20-40 hours"
    echo ""
    echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
    sleep 5

    echo ""
    echo "Running 10-fold cross-validation..."
    echo "------------------------------------------------------------"

    python - <<PYTHON_SCRIPT
import sys
sys.path.append('.')

from src.training.cross_validation_baseline import run_cross_validation
from src.evaluation.report import ReportGenerator

# Run cross-validation
cv_results = run_cross_validation(
    config_path="$CONFIG",
    num_folds=10,
    output_dir="$OUTPUT_DIR"
)

# Generate reports
# Convert to percentage format for reporting
cv_stats_pct = {}
for metric_name, stats in cv_results.items():
    cv_stats_pct[metric_name] = {
        'mean': stats['mean'] * 100,
        'std': stats['std'] * 100
    }

generator = ReportGenerator({}, cv_stats_pct)
generator.save_all_formats("$OUTPUT_DIR/results/")

# Save with model type label
import json
with open("$OUTPUT_DIR/results/cv_results.json", 'r') as f:
    data = json.load(f)
data['model_type'] = 'baseline'
data['dataset'] = 'tafeng'
data['num_folds'] = 10
with open("$OUTPUT_DIR/results/cv_results.json", 'w') as f:
    json.dump(data, f, indent=2)

print("\n" + "=" * 80)
print("Baseline Cross-Validation Complete!")
print("=" * 80)
print("\nResults saved to: $OUTPUT_DIR/results/")
print("\nCross-Validation Results:")
print(generator.generate_main_table(format='text'))
PYTHON_SCRIPT

    if [ $? -eq 0 ]; then
        echo ""
        echo "============================================================"
        echo "Cross-validation successful!"
        echo "============================================================"
    else
        echo ""
        echo "============================================================"
        echo "Cross-validation failed! Check logs for details."
        echo "============================================================"
        exit 1
    fi
fi
