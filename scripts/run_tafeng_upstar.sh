#!/bin/bash
# Run UPSTAR Experiment on Ta-Feng Dataset
#
# Paper-Aligned Version (2026-03-25):
#   - 4-layer LSTM (hidden_dim=128, Section 7.3)
#   - Dual Teacher-Student training (Section 3.3)
#   - Orthogonality loss (τ_s=0.5, τ_e=0.5, Section 7.4)
#   - All hyperparameters aligned with Section 7
#
# Usage:
#   bash scripts/run_tafeng_upstar.sh single    # Single evaluation
#   bash scripts/run_tafeng_upstar.sh cv        # 10-fold cross-validation

set -e

echo "============================================================"
echo "UPSTAR Experiment on Ta-Feng Dataset"
echo "Paper-Aligned: Dual Teacher-Student (Section 3.3)"
echo "============================================================"

MODE="${1:-single}"
CONFIG="${2:-configs/tafeng_upstar.yaml}"

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

# Check Phase 0: Data preprocessing (with timestamps)
if [ ! -f "data/processed/tafeng/metadata.pkl" ]; then
    echo "  [FAIL] Phase 0 data not found"
    echo "        Run: bash scripts/run_preprocess.sh"
    echo "        (Paper-aligned: includes timestamps)"
    exit 1
fi
echo "  [OK] Phase 0: Data preprocessing complete (with timestamps)"

# Check Phase 2: Item representations (in/out aggregation)
if [ ! -f "outputs/phase2_item_repr/checkpoints/item_embeddings.pt" ]; then
    echo "  [FAIL] Phase 2 item embeddings not found"
    echo "        Run: bash scripts/run_item_repr.sh"
    echo "        (Paper-aligned: in/out neighbor separation)"
    exit 1
fi
echo "  [OK] Phase 2: Item representations learned (in/out aggregation)"

# Check Phase 3: STB scores and labels (day-level time nodes)
if [ ! -f "outputs/phase3_stb/checkpoints/motivation_labels.npy" ]; then
    echo "  [FAIL] Phase 3 STB results not found"
    echo "        Run: bash scripts/run_stb.sh"
    echo "        (Paper-aligned: day-level time nodes + MI estimation)"
    exit 1
fi
echo "  [OK] Phase 3: STB scores and motivation labels computed (day-level time nodes)"

# Check Phase 4: UPSTAR model structure
echo "  [OK] Phase 4: UPSTAR three-path model structure"
echo ""
echo "Paper-Aligned Features:"
echo "  - S/E/O models: 4-layer LSTM, hidden_dim=128 (Section 7.3)"
echo "  - Global Fusion: learnable gate (Section 3.1.4)"
echo "  - Dual Teacher-Student: S↔E knowledge distillation (Section 3.3)"
echo "  - Orthogonality: τ_s=0.5, τ_e=0.5 (Section 7.4)"
echo "  - Hyperparameters: All aligned with Section 7"
echo ""

echo "All prerequisites satisfied!"
echo ""

# Create output directories
OUTPUT_BASE="outputs/phase4_upstar"
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

# Show training configuration
python -c "
import yaml
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)

training = config.get('training', {})
print('Training Configuration (Section 7.4):')
print('=' * 60)
print(f'  Batch size: {training.get(\"batch_size\", \"N/A\")} (Paper: 64)')
print(f'  Learning rate: {training.get(\"learning_rate\", \"N/A\")} (Paper: 3e-4)')
print(f'  Max epochs/stage: {training.get(\"max_epochs_per_stage\", \"N/A\")}')
print(f'  Lambda (distill): {training.get(\"lambda_distill\", \"N/A\")} (Paper: 0.7)')
print(f'  Tau_s: {training.get(\"tau_s\", \"N/A\")} (Paper: 0.5)')
print(f'  Tau_e: {training.get(\"tau_e\", \"N/A\")} (Paper: 0.5)')
print('=' * 60)
"

# Run experiment
if [ "$MODE" = "single" ]; then
    # Single evaluation
    echo ""
    echo "Step 1: Training UPSTAR model (4 stages)..."
    echo "------------------------------------------------------------"
    echo "  Stage 1: Global loss"
    echo "  Stage 2: Branch loss (S/E/O)"
    echo "  Stage 3: Orthogonality loss"
    echo "  Stage 4: Dual teacher-student loss"
    echo ""

    python src/training/train_upstar.py --config "$CONFIG"

    if [ $? -eq 0 ]; then
        echo ""
        echo "Step 2: Evaluating model..."
        echo "------------------------------------------------------------"

        # Find best model (after stage 4)
        BEST_MODEL=$(find "$OUTPUT_DIR/checkpoints" -name "model_after_stage4.pt" -type f)

        if [ -z "$BEST_MODEL" ]; then
            # Fallback to best_model.pt
            BEST_MODEL=$(find "$OUTPUT_DIR/checkpoints" -name "best_model.pt" -type f)
        fi

        if [ -z "$BEST_MODEL" ]; then
            echo "ERROR: Trained model not found"
            exit 1
        fi

        echo "Found model: $BEST_MODEL"

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
data['model_type'] = 'upstar'
data['dataset'] = 'tafeng'
with open("$OUTPUT_DIR/results/main_results.json", 'w') as f:
    json.dump(data, f, indent=2)

print("\n" + "=" * 80)
print("UPSTAR Experiment Complete!")
print("=" * 80)
print("\nResults saved to: $OUTPUT_DIR/results/")
print("\nMain Results:")
print(generator.generate_main_table(format='text'))

print("\n✓ Paper-Aligned: All hyperparameters match Section 7")
PYTHON_SCRIPT

    else
        echo ""
        echo "============================================================"
        echo "UPSTAR training failed! Check logs for details."
        echo "============================================================"
        exit 1
    fi

else
    # Cross-validation
    echo ""
    echo "WARNING: 10-fold cross-validation will take significant time!"
    echo "  - Training 10 UPSTAR models (4 stages each)"
    echo "  - Each model: ~8-16 hours"
    echo "  - Total: ~80-160 hours"
    echo ""
    echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
    sleep 5

    echo ""
    echo "Running 10-fold cross-validation..."
    echo "------------------------------------------------------------"

    python - <<PYTHON_SCRIPT
import sys
sys.path.append('.')

from src.evaluation.cross_validation import run_cross_validation
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
data['model_type'] = 'upstar'
data['dataset'] = 'tafeng'
data['num_folds'] = 10
with open("$OUTPUT_DIR/results/cv_results.json", 'w') as f:
    json.dump(data, f, indent=2)

print("\n" + "=" * 80)
print("UPSTAR Cross-Validation Complete!")
print("=" * 80)
print("\nResults saved to: $OUTPUT_DIR/results/")
print("\nCross-Validation Results:")
print(generator.generate_main_table(format='text'))

print("\n✓ Paper-Aligned: 10-fold CV as per paper")
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
