#!/bin/bash
# Compare Baseline and UPSTAR Results
#
# Usage:
#   bash scripts/compare_results.sh

set -e

echo "============================================================"
echo "Comparing Baseline vs UPSTAR Results"
echo "============================================================"

echo ""
echo "Looking for results..."

# Find result directories
BASELINE_DIR=$(find outputs/phase1_baseline -name "main_results.json" -exec dirname {} \; 2>/dev/null | head -1)
UPSTAR_DIR=$(find outputs/phase4_upstar -name "main_results.json" -exec dirname {} \; 2>/dev/null | head -1)

if [ -z "$BASELINE_DIR" ]; then
    echo "ERROR: Baseline results not found"
    echo "Expected: outputs/phase1_baseline/single_run/results/main_results.json"
    echo "          or: outputs/phase1_baseline/cross_validation/results/main_results.json"
    echo ""
    echo "Please run baseline first:"
    echo "  bash scripts/run_tafeng_baseline.sh single"
    exit 1
fi

if [ -z "$UPSTAR_DIR" ]; then
    echo "ERROR: UPSTAR results not found"
    echo "Expected: outputs/phase4_upstar/single_run/results/main_results.json"
    echo "          or: outputs/phase4_upstar/cross_validation/results/main_results.json"
    echo ""
    echo "Please run UPSTAR first:"
    echo "  bash scripts/run_tafeng_upstar.sh single"
    exit 1
fi

echo ""
echo "Found results:"
echo "  Baseline: $BASELINE_DIR"
echo "  UPSTAR:   $UPSTAR_DIR"
echo ""

# Create output directory
COMPARISON_DIR="outputs/comparison"
mkdir -p "$COMPARISON_DIR"

echo "Generating comparison..."
echo "------------------------------------------------------------"

python - <<PYTHON_SCRIPT
import sys
sys.path.append('.')

from src.evaluation.comparison import ModelComparator

# Compare
comparator = ModelComparator(
    baseline_results_dir="$BASELINE_DIR",
    upstar_results_dir="$UPSTAR_DIR",
    output_dir="$COMPARISON_DIR"
)

# Save comparison
comparator.save_comparison()

# Print executive summary
print(comparator.generate_executive_summary())
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Comparison complete!"
    echo "============================================================"
    echo ""
    echo "Results saved to: $COMPARISON_DIR"
    echo ""
    echo "View comparison:"
    echo "  cat $COMPARISON_DIR/comparison.txt"
    echo ""
    echo "View improvement analysis:"
    echo "  cat $COMPARISON_DIR/improvement_analysis.txt"
else
    echo ""
    echo "Comparison failed!"
    exit 1
fi
