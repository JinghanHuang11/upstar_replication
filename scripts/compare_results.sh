#!/bin/bash
# Compare Baseline and UPSTAR Results (CV10-ONLY)
#
# Usage:
#   bash scripts/compare_results.sh

set -e

echo "============================================================"
echo "Comparing Baseline vs UPSTAR Results (10-Fold CV)"
echo "============================================================"

echo ""
echo "Looking for results..."

# Find result directories (CV10-only mode)
BASELINE_DIR="outputs/phase1_baseline/results"
UPSTAR_DIR="outputs/phase4_upstar/results"

if [ ! -f "$BASELINE_DIR/cv_results.json" ]; then
    echo "ERROR: Baseline results not found"
    echo "Expected: $BASELINE_DIR/cv_results.json"
    echo ""
    echo "Please run baseline first:"
    echo "  bash scripts/run_tafeng_baseline.sh"
    exit 1
fi

if [ ! -f "$UPSTAR_DIR/cv_results.json" ]; then
    echo "ERROR: UPSTAR results not found"
    echo "Expected: $UPSTAR_DIR/cv_results.json"
    echo ""
    echo "Please run UPSTAR first:"
    echo "  bash scripts/run_tafeng_upstar.sh"
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
import json
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
