#!/bin/bash
# Clean old data and cache before paper-aligned retraining
#
# This script removes all processed data, cache, and checkpoints
# that were generated before the paper-aligned improvements.
#
# Usage:
#   bash scripts/clean_old_data.sh

set -e

echo "============================================================"
echo "Clean Old Data (Paper-Aligned Improvements)"
echo "============================================================"
echo ""
echo "⚠️  WARNING: This will DELETE the following:"
echo "   - Processed data (old format without timestamps)"
echo "   - Graph cache (old time nodes)"
echo "   - Model checkpoints (old architectures)"
echo ""
echo "Required for applying paper-aligned improvements:"
echo "   Module 1: Day-level item-time graph (Section 7.1)"
echo "   Module 2: Item-GNN in/out aggregation (Section 3.2)"
echo "   Module 3: STB mutual information (Section 3.1.3)"
echo "   Module 4: Dual teacher-student (Section 3.3)"
echo "   Module 5: Hyperparameters (Section 7)"
echo ""
echo "============================================================"
echo ""

# Confirm
read -p "Continue with cleanup? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo ""

# Directories to clean
declare -a DIRS_TO_CLEAN=(
    "data/processed/tafeng"
    "data/cache/tafeng"
    "outputs/phase2_item_repr"
    "outputs/phase3_stb"
    "outputs/phase4_upstar"
)

# Clean each directory
for dir_path in "${DIRS_TO_CLEAN[@]}"; do
    if [ -d "$dir_path" ]; then
        echo "Removing: $dir_path"
        rm -rf "$dir_path"
        echo "  ✓ Removed"
    else
        echo "Skipping (not found): $dir_path"
    fi
done

echo ""
echo "============================================================"
echo "✓ Cleanup Complete!"
echo "============================================================"
echo ""
echo "Removed:"
echo "  - Old processed data (without timestamps)"
echo "  - Old graph cache (without day-level time nodes)"
echo "  - Old model checkpoints (without paper-aligned improvements)"
echo ""
echo "Next steps:"
echo "  1. Re-preprocess data: bash scripts/run_preprocess.sh"
echo "  2. Train Item-GNN: bash scripts/run_item_repr.sh"
echo "  3. Compute STB: bash scripts/run_stb.sh"
echo "  4. Train Baseline: bash scripts/run_tafeng_baseline.sh"
echo "  5. Train UPSTAR: bash scripts/run_tafeng_upstar.sh"
echo ""
echo "See RUN_EXPERIMENTS.md for complete experiment guide."
echo ""
