#!/bin/bash
# Run Advanced STB Computation (Phase 6)

set -e

echo "============================================================"
echo "Advanced STB Computation (Phase 6)"
echo "============================================================"

CONFIG="${1:-configs/stb_advanced.yaml}"
shift

# Check prerequisites
echo ""
echo "Checking prerequisites..."

# Check config
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi
echo "  [OK] Config: $CONFIG"

# Check Phase 2: Item embeddings
if [ ! -f "outputs/phase2_item_repr/checkpoints/item_embeddings.pt" ]; then
    echo "  [FAIL] Item embeddings not found"
    echo "        Run: bash scripts/run_item_repr.sh"
    exit 1
fi
echo "  [OK] Phase 2: Item representations learned"

echo ""
echo "All prerequisites satisfied!"
echo ""

# Create output directories
mkdir -p outputs/logs/stb_advanced
mkdir -p outputs/phase3_stb_advanced/checkpoints

# Run
echo "Starting advanced STB computation..."
echo "============================================================"
echo ""

python src/training/train_stb_advanced.py --config "$CONFIG" "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Advanced STB computation complete!"
    echo "============================================================"
    echo ""
    echo "Results saved to: outputs/phase3_stb_advanced/checkpoints/"
    echo ""
    echo "Version comparison:"
    ls -lh outputs/phase3_stb_advanced/checkpoints/*/motivation_labels.npy 2>/dev/null || echo "  (No results found)"
else
    echo ""
    echo "============================================================"
    echo "Computation failed! Check logs for details."
    echo "============================================================"
    exit 1
fi
