#!/bin/bash
# Compute STB scores and motivation labels (Phase 3)
#
# Paper-Aligned Version (2026-03-25):
#   - Day-level item-time graph (Section 7.1: ρ=50)
#   - Mutual Information estimation (Section 3.1.3)
#   - Perturbation parameters aligned with Section 7.1
#
# Usage:
#   bash scripts/run_stb.sh [config]

set -e

echo "============================================================"
echo "STB Computation (Phase 3)"
echo "Paper-Aligned Improvements:"
echo "  - Day-level time nodes (Section 7.1: ρ=50)"
echo "  - Mutual Information estimation (Section 3.1.3)"
echo "  - Perturbation: α=0.4, ε=0.1, β=40 (Section 7.1)"
echo "============================================================"

# Configuration
CONFIG=${1:-configs/stb.yaml}

echo "Using config: $CONFIG"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Phase 0
if [ ! -f "data/processed/tafeng/metadata.pkl" ]; then
    echo "ERROR: Phase 0 data not found!"
    echo "Run: bash scripts/run_preprocess.sh"
    exit 1
fi
echo "  [OK] Phase 0: Data preprocessing complete"

# Check Phase 2
if [ ! -f "outputs/phase2_item_repr/checkpoints/item_embeddings.pt" ]; then
    echo "ERROR: Phase 2 item embeddings not found!"
    echo "Run: bash scripts/run_item_repr.sh"
    exit 1
fi
echo "  [OK] Phase 2: Item representations learned"

echo ""
echo "All prerequisites satisfied!"
echo ""

# Show STB configuration
python -c "
import yaml
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)

print('STB Configuration (from config):')
print('=' * 60)
stb_cfg = config.get('stb', {})
print(f'  Time granularity: {config[\"model\"][\"hidden_dim\"]}D hidden')
print(f'  Perturbation rounds (β): {stb_cfg.get(\"perturbation_rounds\", \"N/A\")}')
print(f'  Feature step size (α): {stb_cfg.get(\"feature_step_size\", \"N/A\")}')
print(f'  Epsilon (ε): {stb_cfg.get(\"epsilon\", \"N/A\")}')
print(f'  Epsilon X (ε_x): {stb_cfg.get(\"epsilon_x\", \"N/A\")}')
print(f'  Time nodes (��): {stb_cfg.get(\"num_time_nodes\", \"N/A\")}')
print('=' * 60)
"

# Run STB computation
echo ""
echo "Running STB computation..."
echo "  - Building day-level item-time graph"
echo "  - Computing STB scores (with MI estimation option)"
echo "  - Classifying items: stable (top 50%), exploratory (bottom 40%)"
echo ""

python src/training/train_stb.py --config $CONFIG

echo ""
echo "============================================================"
echo "STB Computation Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - outputs/phase3_stb/checkpoints/stb_scores.npy"
echo "  - outputs/phase3_stb/checkpoints/motivation_labels.npy"
echo "  - outputs/phase3_stb/checkpoints/stb_stats.pkl"
echo ""
echo "✓ Paper-Aligned: Day-level time nodes + MI estimation"
echo ""
echo "To visualize results:"
echo "  python -c \"import numpy as np; scores=np.load('outputs/phase3_stb/checkpoints/stb_scores.npy'); print(f'STB scores: mean={scores.mean():.4f}, std={scores.std():.4f}')\""
echo ""
echo "Next step:"
echo "  UPSTAR Training: bash scripts/run_tafeng_upstar.sh"
