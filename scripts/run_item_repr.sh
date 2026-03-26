#!/bin/bash
# Train Item Representations (Phase 2)
#
# Paper-Aligned Version (2026-03-25):
#   - Item-GNN with in/out neighbor separation (Section 3.2)
#   - Architecture: h_n^{g(k)} = ReLU(W_in · Σ_in + W_out · Σ_out)
#
# Usage:
#   bash scripts/run_item_repr.sh [config]

set -e

echo "============================================================"
echo "Item Representation Learning (Phase 2)"
echo "Paper-Aligned: In/Out Neighbor Separation"
echo "============================================================"

# Configuration
CONFIG=${1:-configs/item_repr.yaml}

echo "Using config: $CONFIG"
echo ""

# Check Phase 0
if [ ! -f "data/processed/tafeng/metadata.pkl" ]; then
    echo "ERROR: Phase 0 data not found!"
    echo "Run: bash scripts/run_preprocess.sh"
    exit 1
fi

echo "✓ Phase 0 data found (with timestamps)"
echo ""

# Step 1: Build item graph
echo "Step 1: Building item graph..."
echo "  - In-session graph: co-occurrence within sessions"
echo "  - Cross-session graph: transitions across sessions"
echo "  - Compatible with new sequence format: [(item, ts), ...]"
echo ""

python -c "
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.graphs.item_graph import ItemGraphBuilder
import pickle

# Load config
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Build graph
builder = ItemGraphBuilder(config)

# Try loading from cache
graph = builder.load_graph('item_graph.pt')

if graph is None:
    print('Building item graph (with new sequence format)...')
    train_sequences = builder.load_sequences('train')
    user_sessions = builder.build_session_sequences(train_sequences)

    in_session_graph = builder.build_in_session_graph(user_sessions)
    cross_session_graph = builder.build_cross_session_graph(user_sessions)

    graph = builder.build_combined_graph(in_session_graph, cross_session_graph)

    builder.save_graph(graph, 'item_graph.pt')
    builder.save_graph(in_session_graph, 'in_session_graph.pt')
    builder.save_graph(cross_session_graph, 'cross_session_graph.pt')

    # Compute statistics
    with open(Path(config['dataset']['processed_dir']) / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    stats = builder.compute_graph_statistics(graph, metadata['num_items'])

    print()
    print('=' * 60)
    print('Item Graph Statistics')
    print('=' * 60)
    for key, value in stats.items():
        print(f'{key:20s}: {value}')
else:
    print('Item graph loaded from cache')
    print('  Note: To apply paper-aligned improvements, clear cache first:')
    print('        rm -rf data/cache/tafeng/*')
"

# Step 2: Train Item-GNN
echo ''
echo "Step 2: Training Item-GNN..."
echo "  Paper-Aligned Architecture (Section 3.2):"
echo "    h_n^{g(k)} = ReLU(W_in · Σ_in_neighbors + W_out · Σ_out_neighbors)"
echo "  - Hidden dim: 128 (Section 7.2)"
echo "  - Layers: 1 (Section 7.2)"
echo ""

python src/training/train_item_repr.py --config $CONFIG

echo ""
echo "============================================================"
echo "Item Representation Learning Complete!"
echo "============================================================"
echo ""
echo "Output: outputs/phase2_item_repr/checkpoints/item_embeddings.pt"
echo ""
echo "✓ Paper-Aligned: In/out neighbor separation applied"
echo ""
echo "Next step:"
echo "  STB Calculation: bash scripts/run_stb.sh"
