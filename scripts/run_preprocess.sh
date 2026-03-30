#!/bin/bash
# Data Preprocessing for Ta-Feng Dataset (Phase 0)
#
# Paper-Aligned Version (2026-03-25):
#   - Preserves timestamps in sequences: [(item, timestamp), ...]
#   - Required for day-level item-time graph construction
#
# Usage:
#   bash scripts/run_preprocess.sh              # Default Ta-Feng
#   bash scripts/run_preprocess.sh [config]    # Custom config

set -e

echo "============================================================"
echo "Ta-Feng Data Preprocessing (Phase 0)"
echo "Paper-Aligned: Preserving timestamps for day-level time nodes"
echo "============================================================"

# Default config
CONFIG="${1:-configs/tafeng_baseline.yaml}"

echo ""
echo "Configuration:"
echo "  Config: $CONFIG"
echo ""

# Check config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

# Check raw data
RAW_DATA=$(python -c "
import yaml
from pathlib import Path
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)
data_dir = Path(config['dataset']['data_dir'])
# Check for ta_feng.csv or ta_feng.txt
if (data_dir / 'ta_feng.csv').exists():
    print(data_dir / 'ta_feng.csv')
elif (data_dir / 'ta_feng.txt').exists():
    print(data_dir / 'ta_feng.txt')
else:
    print('NOT_FOUND')
" 2>/dev/null)

if [ "$RAW_DATA" = "NOT_FOUND" ]; then
    echo "ERROR: Raw Ta-Feng data not found!"
    echo "Expected location: data/raw/ta_feng.csv"
    echo ""
    echo "Please download Ta-Feng dataset and place it in data/raw/"
    exit 1
fi

echo "Found raw data: $RAW_DATA"
echo ""

# Create output directory
PROCESSED_DIR=$(python -c "
import yaml
from pathlib import Path
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)
print(Path(config['dataset']['processed_dir']))
" 2>/dev/null)

echo "Output directory: $PROCESSED_DIR"
echo ""

# Run preprocessing
echo "Step 1: Loading and preprocessing raw data..."
echo "------------------------------------------------------------"

python - <<PYTHON_SCRIPT
import sys
import yaml
from pathlib import Path
import pandas as pd
import pickle

sys.path.insert(0, str(Path.cwd()))

from src.data.preprocess import Preprocessor
from src.data.build_sequences import SequenceBuilder

# Load config
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)

print(f"Dataset: {config['dataset']['name']}")
print(f"Raw data: data/raw/ta_feng.csv")

# Step 1: Preprocess raw data
print("\n[Step 1] Preprocessing raw data...")
preprocessor = Preprocessor(config)
df = preprocessor.run()

# Step 2: Build sequences
print("\n[Step 2] Building sequences (with timestamps)...")
print("  New format: [(item, timestamp), ...]")
print("  This is required for day-level item-time graph (Section 7.1)")
builder = SequenceBuilder(config)
builder.run(df)

print("\n" + "=" * 60)
print("Preprocessing Complete!")
print("=" * 60)
print("\nOutput files:")
processed_dir = Path('$PROCESSED_DIR')
for f in processed_dir.glob('*'):
    if f.is_file():
        size = f.stat().st_size
        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
        print(f"  {f.name}: {size_str}")

print("\n✓ NEW: Sequences now include timestamps!")
print("  Format: [(item_idx, timestamp), ...]")
print("  Required for: day-level item-time graph (Module 1)")

print("\nNext steps:")
print("  1. Item Representations: bash scripts/run_item_repr.sh")
print("  2. STB Calculation: bash scripts/run_stb.sh")
print("  3. Baseline Training: bash scripts/run_tafeng_baseline.sh")
print("  4. UPSTAR Training: bash scripts/run_tafeng_upstar.sh")

print("\n⚠️  IMPORTANT:")
print("  All subsequent phases must be re-run due to paper-aligned improvements:")
print("  - Module 1: Item-Time Graph uses day-level time nodes")
print("  - Module 2: Item-GNN uses in/out neighbor separation")
print("  - Module 3: STB uses mutual information estimation")
print("  - Module 4: Sequence models use dual teacher-student")
print("  - Module 5: All hyperparameters aligned with Section 7")
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Preprocessing successful!"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Preprocessing failed! Check error messages above."
    echo "============================================================"
    exit 1
fi
