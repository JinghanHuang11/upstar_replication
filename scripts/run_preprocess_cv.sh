#!/bin/bash
# Data Preprocessing for Ta-Feng Dataset (Phase 0) - 10-Fold CV Mode
#
# Paper-Aligned Version (2026-03-26):
#   - Uses 10-fold cross-validation (user-level split)
#   - Preserves timestamps in sequences: [(item, timestamp), ...]
#   - Required for day-level item-time graph construction
#
# Paper Reference: UPSTAR, Section 7 (Experimental Setup)
#
# Usage:
#   bash scripts/run_preprocess_cv.sh             # Default 10-fold CV
#   bash scripts/run_preprocess_cv.sh [config]    # Custom config

set -e

echo "============================================================"
echo "Ta-Feng Data Preprocessing (Phase 0) - 10-Fold CV Mode"
echo "Paper-Aligned: 10-fold cross-validation with timestamps"
echo "============================================================"

# Default config for 10-fold CV
CONFIG="${1:-configs/tafeng_cv.yaml}"

echo ""
echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Split Method: 10-fold cross-validation (user-level)"
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

# Run preprocessing
echo "Step 1: Loading and preprocessing raw data..."
echo "------------------------------------------------------------"

python - <<PYTHON_SCRIPT
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.data.preprocess import Preprocessor
from src.data.build_sequences import SequenceBuilder

# Load config
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)

print(f"Dataset: {config['dataset']['name']}")
print(f"Split method: {config['dataset'].get('split_method', 'leave_one_out')}")

# Step 1: Preprocess raw data
print("\n[Step 1] Preprocessing raw data...")
preprocessor = Preprocessor(config)
df = preprocessor.run()

# Step 2: Build sequences with 10-fold CV
print("\n[Step 2] Building sequences (10-fold CV with timestamps)...")
builder = SequenceBuilder(config)
builder.run(df)

print("\n" + "=" * 60)
print("✓ 10-Fold CV Preprocessing Complete!")
print("=" * 60)
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
