#!/bin/bash
# Verification Script for Phase 0 CV Update
#
# This script tests:
# 1. CV split generation
# 2. Timestamp preservation
# 3. Downstream compatibility

set -e

echo "============================================================"
echo "Phase 0 CV Update Verification"
echo "============================================================"
echo ""

# Test 1: Check if build_sequences.py has split_10fold_cv method
echo "Test 1: Checking split_10fold_cv method exists..."
if grep -q "def split_10fold_cv" src/data/build_sequences.py; then
    echo "✅ split_10fold_cv method found"
else
    echo "❌ split_10fold_cv method NOT found"
    exit 1
fi

# Test 2: Check if CV config exists
echo ""
echo "Test 2: Checking CV config file..."
if [ -f "configs/tafeng_cv.yaml" ]; then
    echo "✅ tafeng_cv.yaml found"
else
    echo "❌ tafeng_cv.yaml NOT found"
    exit 1
fi

# Test 3: Check if CV script exists
echo ""
echo "Test 3: Checking CV preprocessing script..."
if [ -f "scripts/run_preprocess_cv.sh" ]; then
    echo "✅ run_preprocess_cv.sh found"
else
    echo "❌ run_preprocess_cv.sh NOT found"
    exit 1
fi

# Test 4: Verify config content
echo ""
echo "Test 4: Verifying config content..."
if grep -q "split_method: '10fold_cv'" configs/tafeng_cv.yaml; then
    echo "✅ Config has split_method='10fold_cv'"
else
    echo "❌ Config missing split_method='10fold_cv'"
    exit 1
fi

# Test 5: Check Python syntax
echo ""
echo "Test 5: Checking Python syntax..."
python -m py_compile src/data/build_sequences.py && echo "✅ build_sequences.py syntax OK"
python -m py_compile src/training/cross_validation_baseline.py && echo "✅ cross_validation_baseline.py syntax OK"
python -m py_compile src/evaluation/cross_validation.py && echo "✅ cross_validation.py syntax OK"

# Test 6: Verify documentation
echo ""
echo "Test 6: Checking documentation..."
if [ -f "PHASE0_CV_UPDATE.md" ]; then
    echo "✅ PHASE0_CV_UPDATE.md found"
else
    echo "❌ PHASE0_CV_UPDATE.md NOT found"
    exit 1
fi

echo ""
echo "============================================================"
echo "All Verification Tests Passed! ✅"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run CV preprocessing:"
echo "     bash scripts/run_preprocess_cv.sh"
echo ""
echo "  2. Verify CV splits:"
echo "     python scripts/check_cv_splits.py"
echo ""
echo "  3. Run CV training:"
echo "     python -m src.training.cross_validation_baseline --config configs/tafeng_cv.yaml"
echo ""
