#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for pipeline steps.

Verifies that all steps can be imported and instantiated correctly.
"""

import sys
from pathlib import Path
import io

# Set stdout to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all steps can be imported"""
    print("Testing imports...")

    try:
        from pipeline_steps import (
            Step0Preprocess,
            Step1Baseline,
            Step2ItemRepr,
            Step3Stb,
            Step4Upstar,
            Step5Evaluate
        )
        print("[OK] All steps imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_instantiation():
    """Test that all steps can be instantiated"""
    print("\nTesting instantiation...")

    try:
        from pipeline_steps import (
            Step0Preprocess,
            Step1Baseline,
            Step2ItemRepr,
            Step3Stb,
            Step4Upstar,
            Step5Evaluate
        )

        # Step 0
        step0 = Step0Preprocess()
        print("[OK] Step0Preprocess instantiated")

        # Step 1
        step1 = Step1Baseline(num_folds=1, quick_test=True)
        print("[OK] Step1Baseline instantiated")

        # Step 2
        step2 = Step2ItemRepr()
        print("[OK] Step2ItemRepr instantiated")

        # Step 3
        step3 = Step3Stb()
        print("[OK] Step3Stb instantiated")

        # Step 4
        step4 = Step4Upstar(num_folds=1, quick_test=True)
        print("[OK] Step4Upstar instantiated")

        # Step 5
        step5 = Step5Evaluate()
        print("[OK] Step5Evaluate instantiated")

        print("\n[OK] All steps instantiated successfully")
        return True

    except Exception as e:
        print(f"[FAIL] Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_step():
    """Test BaseStep functionality"""
    print("\nTesting BaseStep...")

    try:
        from pipeline_steps.base_step import BaseStep

        # Create a test step
        class TestStep(BaseStep):
            def check_prerequisites(self):
                return True

            def check_output_exists(self):
                return False

            def run(self):
                return True

        step = TestStep()
        print("[OK] BaseStep can be subclassed")

        # Test execution
        success = step.execute()
        if success:
            print("[OK] BaseStep.execute() works")
        else:
            print("[FAIL] BaseStep.execute() failed")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] BaseStep test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_help_messages():
    """Test that help messages work"""
    print("\nTesting help messages...")

    import subprocess

    steps = [
        'pipeline_steps.step0_preprocess',
        'pipeline_steps.step1_baseline',
        'pipeline_steps.step2_item_repr',
        'pipeline_steps.step3_stb',
        'pipeline_steps.step4_upstar',
        'pipeline_steps.step5_evaluate',
    ]

    for step in steps:
        try:
            result = subprocess.run(
                ['python', '-m', step, '--help'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"[OK] {step} --help works")
            else:
                print(f"[FAIL] {step} --help failed")
                return False
        except Exception as e:
            print(f"[FAIL] {step} --help error: {e}")
            return False

    return True


def main():
    """Run all tests"""
    print("=" * 80)
    print("Pipeline Steps Test Suite")
    print("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("Instantiation", test_instantiation),
        ("BaseStep", test_base_step),
        ("Help Messages", test_help_messages),
    ]

    results = {}
    for name, test_func in tests:
        print(f"\n{'=' * 80}")
        print(f"Test: {name}")
        print('=' * 80)
        results[name] = test_func()

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")

    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("[SUCCESS] All tests passed!")
        print("=" * 80)
        print("\nYou can now use the pipeline steps:")
        print("  python run_pipeline.py")
        print("  python -m pipeline_steps.step0_preprocess")
        print("  etc.")
        return 0
    else:
        print("[FAILURE] Some tests failed")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
