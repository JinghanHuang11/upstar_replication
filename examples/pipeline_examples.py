#!/usr/bin/env python
"""
Pipeline Steps Usage Examples

This file demonstrates various ways to use the modular pipeline steps.
"""

# Example 1: Run complete pipeline
def example1_complete_pipeline():
    """Example 1: Run complete pipeline using subprocess"""
    import subprocess

    print("Example 1: Running complete pipeline...")
    print("=" * 80)

    # Quick test mode
    result = subprocess.run(
        ['python', 'run_pipeline.py', '--quick-test'],
        check=True
    )

    if result.returncode == 0:
        print("✓ Pipeline completed successfully")


# Example 2: Run specific steps
def example2_specific_steps():
    """Example 2: Run only steps 0, 1, 2"""
    import subprocess

    print("Example 2: Running steps 0, 1, 2...")
    print("=" * 80)

    result = subprocess.run(
        ['python', 'run_pipeline.py', '--steps', '0', '1', '2'],
        check=True
    )

    if result.returncode == 0:
        print("✓ Steps 0, 1, 2 completed successfully")


# Example 3: Run steps independently
def example3_independent_steps():
    """Example 3: Run each step independently"""
    import subprocess

    print("Example 3: Running steps independently...")
    print("=" * 80)

    steps = [
        ('step0_preprocess', []),
        ('step1_baseline', ['--quick-test']),
        ('step2_item_repr', []),
        ('step3_stb', []),
        ('step4_upstar', ['--quick-test']),
        ('step5_evaluate', []),
    ]

    for step_name, args in steps:
        print(f"\nRunning: {step_name}")
        cmd = ['python', '-m', f'pipeline_steps.{step_name}'] + args
        result = subprocess.run(cmd, check=True)
        print(f"✓ {step_name} completed")


# Example 4: Use steps in Python code
def example4_python_usage():
    """Example 4: Import and use steps in Python code"""
    from pipeline_steps import (
        Step0Preprocess,
        Step1Baseline,
        Step2ItemRepr
    )

    print("Example 4: Using steps in Python code...")
    print("=" * 80)

    # Step 0: Preprocessing
    print("\nStep 0: Preprocessing...")
    step0 = Step0Preprocess()
    success0 = step0.execute()

    if not success0:
        print("✗ Step 0 failed")
        return

    # Step 1: Baseline (quick test)
    print("\nStep 1: Baseline training (quick test)...")
    step1 = Step1Baseline(num_folds=1, quick_test=True)
    success1 = step1.execute()

    if not success1:
        print("✗ Step 1 failed")
        return

    # Step 2: Item representations
    print("\nStep 2: Item representation learning...")
    step2 = Step2ItemRepr()
    success2 = step2.execute()

    if not success2:
        print("✗ Step 2 failed")
        return

    print("\n✓ All steps completed successfully")


# Example 5: Custom step sequence
def example5_custom_sequence():
    """Example 5: Run custom sequence of steps"""
    import subprocess

    print("Example 5: Running custom step sequence...")
    print("=" * 80)

    # Run preprocessing, then skip to evaluation
    print("Running: Step 0 → Step 5")
    result = subprocess.run(
        ['python', 'run_pipeline.py', '--steps', '0', '5'],
        check=True
    )

    if result.returncode == 0:
        print("✓ Custom sequence completed successfully")


# Example 6: Resume from failed step
def example6_resume_from_step():
    """Example 6: Resume from a specific step"""
    import subprocess

    print("Example 6: Resuming from Step 3...")
    print("=" * 80)

    # If pipeline failed at step 3, resume from there
    result = subprocess.run(
        ['python', 'run_pipeline.py', '--start-step', '3'],
        check=True
    )

    if result.returncode == 0:
        print("✓ Resumed and completed successfully")


# Example 7: Skip existing outputs
def example7_skip_existing():
    """Example 7: Skip steps that already have outputs"""
    import subprocess

    print("Example 7: Skipping existing outputs...")
    print("=" * 80)

    # Only run steps that don't have output yet
    result = subprocess.run(
        ['python', 'run_pipeline.py', '--skip-existing'],
        check=True
    )

    if result.returncode == 0:
        print("✓ Pipeline completed (skipped existing outputs)")


# Example 8: Error handling
def example8_error_handling():
    """Example 8: Handle errors gracefully"""
    from pipeline_steps import Step0Preprocess

    print("Example 8: Error handling...")
    print("=" * 80)

    step = Step0Preprocess()

    try:
        success = step.execute()
        if success:
            print("✓ Step completed successfully")
        else:
            print("✗ Step failed (but no exception)")
    except Exception as e:
        print(f"✗ Step failed with exception: {e}")
        # Handle error, log it, or retry


# Example 9: Check prerequisites before running
def example9_check_prerequisites():
    """Example 9: Check prerequisites before running"""
    from pipeline_steps import Step3Stb

    print("Example 9: Checking prerequisites...")
    print("=" * 80)

    step = Step3Stb()

    # Check prerequisites without running
    if step.check_prerequisites():
        print("✓ All prerequisites satisfied")
        # Now run the step
        success = step.execute()
    else:
        print("✗ Prerequisites not satisfied")
        print("Please run previous steps first")


# Example 10: Custom configuration
def example10_custom_config():
    """Example 10: Use custom configuration files"""
    import subprocess

    print("Example 10: Using custom configurations...")
    print("=" * 80)

    result = subprocess.run([
        'python', 'run_pipeline.py',
        '--baseline-config', 'configs/tafeng_baseline.yaml',
        '--upstar-config', 'configs/tafeng_upstar.yaml',
        '--num-folds', '2'
    ], check=True)

    if result.returncode == 0:
        print("✓ Pipeline completed with custom config")


def main():
    """Run examples"""
    import sys

    examples = {
        '1': ('Complete Pipeline', example1_complete_pipeline),
        '2': ('Specific Steps', example2_specific_steps),
        '3': ('Independent Steps', example3_independent_steps),
        '4': ('Python Usage', example4_python_usage),
        '5': ('Custom Sequence', example5_custom_sequence),
        '6': ('Resume from Step', example6_resume_from_step),
        '7': ('Skip Existing', example7_skip_existing),
        '8': ('Error Handling', example8_error_handling),
        '9': ('Check Prerequisites', example9_check_prerequisites),
        '10': ('Custom Config', example10_custom_config),
    }

    print("=" * 80)
    print("Pipeline Steps Usage Examples")
    print("=" * 80)
    print()

    if len(sys.argv) > 1:
        # Run specific example
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"Running Example {example_num}: {name}")
            func()
        else:
            print(f"Unknown example: {example_num}")
            print("Available examples:", ', '.join(examples.keys()))
    else:
        # List all examples
        print("Available examples:")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        print()
        print("Usage:")
        print("  python examples/pipeline_examples.py <example_number>")
        print()
        print("Example:")
        print("  python examples/pipeline_examples.py 1")


if __name__ == '__main__':
    main()
