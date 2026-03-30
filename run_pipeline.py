#!/usr/bin/env python
"""
UPSTAR Pipeline Runner

Runs the complete UPSTAR experiment pipeline using modular step-based approach.
Each step can be run independently or as part of the full pipeline.

Usage:
    # Run complete pipeline
    python run_pipeline.py

    # Run specific steps
    python run_pipeline.py --steps 0 1 2 3 4 5

    # Start from specific step
    python run_pipeline.py --start-step 2

    # Stop at specific step
    python run_pipeline.py --stop-step 3

    # Quick test mode
    python run_pipeline.py --quick-test

Examples:
    # Run all steps
    python run_pipeline.py

    # Run only preprocessing and baseline
    python run_pipeline.py --steps 0 1

    # Run from step 2 onwards
    python run_pipeline.py --start-step 2

    # Quick test (1 fold, minimal epochs)
    python run_pipeline.py --quick-test
"""

import argparse
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('outputs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run UPSTAR experiment pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Step Numbers:
  0: Data preprocessing (CV10 splits)
  1: Baseline training (10-fold CV)
  2: Item representation learning (Item-GNN)
  3: STB computation (day-level time nodes)
  4: UPSTAR training (10-fold CV)
  5: Evaluation and comparison

Examples:
  # Run all steps
  python run_pipeline.py

  # Run specific steps
  python run_pipeline.py --steps 0 1 2

  # Start from step 2
  python run_pipeline.py --start-step 2

  # Run steps 2-4
  python run_pipeline.py --start-step 2 --stop-step 4

  # Quick test (1 fold, minimal epochs)
  python run_pipeline.py --quick-test

  # Skip existing outputs
  python run_pipeline.py --skip-existing
        """
    )

    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        choices=[0, 1, 2, 3, 4, 5],
        help='Specific steps to run (default: all steps 0-5)'
    )

    parser.add_argument(
        '--start-step',
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=0,
        help='Start from this step (default: 0)'
    )

    parser.add_argument(
        '--stop-step',
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=5,
        help='Stop at this step (default: 5)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode: 1 fold, minimal epochs'
    )

    parser.add_argument(
        '--num-folds',
        type=int,
        default=10,
        help='Number of folds for cross-validation (default: 10)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip steps that already have output files'
    )

    parser.add_argument(
        '--baseline-config',
        type=str,
        default='configs/tafeng_baseline.yaml',
        help='Baseline config file'
    )

    parser.add_argument(
        '--upstar-config',
        type=str,
        default='configs/tafeng_upstar.yaml',
        help='UPSTAR config file'
    )

    return parser.parse_args()


def get_step_class(step_num: int, quick_test: bool = False, num_folds: int = 10):
    """Get the step class for a given step number"""

    if step_num == 0:
        from pipeline_steps.step0_preprocess import Step0Preprocess
        return Step0Preprocess

    elif step_num == 1:
        from pipeline_steps.step1_baseline import Step1Baseline
        return lambda: Step1Baseline(
            config_path='configs/tafeng_baseline.yaml',
            num_folds=num_folds,
            quick_test=quick_test
        )

    elif step_num == 2:
        from pipeline_steps.step2_item_repr import Step2ItemRepr
        return Step2ItemRepr

    elif step_num == 3:
        from pipeline_steps.step3_stb import Step3Stb
        return Step3Stb

    elif step_num == 4:
        from pipeline_steps.step4_upstar import Step4Upstar
        return lambda: Step4Upstar(
            config_path='configs/tafeng_upstar.yaml',
            num_folds=num_folds,
            quick_test=quick_test
        )

    elif step_num == 5:
        from pipeline_steps.step5_evaluate import Step5Evaluate
        return Step5Evaluate

    else:
        raise ValueError(f"Invalid step number: {step_num}")


def main():
    """Main entry point"""
    args = parse_args()

    # Determine which steps to run
    if args.steps:
        # Specific steps provided
        steps_to_run = sorted(set(args.steps))
    else:
        # Run from start to stop
        steps_to_run = list(range(args.start_step, args.stop_step + 1))

    logger.info("=" * 80)
    logger.info("UPSTAR Pipeline Runner")
    logger.info("=" * 80)
    logger.info(f"Steps to run: {steps_to_run}")
    logger.info(f"Quick test: {args.quick_test}")
    logger.info(f"Num folds: {args.num_folds}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("=" * 80)
    logger.info("")

    # Step names
    step_names = {
        0: "Data Preprocessing",
        1: "Baseline Training",
        2: "Item Representation Learning",
        3: "STB Computation",
        4: "UPSTAR Training",
        5: "Evaluation and Comparison"
    }

    # Run steps
    completed_steps = []
    failed_steps = []
    skipped_steps = []

    for step_num in steps_to_run:
        step_name = step_names[step_num]
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Step {step_num}: {step_name}")
        logger.info("=" * 80)

        try:
            # Get step class
            step_class_func = get_step_class(step_num, args.quick_test, args.num_folds)
            step = step_class_func() if callable(step_class_func) else step_class_func

            # Check if output exists
            if args.skip_existing and step.check_output_exists():
                logger.info(f"Skipping Step {step_num} (output already exists)")
                skipped_steps.append(step_num)
                continue

            # Run step
            success = step.execute()

            if success:
                logger.info(f"✓ Step {step_num} completed successfully")
                completed_steps.append(step_num)
            else:
                logger.error(f"✗ Step {step_num} failed")
                failed_steps.append(step_num)
                break

        except Exception as e:
            logger.error(f"✗ Step {step_num} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed_steps.append(step_num)
            break

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Pipeline Summary")
    logger.info("=" * 80)
    logger.info(f"Completed steps: {completed_steps}")
    logger.info(f"Skipped steps: {skipped_steps}")
    logger.info(f"Failed steps: {failed_steps}")

    if failed_steps:
        logger.error(f"Pipeline failed at step {failed_steps[0]}")
        logger.info(f"To resume, run: python run_pipeline.py --start-step {failed_steps[0]}")
        sys.exit(1)
    else:
        logger.info("✅ All steps completed successfully!")
        logger.info("")
        logger.info("Results saved to:")
        logger.info("  - Phase 1 (Baseline): outputs/phase1_baseline/results/")
        logger.info("  - Phase 4 (UPSTAR): outputs/phase4_upstar/results/")
        logger.info("  - Comparison: outputs/comparison/")
        sys.exit(0)


if __name__ == '__main__':
    main()
