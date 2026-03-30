#!/usr/bin/env python
"""
UPSTAR Full Experiment Pipeline (CV10-ONLY)

Clean orchestration script that runs the complete UPSTAR experiment pipeline.
Directly calls the current CV10-ONLY training implementations.

Phase Structure:
  Phase 0: Data preprocessing (CV10 splits)
  Phase 1: Baseline training (10-fold CV)
  Phase 2: Item representation learning (Item-GNN)
  Phase 3: STB computation (day-level time nodes)
  Phase 4: UPSTAR training (10-fold CV)
  Phase 5: Evaluation and comparison

Usage:
  # Run complete pipeline
  python run_full_experiment.py

  # Start from specific phase
  python run_full_experiment.py --start-phase 2

  # Run specific phases
  python run_full_experiment.py --start-phase 1 --stop-phase 3

  # Clean old data before running
  python run_full_experiment.py --clean-old-data

  # Force rerun (skip completed phase checks)
  python run_full_experiment.py --force-rerun
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
import shutil
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('outputs/experiment.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run complete UPSTAR experiment pipeline (CV10-ONLY)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete pipeline
  python run_full_experiment.py

  # Start from Phase 2 (Item representations)
  python run_full_experiment.py --start-phase 2

  # Run only Phases 2-3
  python run_full_experiment.py --start-phase 2 --stop-phase 3

  # Clean old data and run complete pipeline
  python run_full_experiment.py --clean-old-data

Phases:
  0: Data preprocessing (CV10 splits)
  1: Baseline training (10-fold CV, ~20-40 hours)
  2: Item representation learning (Item-GNN)
  3: STB computation (day-level time nodes)
  4: UPSTAR training (10-fold CV, ~80-160 hours)
  5: Evaluation and comparison

IMPORTANT: All phases use CV10-ONLY mode (10-fold cross-validation)
        """
    )

    parser.add_argument(
        '--start-phase',
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=0,
        help='Phase to start from (default: 0)'
    )

    parser.add_argument(
        '--stop-phase',
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=5,
        help='Phase to stop at (default: 5)'
    )

    parser.add_argument(
        '--clean-old-data',
        action='store_true',
        help='Clean old processed data, cache, and checkpoints before running'
    )

    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Force rerun all phases (skip completion checks)'
    )

    parser.add_argument(
        '--baseline-config',
        type=str,
        default='configs/tafeng_baseline.yaml',
        help='Baseline config file (default: configs/tafeng_baseline.yaml)'
    )

    parser.add_argument(
        '--upstar-config',
        type=str,
        default='configs/tafeng_upstar.yaml',
        help='UPSTAR config file (default: configs/tafeng_upstar.yaml)'
    )

    return parser.parse_args()


def clean_old_data():
    """Clean old processed data, cache, and checkpoints"""
    logger.info("=" * 80)
    logger.info("Cleaning Old Data")
    logger.info("=" * 80)

    directories_to_clean = [
        'data/processed/tafeng',
        'data/cache/tafeng',
        'outputs/phase2_item_repr',
        'outputs/phase3_stb',
        'outputs/phase4_upstar'
    ]

    for dir_path in directories_to_clean:
        path = Path(dir_path)
        if path.exists():
            logger.info(f"Removing: {dir_path}")
            try:
                shutil.rmtree(path)
                logger.info(f"  ✓ Removed: {dir_path}")
            except Exception as e:
                logger.warning(f"  ✗ Failed to remove {dir_path}: {e}")
        else:
            logger.info(f"  - Not found (skipped): {dir_path}")

    logger.info("✓ Old data cleaning complete!")


def check_phase_completion(phase_num: int) -> bool:
    """Check if a phase has been completed"""
    phase_outputs = {
        0: ['data/processed/tafeng/metadata.pkl'],
        1: ['outputs/phase1_baseline/results/cv_results.json'],
        2: ['outputs/phase2_item_repr/checkpoints/item_embeddings.pt'],
        3: ['outputs/phase3_stb/checkpoints/motivation_labels.npy'],
        4: ['outputs/phase4_upstar/results/cv_results.json'],
        5: ['outputs/comparison/comparison.txt']
    }

    required_files = phase_outputs.get(phase_num, [])
    all_exist = all(Path(f).exists() for f in required_files)

    if all_exist:
        logger.info(f"  ✓ Phase {phase_num} already completed (outputs found)")
        return True
    else:
        logger.info(f"  → Phase {phase_num} needs to run")
        return False


def run_command(cmd: list, description: str):
    """Run a command and log output"""
    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        # Log output
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False


def run_phase0(args):
    """Phase 0: Data preprocessing"""
    logger.info("=" * 80)
    logger.info("Phase 0: Data Preprocessing (CV10 Splits)")
    logger.info("=" * 80)

    cmd = ['bash', 'scripts/run_preprocess.sh']
    return run_command(cmd, "Data preprocessing")


def run_phase1(args):
    """Phase 1: Baseline training (10-fold CV)"""
    logger.info("=" * 80)
    logger.info("Phase 1: Baseline Training (10-Fold CV)")
    logger.info("=" * 80)
    logger.info("Estimated time: 20-40 hours")

    cmd = ['bash', 'scripts/run_tafeng_baseline.sh', args.baseline_config]
    return run_command(cmd, "Baseline training")


def run_phase2(args):
    """Phase 2: Item representation learning"""
    logger.info("=" * 80)
    logger.info("Phase 2: Item Representation Learning (Item-GNN)")
    logger.info("=" * 80)

    cmd = ['bash', 'scripts/run_item_repr.sh']
    return run_command(cmd, "Item representation learning")


def run_phase3(args):
    """Phase 3: STB computation"""
    logger.info("=" * 80)
    logger.info("Phase 3: STB Computation (Day-Level Time Nodes)")
    logger.info("=" * 80)

    cmd = ['bash', 'scripts/run_stb.sh']
    return run_command(cmd, "STB computation")


def run_phase4(args):
    """Phase 4: UPSTAR training (10-fold CV)"""
    logger.info("=" * 80)
    logger.info("Phase 4: UPSTAR Training (10-Fold CV)")
    logger.info("=" * 80)
    logger.info("Estimated time: 80-160 hours")

    cmd = ['bash', 'scripts/run_tafeng_upstar.sh', args.upstar_config]
    return run_command(cmd, "UPSTAR training")


def run_phase5(args):
    """Phase 5: Evaluation and comparison"""
    logger.info("=" * 80)
    logger.info("Phase 5: Evaluation and Comparison")
    logger.info("=" * 80)

    cmd = ['bash', 'scripts/compare_results.sh']
    return run_command(cmd, "Results comparison")


def main():
    """Main entry point"""
    args = parse_args()

    # Print experiment info
    logger.info("=" * 80)
    logger.info("UPSTAR Full Experiment Pipeline (CV10-ONLY)")
    logger.info("=" * 80)
    logger.info(f"Start phase: {args.start_phase}")
    logger.info(f"Stop phase: {args.stop_phase}")
    logger.info(f"Force rerun: {args.force_rerun}")
    logger.info(f"Clean old data: {args.clean_old_data}")
    logger.info("=" * 80)
    logger.info("")

    # Clean old data if requested
    if args.clean_old_data:
        clean_old_data()
        logger.info("")

    # Phase functions
    phase_functions = {
        0: run_phase0,
        1: run_phase1,
        2: run_phase2,
        3: run_phase3,
        4: run_phase4,
        5: run_phase5
    }

    phase_descriptions = {
        0: "Data Preprocessing (CV10 Splits)",
        1: "Baseline Training (10-Fold CV, ~20-40 hours)",
        2: "Item Representation Learning (Item-GNN)",
        3: "STB Computation (Day-Level Time Nodes)",
        4: "UPSTAR Training (10-Fold CV, ~80-160 hours)",
        5: "Evaluation and Comparison"
    }

    # Run phases
    start_time = time.time()
    completed_phases = []
    failed_phases = []

    for phase_num in range(args.start_phase, args.stop_phase + 1):
        phase_func = phase_functions[phase_num]
        phase_desc = phase_descriptions[phase_num]

        logger.info("")
        logger.info(f"Starting Phase {phase_num}: {phase_desc}")

        # Check if already completed (skip if force-rerun is False)
        if not args.force_rerun and check_phase_completion(phase_num):
            logger.info(f"Skipping Phase {phase_num} (already completed)")
            completed_phases.append(phase_num)
            continue

        # Run phase
        try:
            success = phase_func(args)

            if success:
                logger.info(f"✓ Phase {phase_num} completed successfully")
                completed_phases.append(phase_num)
            else:
                logger.error(f"✗ Phase {phase_num} failed")
                failed_phases.append(phase_num)
                break

        except Exception as e:
            logger.error(f"✗ Phase {phase_num} failed with exception: {e}")
            failed_phases.append(phase_num)
            break

    # Print summary
    elapsed_time = time.time() - start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("Experiment Summary")
    logger.info("=" * 80)
    logger.info(f"Elapsed time: {elapsed_time / 3600:.2f} hours")
    logger.info(f"Completed phases: {completed_phases}")
    logger.info(f"Failed phases: {failed_phases}")

    if failed_phases:
        logger.error(f"Experiment failed at phase {failed_phases[0]}")
        logger.info(f"To resume, run: python run_full_experiment.py --start-phase {failed_phases[0]}")
        sys.exit(1)
    else:
        logger.info("✅ All phases completed successfully!")
        logger.info("")
        logger.info("Results saved to:")
        logger.info("  - Phase 1 (Baseline): outputs/phase1_baseline/results/")
        logger.info("  - Phase 4 (UPSTAR): outputs/phase4_upstar/results/")
        logger.info("  - Comparison: outputs/comparison/")
        sys.exit(0)


if __name__ == '__main__':
    main()
