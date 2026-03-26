#!/usr/bin/env python
"""
Run Full UPSTAR Experiment

Complete pipeline from data preprocessing to final evaluation

Paper-Aligned Version (2026-03-25):
  - Module 1: Item-Time Graph with day-level time nodes
  - Module 2: Item-GNN with in/out neighbor separation
  - Module 3: STB with mutual information estimation (MINE)
  - Module 4: Sequence models with dual teacher-student
  - Module 5: Hyperparameters aligned with Section 7

Usage:
    # Full experiment (with force rerun to apply paper-aligned improvements)
    python run_full_experiment.py --mode full --force-rerun

    # Quick test (small dataset, few epochs)
    python run_full_experiment.py --mode quick --force-rerun

    # Resume from last incomplete phase
    python run_full_experiment.py --resume

    # Start from specific phase
    python run_full_experiment.py --start-phase 2 --force-rerun
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
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
        description='Run complete UPSTAR experiment pipeline (Paper-Aligned)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full experiment with paper-aligned improvements
  python run_full_experiment.py --mode full --force-rerun

  # Quick test (recommended for testing paper-aligned changes)
  python run_full_experiment.py --mode quick --force-rerun

  # Resume from where it stopped
  python run_full_experiment.py --resume

  # Start from specific phase (0-5)
  python run_full_experiment.py --start-phase 3 --force-rerun

  # Run only specific phases
  python run_full_experiment.py --start-phase 1 --stop-phase 2 --force-rerun

Paper-Aligned Improvements (2026-03-25):
  Module 1: Item-Time Graph - day-level time nodes (Section 7.1)
  Module 2: Item-GNN - in/out neighbor separation (Section 3.2)
  Module 3: STB - mutual information estimation (Section 3.1.3)
  Module 4: Sequence Models - dual teacher-student (Section 3.3)
  Module 5: Hyperparameters - aligned with Section 7

Phases:
  0: Data preprocessing (NEW FORMAT: includes timestamps)
  1: Baseline training
  2: Item representation learning (NEW ARCH: in/out aggregation)
  3: STB calculation (NEW: day-level time nodes)
  4: UPSTAR training (4 stages with dual teacher-student)
  5: Evaluation and comparison

IMPORTANT: Use --force-rerun to apply paper-aligned improvements!
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'full', 'compare-only'],
        default='full',
        help='Experiment mode (default: full)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_tafeng.yaml',
        help='Path to experiment config file'
    )

    parser.add_argument(
        '--start-phase',
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=None,
        help='Phase to start from (0-5)'
    )

    parser.add_argument(
        '--stop-phase',
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=None,
        help='Phase to stop at (0-5)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last incomplete phase'
    )

    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Re-run even if phases are completed (REQUIRED for paper-aligned improvements)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/experiments',
        help='Base output directory for experiments'
    )

    parser.add_argument(
        '--clean-old-data',
        action='store_true',
        help='Clean old processed data and cache before running (recommended for paper-aligned improvements)'
    )

    return parser.parse_args()


def get_mode_config(mode: str) -> str:
    """Get config file based on mode"""
    config_mapping = {
        'quick': 'configs/experiment_tafeng_quick.yaml',
        'full': 'configs/experiment_tafeng.yaml',
        'compare-only': 'configs/experiment_compare_only.yaml'
    }
    return config_mapping.get(mode, 'configs/experiment_tafeng.yaml')


def validate_args(args):
    """Validate command line arguments"""
    # Check if resume conflicts with start-phase
    if args.resume and args.start_phase is not None:
        logger.warning("--resume and --start-phase both specified. Using --start-phase.")
        args.resume = False

    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        logger.info(f"Creating default config from mode: {args.mode}")

        # Create default config if doesn't exist
        create_default_config(args.mode)
        args.config = get_mode_config(args.mode)

    # Warn if force-rerun is not set (for paper alignment)
    if not args.force_rerun and not args.resume:
        logger.warning("")
        logger.warning("[WARNING] PAPER-ALIGNED IMPROVEMENTS NOT APPLIED!")
        logger.warning("   The following modules have been updated to align with the paper:")
        logger.warning("   - Module 1: Item-Time Graph (day-level time nodes)")
        logger.warning("   - Module 2: Item-GNN (in/out neighbor separation)")
        logger.warning("   - Module 3: STB (mutual information estimation)")
        logger.warning("   - Module 4: Sequence Models (dual teacher-student)")
        logger.warning("   - Module 5: Hyperparameters (Section 7)")
        logger.warning("")
        logger.warning("   To apply these improvements, use --force-rerun flag")
        logger.warning("")
        logger.warning("   This will:")
        logger.warning("   - Re-preprocess data (new format with timestamps)")
        logger.warning("   - Rebuild item graph (new in/out aggregation)")
        logger.warning("   - Recompute STB (new day-level time nodes)")
        logger.warning("   - Retrain models (new architectures)")
        logger.warning("")


def create_default_config(mode: str):
    """Create default experiment configuration"""
    config = {}

    if mode == 'quick':
        config = {
            'name': 'upstar_tafeng_quick',
            'dataset': {
                'name': 'tafeng',
                'data_dir': 'data/raw/',
                'processed_dir': 'data/processed/tafeng/',
                'min_interactions': 5,
                'time_window': 86400
            },
            'baseline_config': 'configs/tafeng_test_quick.yaml',
            'item_repr_config': 'configs/item_repr.yaml',
            'stb_config': 'configs/stb.yaml',
            'upstar_config': 'configs/tafeng_test_quick.yaml',
            'stb': {
                'stb_scores_path': 'outputs/phase3_stb/checkpoints/stb_scores.npy',
                'motivation_labels_path': 'outputs/phase3_stb/checkpoints/motivation_labels.npy',
                'item_embeddings_path': 'outputs/phase2_item_repr/checkpoints/item_embeddings.pt'
            },
            'baseline': {
                'model_path': 'outputs/phase1_baseline/checkpoints/best_model.pt'
            },
            'upstar': {
                'model_path': 'outputs/phase4_upstar/checkpoints/model_after_stage4.pt'
            }
        }
    else:  # full mode
        config = {
            'name': 'upstar_tafeng_full',
            'dataset': {
                'name': 'tafeng',
                'data_dir': 'data/raw/',
                'processed_dir': 'data/processed/tafeng/',
                'min_interactions': 5,
                'time_window': 86400
            },
            'baseline_config': 'configs/tafeng_baseline.yaml',
            'item_repr_config': 'configs/item_repr.yaml',
            'stb_config': 'configs/stb.yaml',
            'upstar_config': 'configs/tafeng_upstar.yaml',
            'stb': {
                'stb_scores_path': 'outputs/phase3_stb/checkpoints/stb_scores.npy',
                'motivation_labels_path': 'outputs/phase3_stb/checkpoints/motivation_labels.npy',
                'item_embeddings_path': 'outputs/phase2_item_repr/checkpoints/item_embeddings.pt'
            },
            'baseline': {
                'model_path': 'outputs/phase1_baseline/checkpoints/best_model.pt'
            },
            'upstar': {
                'model_path': 'outputs/phase4_upstar/checkpoints/model_after_stage4.pt'
            }
        }

    # Save config
    import yaml
    config_path = Path(get_mode_config(mode))
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Created config file: {config_path}")


def clean_old_data():
    """Clean old processed data and cache"""
    import shutil

    logger.info("")
    logger.info("=" * 80)
    logger.info("Cleaning Old Data (Paper-Aligned Improvements)")
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

    logger.info("")
    logger.info("✓ Old data cleaning complete!")
    logger.info("")


def main():
    """Main entry point"""
    args = parse_args()
    validate_args(args)

    # Clean old data if requested
    if args.clean_old_data:
        clean_old_data()

    logger.info("UPSTAR Full Experiment Pipeline (Paper-Aligned)")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Force rerun: {args.force_rerun}")
    if args.resume:
        logger.info("Resume mode: enabled")
    if args.start_phase is not None:
        logger.info(f"Start phase: {args.start_phase}")
    if args.stop_phase is not None:
        logger.info(f"Stop phase: {args.stop_phase}")
    logger.info("="*80)

    # Import here to avoid issues if paths aren't set up yet
    try:
        from src.experiments.experiment_manager import ExperimentManager
    except ImportError as e:
        logger.error(f"Failed to import ExperimentManager: {e}")
        logger.error("Make sure you're running from the project root directory")
        sys.exit(1)

    # Create experiment manager
    try:
        manager = ExperimentManager(
            config_path=args.config,
            output_base_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize experiment manager: {e}")
        sys.exit(1)

    # Run experiment
    try:
        start_phase = args.start_phase if args.start_phase is not None else (0 if not args.resume else None)
        skip_completed = not args.force_rerun

        manager.run(
            start_phase=start_phase if start_phase is not None else 0,
            stop_phase=args.stop_phase,
            skip_completed=skip_completed,
            resume=args.resume
        )

        # Print summary
        manager.print_summary()

        logger.info("\n✅ Experiment completed successfully!")
        logger.info(f"Results saved to: {manager.output_dir}")

    except KeyboardInterrupt:
        logger.info("\n[INTERRUPTED] Experiment interrupted by user")
        logger.info("You can resume using --resume flag")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n[FAILED] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
