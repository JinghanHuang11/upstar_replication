#!/usr/bin/env python
"""
Step 1: Baseline Training

Trains baseline LSTM model with 10-fold cross-validation.

Output:
    - outputs/phase1_baseline/checkpoints/
    - outputs/phase1_baseline/logs/
    - outputs/phase1_baseline/results/main_results.json

Usage:
    python -m pipeline_steps.step1_baseline
    python pipeline_steps/step1_baseline.py --config configs/tafeng_baseline.yaml
    python pipeline_steps/step1_baseline.py --num-folds 1 --quick-test
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_steps.base_step import BaseStep


class Step1Baseline(BaseStep):
    """Step 1: Baseline training with 10-fold CV"""

    def __init__(
        self,
        config_path: str = "configs/tafeng_baseline.yaml",
        num_folds: int = 10,
        quick_test: bool = False
    ):
        output_dir = "outputs/phase1_baseline"
        super().__init__(config_path, output_dir)
        self.num_folds = num_folds
        self.quick_test = quick_test

    def check_prerequisites(self) -> bool:
        """Check if preprocessing is complete"""
        self.logger.info("Checking prerequisites...")

        # Check config
        config_path = Path(self.config_path)
        if not config_path.exists():
            self.logger.error(f"  ✗ Config not found: {config_path}")
            return False
        self.logger.info(f"  ✓ Config: {config_path}")

        # Check Phase 0 output
        try:
            config = self.load_config()
            processed_dir = Path(config['dataset']['processed_dir'])
            metadata_file = processed_dir / 'metadata.pkl'

            if not metadata_file.exists():
                self.logger.error(f"  ✗ Phase 0 output not found: {metadata_file}")
                self.logger.error("    Run: python -m pipeline_steps.step0_preprocess")
                return False

            self.logger.info(f"  ✓ Phase 0 complete: {metadata_file}")

        except Exception as e:
            self.logger.error(f"  ✗ Error checking prerequisites: {e}")
            return False

        return True

    def check_output_exists(self) -> bool:
        """Check if output already exists"""
        if not self.output_dir:
            return False

        results_file = self.output_dir / 'results' / 'main_results.json'
        if results_file.exists():
            self.logger.info(f"  ✓ Output exists: {results_file}")
            return True
        else:
            self.logger.info(f"  → Output not found, will run training")
            return False

    def create_output_dir(self):
        """Create output directory structure"""
        if self.output_dir:
            (self.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'results').mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {self.output_dir}")

    def run(self) -> bool:
        """Run baseline training"""
        try:
            from src.training.cross_validation_baseline import run_cross_validation
            from src.evaluation.report import ReportGenerator
            import json

            self.logger.info(f"Running {self.num_folds}-fold cross-validation...")

            if self.quick_test:
                self.logger.info("  Quick test mode: 1 fold, 3 epochs")
                self.logger.info("  Estimated time: 5-10 minutes")
            else:
                self.logger.info(f"  Estimated time: {self.num_folds * 2}-{self.num_folds * 4} hours")

            # Run cross-validation
            cv_results = run_cross_validation(
                config_path=self.config_path,
                num_folds=self.num_folds,
                output_dir=str(self.output_dir),
                quick_test=self.quick_test
            )

            # Generate reports (pass raw decimal values, ReportGenerator converts to %)
            generator = ReportGenerator({}, cv_results)
            generator.save_all_formats(str(self.output_dir / 'results'))

            # Save with model type label (use cv_results dict directly, not file)
            results_file = self.output_dir / 'results' / 'main_results.json'
            data = {
                'model_type': 'baseline',
                'dataset': 'tafeng',
                'num_folds': self.num_folds,
                'metrics': cv_results
            }
            with open(results_file, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info("")
            self.logger.info("✓ Baseline training complete!")
            self.logger.info(f"Results: {results_file}")
            self.logger.info("")
            self.logger.info("Cross-Validation Results:")
            self.logger.info(generator.generate_main_table(format='text'))

            return True

        except Exception as e:
            self.logger.error(f"Baseline training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Step 1: Baseline Training (10-Fold CV)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full 10-fold CV
  python -m pipeline_steps.step1_baseline

  # Quick test (1 fold, 3 epochs)
  python -m pipeline_steps.step1_baseline --quick-test

  # Run 2-fold CV
  python -m pipeline_steps.step1_baseline --num-folds 2

  # Use custom config
  python -m pipeline_steps.step1_baseline --config configs/baseline.yaml
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/tafeng_baseline.yaml',
        help='Path to config file (default: configs/tafeng_baseline.yaml)'
    )

    parser.add_argument(
        '--num-folds',
        type=int,
        default=10,
        help='Number of folds for cross-validation (default: 10)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode: 1 fold, 3 epochs'
    )

    args = parser.parse_args()

    # Create and run step
    step = Step1Baseline(
        config_path=args.config,
        num_folds=args.num_folds,
        quick_test=args.quick_test
    )
    success = step.execute()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
