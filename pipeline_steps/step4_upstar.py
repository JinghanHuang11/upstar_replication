#!/usr/bin/env python
"""
Step 4: UPSTAR Training

Trains UPSTAR model with 10-fold cross-validation and dual teacher-student.

Output:
    - outputs/phase4_upstar/checkpoints/
    - outputs/phase4_upstar/logs/
    - outputs/phase4_upstar/results/cv_results.json

Usage:
    python -m pipeline_steps.step4_upstar
    python pipeline_steps/step4_upstar.py --config configs/tafeng_upstar.yaml
    python pipeline_steps/step4_upstar.py --num-folds 1 --quick-test
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_steps.base_step import BaseStep


class Step4Upstar(BaseStep):
    """Step 4: UPSTAR training with 10-fold CV"""

    def __init__(
        self,
        config_path: str = "configs/tafeng_upstar.yaml",
        num_folds: int = 10,
        quick_test: bool = False
    ):
        output_dir = "outputs/phase4_upstar"
        super().__init__(config_path, output_dir)
        self.num_folds = num_folds
        self.quick_test = quick_test

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are satisfied"""
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
            self.logger.error(f"  ✗ Error checking Phase 0: {e}")
            return False

        # Check Phase 2 output
        embeddings_file = Path("outputs/phase2_item_repr/checkpoints/item_embeddings.pt")
        if not embeddings_file.exists():
            self.logger.error(f"  ✗ Phase 2 output not found: {embeddings_file}")
            self.logger.error("    Run: python -m pipeline_steps.step2_item_repr")
            return False

        self.logger.info(f"  ✓ Phase 2 complete: {embeddings_file}")

        # Check Phase 3 output
        labels_file = Path("outputs/phase3_stb/checkpoints/motivation_labels.npy")
        if not labels_file.exists():
            self.logger.error(f"  ✗ Phase 3 output not found: {labels_file}")
            self.logger.error("    Run: python -m pipeline_steps.step3_stb")
            return False

        self.logger.info(f"  ✓ Phase 3 complete: {labels_file}")

        # Show training configuration
        self.logger.info("")
        self.logger.info("Paper-Aligned Features:")
        self.logger.info("  - S/E/O models: 4-layer LSTM, hidden_dim=128")
        self.logger.info("  - Global Fusion: learnable gate")
        self.logger.info("  - Dual Teacher-Student: S↔E knowledge distillation")
        self.logger.info("  - Orthogonality: τ_s=0.5, τ_e=0.5")
        self.logger.info("")

        return True

    def check_output_exists(self) -> bool:
        """Check if output already exists"""
        if not self.output_dir:
            return False

        results_file = self.output_dir / 'results' / 'cv_results.json'
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
            (self.output_dir / 'predictions').mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {self.output_dir}")

    def run(self) -> bool:
        """Run UPSTAR training"""
        try:
            from src.training.train_upstar_cv import run_cross_validation
            from src.evaluation.report import ReportGenerator
            import json

            self.logger.info(f"Running {self.num_folds}-fold cross-validation...")

            if self.quick_test:
                self.logger.info("  Quick test mode: 1 fold, 2 epochs per stage")
                self.logger.info("  Estimated time: 10-20 minutes")
            else:
                self.logger.info(f"  Estimated time: {self.num_folds * 8}-{self.num_folds * 16} hours")

            # Run cross-validation
            cv_results = run_cross_validation(
                config_path=self.config_path,
                num_folds=self.num_folds,
                output_dir=str(self.output_dir),
                quick_test=self.quick_test
            )

            # Generate reports
            cv_stats_pct = {}
            for metric_name, stats in cv_results.items():
                cv_stats_pct[metric_name] = {
                    'mean': stats['mean'] * 100,
                    'std': stats['std'] * 100
                }

            generator = ReportGenerator({}, cv_stats_pct)
            generator.save_all_formats(str(self.output_dir / 'results'))

            # Save with model type label
            results_file = self.output_dir / 'results' / 'cv_results.json'
            with open(results_file, 'r') as f:
                data = json.load(f)
            data['model_type'] = 'upstar'
            data['dataset'] = 'tafeng'
            data['num_folds'] = self.num_folds
            with open(results_file, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info("")
            self.logger.info("✓ UPSTAR training complete!")
            self.logger.info(f"Results: {results_file}")
            self.logger.info("")
            self.logger.info("Cross-Validation Results:")
            self.logger.info(generator.generate_main_table(format='text'))

            return True

        except Exception as e:
            self.logger.error(f"UPSTAR training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Step 4: UPSTAR Training (10-Fold CV)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full 10-fold CV
  python -m pipeline_steps.step4_upstar

  # Quick test (1 fold, 2 epochs per stage)
  python -m pipeline_steps.step4_upstar --quick-test

  # Run 2-fold CV
  python -m pipeline_steps.step4_upstar --num-folds 2

  # Use custom config
  python -m pipeline_steps.step4_upstar --config configs/upstar.yaml
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/tafeng_upstar.yaml',
        help='Path to config file (default: configs/tafeng_upstar.yaml)'
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
        help='Quick test mode: 1 fold, 2 epochs per stage'
    )

    args = parser.parse_args()

    # Create and run step
    step = Step4Upstar(
        config_path=args.config,
        num_folds=args.num_folds,
        quick_test=args.quick_test
    )
    success = step.execute()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
