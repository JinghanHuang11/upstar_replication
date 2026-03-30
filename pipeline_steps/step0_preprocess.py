#!/usr/bin/env python
"""
Step 0: Data Preprocessing

Preprocesses raw Ta-Feng data and creates CV10 splits.

Output:
    - data/processed/tafeng/metadata.pkl
    - data/processed/tafeng/cv_splits/

Usage:
    python -m pipeline_steps.step0_preprocess
    python pipeline_steps/step0_preprocess.py --config configs/tafeng_baseline.yaml
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_steps.base_step import BaseStep


class Step0Preprocess(BaseStep):
    """Step 0: Data preprocessing"""

    def __init__(self, config_path: str = "configs/tafeng_baseline.yaml"):
        super().__init__(config_path)
        self.default_config = "configs/tafeng_baseline.yaml"

    def check_prerequisites(self) -> bool:
        """Check if raw data exists"""
        self.logger.info("Checking prerequisites...")

        # Load config to get data directory
        try:
            config = self.load_config()
            data_dir = Path(config['dataset']['data_dir'])

            # Check for raw data
            raw_data = None
            for filename in ['ta_feng.csv', 'ta_feng.txt']:
                potential_path = data_dir / filename
                if potential_path.exists():
                    raw_data = potential_path
                    break

            if raw_data:
                self.logger.info(f"  ✓ Raw data found: {raw_data}")
                return True
            else:
                self.logger.error(f"  ✗ Raw data not found in: {data_dir}")
                self.logger.error("    Expected: ta_feng.csv or ta_feng.txt")
                return False

        except Exception as e:
            self.logger.error(f"  ✗ Error checking prerequisites: {e}")
            return False

    def check_output_exists(self) -> bool:
        """Check if output already exists"""
        try:
            config = self.load_config()
            processed_dir = Path(config['dataset']['processed_dir'])
            metadata_file = processed_dir / 'metadata.pkl'

            if metadata_file.exists():
                self.logger.info(f"  ✓ Output exists: {metadata_file}")
                return True
            else:
                self.logger.info(f"  → Output not found, will run preprocessing")
                return False

        except Exception as e:
            self.logger.warning(f"  Could not check output: {e}")
            return False

    def run(self) -> bool:
        """Run preprocessing"""
        try:
            from src.data.preprocess import Preprocessor
            from src.data.build_sequences import SequenceBuilder

            config = self.load_config()

            self.logger.info("Step 1: Loading and preprocessing raw data...")
            preprocessor = Preprocessor(config)
            df = preprocessor.run()

            self.logger.info("Step 2: Building sequences (with timestamps)...")
            self.logger.info("  New format: [(item, timestamp), ...]")
            self.logger.info("  Required for day-level item-time graph")

            builder = SequenceBuilder(config)
            builder.run(df)

            # List output files
            processed_dir = Path(config['dataset']['processed_dir'])
            self.logger.info("")
            self.logger.info("Output files:")
            for f in sorted(processed_dir.glob('*')):
                if f.is_file():
                    size = f.stat().st_size
                    size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
                    self.logger.info(f"  {f.name}: {size_str}")

            self.logger.info("")
            self.logger.info("✓ Preprocessing complete!")
            self.logger.info("  Sequences include timestamps for day-level item-time graph")

            return True

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Step 0: Data Preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m pipeline_steps.step0_preprocess

  # Run with custom config
  python -m pipeline_steps.step0_preprocess --config configs/tafeng_baseline.yaml
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/tafeng_baseline.yaml',
        help='Path to config file (default: configs/tafeng_baseline.yaml)'
    )

    args = parser.parse_args()

    # Create and run step
    step = Step0Preprocess(config_path=args.config)
    success = step.execute()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
