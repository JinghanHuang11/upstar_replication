#!/usr/bin/env python
"""
Step 2: Item Representation Learning

Trains Item-GNN to learn item embeddings with in/out neighbor separation.

Output:
    - outputs/phase2_item_repr/checkpoints/item_embeddings.pt

Usage:
    python -m pipeline_steps.step2_item_repr
    python pipeline_steps/step2_item_repr.py --config configs/item_repr.yaml
"""

import sys
from pathlib import Path
import argparse
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_steps.base_step import BaseStep


class Step2ItemRepr(BaseStep):
    """Step 2: Item representation learning"""

    def __init__(self, config_path: str = "configs/item_repr.yaml"):
        output_dir = "outputs/phase2_item_repr"
        super().__init__(config_path, output_dir)

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

        embeddings_file = self.output_dir / 'checkpoints' / 'item_embeddings.pt'
        if embeddings_file.exists():
            self.logger.info(f"  ✓ Output exists: {embeddings_file}")
            return True
        else:
            self.logger.info(f"  → Output not found, will run training")
            return False

    def run(self) -> bool:
        """Run item representation learning"""
        try:
            # Call the training script directly
            # This is more reliable than importing, as the script handles all setup

            cmd = [
                'python',
                'src/training/train_item_repr.py',
                '--config', self.config_path
            ]

            self.logger.info("Running item representation learning...")
            self.logger.info("  Paper-Aligned: In/out neighbor separation")
            self.logger.info("  Architecture: h_n = ReLU(W_in · Σ_in + W_out · Σ_out)")

            result = subprocess.run(
                cmd,
                check=True
            )

            # Check output
            embeddings_file = self.output_dir / 'checkpoints' / 'item_embeddings.pt'
            if embeddings_file.exists():
                self.logger.info("")
                self.logger.info("✓ Item representation learning complete!")
                self.logger.info(f"Output: {embeddings_file}")
                return True
            else:
                self.logger.error("Training completed but output file not found")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Item representation learning failed")
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Item representation learning failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Step 2: Item Representation Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m pipeline_steps.step2_item_repr

  # Run with custom config
  python -m pipeline_steps.step2_item_repr --config configs/item_repr.yaml
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/item_repr.yaml',
        help='Path to config file (default: configs/item_repr.yaml)'
    )

    args = parser.parse_args()

    # Create and run step
    step = Step2ItemRepr(config_path=args.config)
    success = step.execute()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
