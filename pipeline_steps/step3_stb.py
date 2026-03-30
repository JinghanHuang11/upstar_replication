#!/usr/bin/env python
"""
Step 3: STB Computation

Computes STB scores and motivation labels using day-level item-time graph.

Output:
    - outputs/phase3_stb/checkpoints/stb_scores.npy
    - outputs/phase3_stb/checkpoints/motivation_labels.npy
    - outputs/phase3_stb/checkpoints/stb_stats.pkl

Usage:
    python -m pipeline_steps.step3_stb
    python pipeline_steps/step3_stb.py --config configs/stb.yaml
"""

import sys
from pathlib import Path
import argparse
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_steps.base_step import BaseStep


class Step3Stb(BaseStep):
    """Step 3: STB computation"""

    def __init__(self, config_path: str = "configs/stb.yaml"):
        output_dir = "outputs/phase3_stb"
        super().__init__(config_path, output_dir)

    def check_prerequisites(self) -> bool:
        """Check if prerequisites are satisfied"""
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

        return True

    def check_output_exists(self) -> bool:
        """Check if output already exists"""
        if not self.output_dir:
            return False

        labels_file = self.output_dir / 'checkpoints' / 'motivation_labels.npy'
        if labels_file.exists():
            self.logger.info(f"  ✓ Output exists: {labels_file}")
            return True
        else:
            self.logger.info(f"  → Output not found, will run STB computation")
            return False

    def run(self) -> bool:
        """Run STB computation"""
        try:
            # Call the training script directly
            cmd = [
                'python',
                'src/training/train_stb.py',
                '--config', self.config_path
            ]

            self.logger.info("Running STB computation...")
            self.logger.info("  Paper-Aligned:")
            self.logger.info("    - Day-level item-time graph (ρ=50)")
            self.logger.info("    - Mutual Information estimation (MINE)")
            self.logger.info("    - Perturbation: α=0.4, ε=0.1, β=40")

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
                        self.logger.info(f"  {line}")

            # Check output
            labels_file = self.output_dir / 'checkpoints' / 'motivation_labels.npy'
            scores_file = self.output_dir / 'checkpoints' / 'stb_scores.npy'

            if labels_file.exists() and scores_file.exists():
                self.logger.info("")
                self.logger.info("✓ STB computation complete!")
                self.logger.info(f"Outputs:")
                self.logger.info(f"  - {scores_file}")
                self.logger.info(f"  - {labels_file}")

                # Load and show statistics
                import numpy as np
                scores = np.load(scores_file)
                labels = np.load(labels_file)

                self.logger.info("")
                self.logger.info("STB Statistics:")
                self.logger.info(f"  Scores: mean={scores.mean():.4f}, std={scores.std():.4f}")
                self.logger.info(f"  Labels distribution:")
                for label in [0, 1, 2]:
                    count = (labels == label).sum()
                    pct = count / len(labels) * 100
                    name = {0: "exploratory", 1: "stable", 2: "uncategorized"}[label]
                    self.logger.info(f"    {name}: {count} ({pct:.1f}%)")

                return True
            else:
                self.logger.error("STB computation completed but output files not found")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"STB computation failed")
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"STB computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Step 3: STB Computation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m pipeline_steps.step3_stb

  # Run with custom config
  python -m pipeline_steps.step3_stb --config configs/stb.yaml
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/stb.yaml',
        help='Path to config file (default: configs/stb.yaml)'
    )

    args = parser.parse_args()

    # Create and run step
    step = Step3Stb(config_path=args.config)
    success = step.execute()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
