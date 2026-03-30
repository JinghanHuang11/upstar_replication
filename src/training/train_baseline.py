"""
Training script for LSTM baseline (cv10 mode ONLY)

This is a THIN WRAPPER around cross_validation_baseline.py.
The ONLY implementation is in cross_validation_baseline.py.

Usage:
    python -m src.training.train_baseline --config configs/tafeng_baseline.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.cross_validation_baseline import run_cross_validation


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Training (10-Fold CV)')
    parser.add_argument('--config', type=str, default='configs/tafeng_baseline.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (optional)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Baseline LSTM Training - 10-Fold Cross-Validation")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir or 'default (from config)'}")
    print("")
    print("Implementation: src/training/cross_validation_baseline.py")
    print("")

    # Delegate to cross_validation_baseline.py
    cv_results = run_cross_validation(args.config, args.output_dir)

    print("")
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)

    return cv_results


if __name__ == '__main__':
    main()
