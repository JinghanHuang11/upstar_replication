"""
Training script for UPSTAR (cv10 mode ONLY)

This is a THIN WRAPPER around train_upstar_cv.py.
The ONLY implementation is in train_upstar_cv.py.

Usage:
    python -m src.training.train_upstar --config configs/tafeng_upstar.yaml
    python -m src.training.train_upstar --config configs/tafeng_upstar.yaml --stage 2
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.train_upstar_cv import run_cross_validation


def parse_args():
    parser = argparse.ArgumentParser(description='UPSTAR Training (10-Fold CV)')
    parser.add_argument('--config', type=str, default='configs/tafeng_upstar.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (optional)')
    parser.add_argument('--stage', type=int, default=None,
                       help='Train specific stage only (1-4, default: all stages)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("UPSTAR Model Training - 10-Fold Cross-Validation")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir or 'default (from config)'}")
    if args.stage:
        print(f"Training stage: {args.stage} (single stage mode)")
    else:
        print("Training stages: 1, 2, 3, 4 (all stages)")
    print("")
    print("Implementation: src/training/train_upstar_cv.py")
    print("")

    # Delegate to train_upstar_cv.py
    cv_results = run_cross_validation(args.config, args.output_dir, args.stage)

    print("")
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)

    return cv_results


if __name__ == '__main__':
    main()
