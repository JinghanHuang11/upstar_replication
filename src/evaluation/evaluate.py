"""
Main evaluation script with command-line interface (CV10-ONLY)

Supports:
- 10-fold cross-validation evaluation (ONLY mode)
- Multiple output formats
"""

import argparse
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.evaluator import UPSTAREvaluator
from src.training.train_upstar_cv import run_cross_validation
from src.evaluation.report import ReportGenerator
from src.utils.logger import get_logger
import logging

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='UPSTAR Evaluation (10-Fold CV)')

    # Config
    parser.add_argument('--config', type=str, default='configs/tafeng_eval.yaml',
                       help='Path to config file')

    # Output directory
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')

    # Output formats
    parser.add_argument('--formats', type=str, nargs='+',
                       default=['json', 'csv', 'txt'],
                       choices=['json', 'csv', 'txt', 'latex', 'markdown'],
                       help='Output formats')

    # Main results only
    parser.add_argument('--main-only', action='store_true',
                       help='Only output main results table')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine output directory
    if args.output_dir is None:
        output_dir = Path(config['logging'].get('cv_dir', 'outputs/evaluation'))
    else:
        output_dir = Path(args.output_dir)

    # Device
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 10-Fold Cross-Validation (ONLY mode)
    logger.info("=" * 80)
    logger.info("UPSTAR 10-Fold Cross-Validation Evaluation")
    logger.info("=" * 80)

    # Run cross-validation using training module
    cv_results = run_cross_validation(
        config_path=args.config,
        output_dir=str(output_dir)
    )

    # Generate reports
    # Convert to percentage for display
    cv_stats_pct = {}
    for metric_name, stats in cv_results.items():
        cv_stats_pct[metric_name] = {
            'mean': stats['mean'] * 100,
            'std': stats['std'] * 100
        }

    generator = ReportGenerator({}, cv_stats_pct)

    # Save in requested formats
    for fmt in args.formats:
        if fmt == 'json':
            generator.save_json(output_dir / 'cv_results.json')
        elif fmt == 'csv':
            generator.save_csv(output_dir / 'cv_results.csv')
        elif fmt == 'txt':
            generator.save_txt(output_dir / 'cv_results.txt', main_only=args.main_only)
        elif fmt == 'latex':
            latex_path = output_dir / 'cv_results.tex'
            with open(latex_path, 'w') as f:
                f.write(generator.generate_main_table(format='latex'))
            logger.info(f"Saved LaTeX to {latex_path}")
        elif fmt == 'markdown':
            md_path = output_dir / 'cv_results.md'
            with open(md_path, 'w') as f:
                f.write(generator.generate_main_table(format='markdown'))
            logger.info(f"Saved Markdown to {md_path}")

    logger.info("\n" + "=" * 80)
    logger.info("10-Fold Cross-Validation Complete!")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
