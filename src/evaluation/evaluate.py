"""
Main evaluation script with command-line interface

Supports:
- Single evaluation
- Cross-validation
- Multiple output formats
"""

import argparse
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.evaluator import UPSTAREvaluator
from src.evaluation.cross_validation import CrossValidator
from src.evaluation.report import ReportGenerator
from src.utils.logger import get_logger
import logging

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='UPSTAR Evaluation')

    # Mode
    parser.add_argument('--mode', type=str, choices=['eval', 'cv'],
                       default='eval', help='Evaluation mode: eval or cv')

    # Config
    parser.add_argument('--config', type=str, default='configs/tafeng_eval.yaml',
                       help='Path to config file')

    # Checkpoint (for eval mode)
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (for eval mode)')

    # Split (for eval mode)
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Split to evaluate on (for eval mode)')

    # Output directory
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')

    # Cross-validation settings
    parser.add_argument('--num-folds', type=int, default=10,
                       help='Number of folds for cross-validation')

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
        if args.mode == 'cv':
            output_dir = Path(config['logging']['cv_dir'])
        else:
            output_dir = Path(config['logging']['eval_dir'])
    else:
        output_dir = Path(args.output_dir)

    # Device
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if args.mode == 'eval':
        # Single evaluation
        logger.info("=" * 80)
        logger.info("UPSTAR Single Evaluation")
        logger.info("=" * 80)

        # Create evaluator
        evaluator = UPSTAREvaluator(
            config=config,
            checkpoint_path=args.checkpoint,
            device=str(device)
        )

        # Evaluate
        metrics = evaluator.evaluate(
            split=args.split,
            k_values=[1, 5, 10, 15, 20, 50],
            save_predictions=True,
            output_dir=str(output_dir)
        )

        # Generate reports
        generator = ReportGenerator(metrics)

        # Save in requested formats
        for fmt in args.formats:
            if fmt == 'json':
                generator.save_json(output_dir / 'main_results.json')
            elif fmt == 'csv':
                generator.save_csv(output_dir / 'main_results.csv')
            elif fmt == 'txt':
                generator.save_txt(output_dir / 'main_results.txt', main_only=args.main_only)
            elif fmt == 'latex':
                latex_path = output_dir / 'main_results.tex'
                with open(latex_path, 'w') as f:
                    f.write(generator.generate_main_table(format='latex'))
                logger.info(f"Saved LaTeX to {latex_path}")
            elif fmt == 'markdown':
                md_path = output_dir / 'main_results.md'
                with open(md_path, 'w') as f:
                    f.write(generator.generate_main_table(format='markdown'))
                logger.info(f"Saved Markdown to {md_path}")

        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Complete!")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to {output_dir}")

    elif args.mode == 'cv':
        # Cross-validation
        logger.info("=" * 80)
        logger.info(f"UPSTAR {args.num_folds}-Fold Cross-Validation")
        logger.info("=" * 80)

        # Create cross-validator
        cv = CrossValidator(
            config=config,
            num_folds=args.num_folds,
            random_seed=config.get('seed', 42)
        )

        # Run cross-validation
        cv_results = cv.run_cross_validation(device, output_dir)

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
        logger.info("Cross-Validation Complete!")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
