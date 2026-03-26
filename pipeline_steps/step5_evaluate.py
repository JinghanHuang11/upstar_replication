#!/usr/bin/env python
"""
Phase 5: Evaluation and Comparison

运行方式:
    python pipeline_steps/step5_evaluate.py
    python pipeline_steps/step5_evaluate.py --config configs/tafeng_eval.yaml
    python pipeline_steps/step5_evaluate.py --force-rerun

前置条件: Phase 1 (baseline) + Phase 4 (UPSTAR) 均已完成
输出目录: outputs/pipeline_steps/phase5_evaluate/
输出文件: comparison_table.txt, comparison.json
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.phases import Phase5_Evaluate

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 5: Evaluation — UPSTAR vs Baseline 性能对比评估'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tafeng_eval.yaml',
        help='配置文件路径 (default: configs/tafeng_eval.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/pipeline_steps',
        help='输出根目录 (default: outputs/pipeline_steps)'
    )
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='即使已完成也强制重新运行'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f'配置文件不存在: {config_path}')
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('=' * 60)
    logger.info('Phase 5: Evaluation and Comparison')
    logger.info(f'  Config  : {config_path}')
    logger.info(f'  Output  : {output_dir}')
    logger.info('  指标    : HR@K / NDCG@K / MRR@K')
    logger.info('=' * 60)

    phase = Phase5_Evaluate(config, output_dir)

    if not phase.check_dependencies():
        logger.error(
            '前置条件不满足: 请确认以下步骤均已完成:\n'
            '  - python pipeline_steps/step1_baseline.py\n'
            '  - python pipeline_steps/step4_upstar.py'
        )
        sys.exit(1)

    if phase.is_completed() and not args.force_rerun:
        logger.info('Phase 5 已完成，跳过。使用 --force-rerun 强制重新运行。')
        results = phase.load_results()
    else:
        results = phase.execute()

    logger.info('\n--- Phase 5 结果摘要 ---')
    for k, v in results.items():
        if isinstance(v, dict):
            logger.info(f'  {k}:')
            for metric, val in v.items():
                logger.info(f'    {metric}: {val}')
        else:
            logger.info(f'  {k}: {v}')
    logger.info('\nPipeline 全部完成！结果保存在 outputs/pipeline_steps/')


if __name__ == '__main__':
    main()
