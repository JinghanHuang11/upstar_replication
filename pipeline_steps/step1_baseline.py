#!/usr/bin/env python
"""
Phase 1: Baseline Training (LSTM)

运行方式:
    python pipeline_steps/step1_baseline.py
    python pipeline_steps/step1_baseline.py --config configs/tafeng_baseline.yaml
    python pipeline_steps/step1_baseline.py --force-rerun

前置条件: Phase 0 已完成 (data/processed/tafeng/ 存在)
输出目录: outputs/pipeline_steps/phase1_baseline/
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.phases import Phase1_Baseline

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 1: Baseline Training — LSTM 基线模型训练'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tafeng_baseline.yaml',
        help='配置文件路径 (default: configs/tafeng_baseline.yaml)'
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
    logger.info('Phase 1: Baseline Training (LSTM)')
    logger.info(f'  Config  : {config_path}')
    logger.info(f'  Output  : {output_dir}')
    logger.info('=' * 60)

    phase = Phase1_Baseline(config, output_dir)

    if not phase.check_dependencies():
        logger.error('前置条件不满足: 请先运行 python pipeline_steps/step0_preprocess.py')
        sys.exit(1)

    if phase.is_completed() and not args.force_rerun:
        logger.info('Phase 1 已完成，跳过。使用 --force-rerun 强制重新运行。')
        results = phase.load_results()
    else:
        results = phase.execute()

    logger.info('\n--- Phase 1 结果摘要 ---')
    for k, v in results.items():
        logger.info(f'  {k}: {v}')
    logger.info('Phase 1 完成！接下来运行: python pipeline_steps/step2_item_repr.py')


if __name__ == '__main__':
    main()
