#!/usr/bin/env python
"""
Phase 4: UPSTAR Training (Four-Stage)

运行方式:
    python pipeline_steps/step4_upstar.py
    python pipeline_steps/step4_upstar.py --config configs/tafeng_upstar.yaml
    python pipeline_steps/step4_upstar.py --force-rerun

前置条件: Phase 1 (baseline) + Phase 2 (item embeddings) + Phase 3 (STB scores) 均已完成
输出目录: outputs/pipeline_steps/phase4_upstar/

四阶段 Loss:
  Stage 1: L_global
  Stage 2: L_global + L_branch
  Stage 3: L_global + L_branch + L_orth
  Stage 4: L_global + L_branch + L_orth + L_distill
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.phases import Phase4_UPSTAR

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 4: UPSTAR Training — 四阶段渐进式训练'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tafeng_upstar.yaml',
        help='配置文件路径 (default: configs/tafeng_upstar.yaml)'
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
    logger.info('Phase 4: UPSTAR Training (Four-Stage)')
    logger.info(f'  Config  : {config_path}')
    logger.info(f'  Output  : {output_dir}')
    logger.info('=' * 60)

    phase = Phase4_UPSTAR(config, output_dir)

    if not phase.check_dependencies():
        logger.error(
            '前置条件不满足: 请确认以下步骤均已完成:\n'
            '  - python pipeline_steps/step1_baseline.py\n'
            '  - python pipeline_steps/step2_item_repr.py\n'
            '  - python pipeline_steps/step3_stb.py'
        )
        sys.exit(1)

    if phase.is_completed() and not args.force_rerun:
        logger.info('Phase 4 已完成，跳过。使用 --force-rerun 强制重新运行。')
        results = phase.load_results()
    else:
        results = phase.execute()

    logger.info('\n--- Phase 4 结果摘要 ---')
    for k, v in results.items():
        logger.info(f'  {k}: {v}')
    logger.info('Phase 4 完成！接下来运行: python pipeline_steps/step5_evaluate.py')


if __name__ == '__main__':
    main()
