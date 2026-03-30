"""
conftest.py — 共享 fixtures，支持真实分布执行（phase0-5 全流程独立测试）

每个 phase 的 fixture 提供：
1. 基于 tmp_path 的独立输出目录（测试间互不干扰）
2. 完整的 config dict（覆盖所有训练脚本需要的 key）
3. 自动创建前置文件（模拟前置 phase 的输出）
4. 路径对齐：Phase N 的输出路径 = Phase N+1 的输入路径
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pickle
import json
import numpy as np
import torch
import pytest
import yaml


# ---------------------------------------------------------------------------
# 基础路径 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_output_dir(tmp_path):
    """测试输出根目录"""
    out = tmp_path / "test_outputs"
    out.mkdir(parents=True)
    return out


# ---------------------------------------------------------------------------
# 完整实验配置（所有训练脚本需要的 key 都在这里）
# ---------------------------------------------------------------------------

def _make_full_config(output_dir: Path, dataset_name: str = "tafeng") -> dict:
    """
    生成覆盖所有训练脚本的完整 config dict。
    所有路径使用绝对路径，基于 output_dir。
    """
    out_dir = str(output_dir)

    return {
        'name': 'test_experiment',
        'seed': 42,
        'dataset': {
            'name': dataset_name,
            'data_dir': str(output_dir / "data" / "raw"),
            'processed_dir': str(output_dir / "data" / "processed"),
            'cache_dir': str(output_dir / "data" / "cache"),
            'min_user_interactions': 2,
            'min_item_frequency': 2,
            'split_method': 'cv10',  # CV10-ONLY mode
            'num_folds': 10,
            'cv_random_seed': 42,
            'time_window': 86400,
        },
        'model': {
            'embed_dim': 64,
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.0,
            'max_seq_length': 50,
            'num_items': None,        # 运行时从 metadata 填充
            'use_gate': True,
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0,
            'max_epochs': 2,
            'max_epochs_per_stage': 2,
            'early_stop_patience': 10,
            'metric_for_best': 'NDCG@10',
            'batch_size_edges': 64,
            'eval_interval': 1,
            'num_workers': 0,
            'log_gate_interval': 5,
            'stage1': {'use_global_loss': True, 'use_branch_loss': False,
                        'use_orthogonality_loss': False, 'use_distillation_loss': False},
            'stage2': {'use_global_loss': True, 'use_branch_loss': True,
                        'use_orthogonality_loss': False, 'use_distillation_loss': False},
            'stage3': {'use_global_loss': True, 'use_branch_loss': True,
                        'use_orthogonality_loss': True, 'use_distillation_loss': False},
            'stage4': {'use_global_loss': True, 'use_branch_loss': True,
                        'use_orthogonality_loss': True, 'use_distillation_loss': True},
            'lambda_global': 1.0,
            'lambda_branch': 0.5,
            'lambda_ortho': 0.1,
            'lambda_distill': 0.3,
            'tau_s': 0.5,
            'tau_e': 0.5,
            'distill_temperature': 3.0,
        },
        'evaluation': {
            'k_values': [5, 10],
            'batch_size': 64,
        },
        'logging': {
            'log_dir': f"{out_dir}/logs",
            'checkpoint_dir': f"{out_dir}/checkpoints",
            'prediction_dir': f"{out_dir}/predictions",
        },
        # Phase 2 输出路径（Phase 3 的输入）
        'stb': {
            'stb_scores_path': f"{out_dir}/checkpoints/stb/stb_scores.npy",
            'motivation_labels_path': f"{out_dir}/checkpoints/stb/motivation_labels.npy",
            'item_embeddings_path': f"{out_dir}/checkpoints/item_repr/item_embeddings.pt",
            'perturbation_rounds': 5,
            'feature_noise_std': 0.1,
            'edge_removal_rate': 0.1,
            'edge_addition_rate': 0.1,
            'time_granularity': 'day',
        },
        # Phase 1 输出路径（Phase 5 的输入）
        'baseline': {
            'model_path': f"{out_dir}/checkpoints/baseline/best_model.pt",
        },
        # Phase 4 输出路径（Phase 5 的输入）
        'upstar': {
            'model_path': f"{out_dir}/checkpoints/upstar/model_after_stage4.pt",
        },
    }


# ---------------------------------------------------------------------------
# Phase fixtures（独立、基于 tmp_path）
# ---------------------------------------------------------------------------

@pytest.fixture
def phase0_config(test_output_dir):
    cfg = _make_full_config(test_output_dir)
    # Phase 0 需要 raw data，准备一个极小 fake CSV
    raw_dir = Path(cfg['dataset']['data_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 生成 10 个用户 × 5 个物品 × 5 条记录的假数据
    import pandas as pd
    records = []
    for user in range(10):
        for item in range(5):
            for t in range(5):
                records.append({
                    'user_id': user,
                    'item_id': 100 + item,   # item ID 避开 0
                    'timestamp': 1609459200 + user * 1000 + item * 100 + t
                })
    df = pd.DataFrame(records)
    (raw_dir / "tafeng.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / "tafeng.csv", index=False)

    return cfg


@pytest.fixture
def phase1_config(phase0_config, test_output_dir):
    """Phase 1 依赖 Phase 0 的输出：metadata.pkl"""
    # Phase 0 已运行，processed_dir 已有 metadata.pkl
    # phase1_config 就是 phase0_config，只是用来构造 Phase1_Baseline
    return phase0_config


@pytest.fixture
def phase2_config(phase0_config, test_output_dir):
    """Phase 2 依赖 Phase 0 的输出：metadata.pkl"""
    return phase0_config


@pytest.fixture
def phase3_config(phase0_config, test_output_dir):
    """Phase 3 依赖 Phase 2 的输出：item_embeddings.pt"""
    return phase0_config


@pytest.fixture
def phase4_config(phase0_config, test_output_dir):
    """Phase 4 依赖 Phase 2 + Phase 3 的输出"""
    return phase0_config


@pytest.fixture
def phase5_config(phase0_config, test_output_dir):
    """Phase 5 依赖 Phase 1 + Phase 4 的输出"""
    return phase0_config


# ---------------------------------------------------------------------------
# Phase 实例（每个独立使用自己的 output_dir）
# ---------------------------------------------------------------------------

@pytest.fixture
def phase0(phase0_config, test_output_dir):
    from src.experiments.phases import Phase0_Preprocess
    return Phase0_Preprocess(phase0_config, test_output_dir)


@pytest.fixture
def phase1(phase1_config, test_output_dir):
    from src.experiments.phases import Phase1_Baseline
    return Phase1_Baseline(phase1_config, test_output_dir)


@pytest.fixture
def phase2(phase2_config, test_output_dir):
    from src.experiments.phases import Phase2_ItemRepr
    return Phase2_ItemRepr(phase2_config, test_output_dir)


@pytest.fixture
def phase3(phase3_config, test_output_dir):
    from src.experiments.phases import Phase3_STB
    return Phase3_STB(phase3_config, test_output_dir)


@pytest.fixture
def phase4(phase4_config, test_output_dir):
    from src.experiments.phases import Phase4_UPSTAR
    return Phase4_UPSTAR(phase4_config, test_output_dir)


@pytest.fixture
def phase5(phase5_config, test_output_dir):
    from src.experiments.phases import Phase5_Evaluate
    return Phase5_Evaluate(phase5_config, test_output_dir)
