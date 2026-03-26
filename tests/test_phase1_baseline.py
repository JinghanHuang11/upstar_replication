"""
test_phase1_baseline.py — Phase 1 真实执行测试

测试目标：LSTM Baseline 模型训练
- 依赖检查（metadata.pkl 存在性）
- execute() 真实调用 train_baseline.main()
- 验证模型文件和指标输出
"""

import pytest
from pathlib import Path


class TestPhase1Dependencies:
    def test_check_dependencies_missing_metadata(self, phase1):
        """metadata.pkl 不存在时应返回 False"""
        assert phase1.check_dependencies() is False

    def test_check_dependencies_with_metadata(self, phase1, phase0):
        """Phase 0 执行后，metadata.pkl 存在，依赖检查通过"""
        phase0.execute()
        assert phase1.check_dependencies() is True


class TestPhase1Execution:
    def test_execute_runs_baseline_training(self, phase1, phase0):
        """execute() 真实调用 train_baseline.main()，验证指标输出"""
        phase0.execute()
        results = phase1.execute()

        assert 'ndcg@10' in results
        assert 'hr@10' in results
        assert 'mrr@10' in results
        assert 'model_path' in results
        assert results['ndcg@10'] >= 0.0
        assert results['hr@10'] >= 0.0
        assert results['mrr@10'] >= 0.0

    def test_execute_creates_checkpoint(self, phase1, phase0, phase1_config):
        """execute() 应生成 best_model.pt"""
        phase0.execute()
        phase1.execute()
        ckpt_path = Path(phase1_config['baseline']['model_path'])
        assert ckpt_path.exists(), f"best_model.pt not found at {ckpt_path}"

    def test_execute_is_completed_after_run(self, phase1, phase0):
        """执行后 is_completed() 应返回 True"""
        phase0.execute()
        phase1.execute()
        assert phase1.is_completed() is True

    def test_execute_saves_results(self, phase1, phase0):
        """执行后 results.json 应包含指标"""
        phase0.execute()
        phase1.execute()
        loaded = phase1.load_results()
        assert loaded is not None
        assert 'ndcg@10' in loaded
