"""
test_phase0_preprocess.py — Phase 0 真实执行测试

测试目标：数据预处理流水线
- 依赖检查（raw CSV 存在性）
- 状态追踪（is_completed / save_status）
- execute() 真实运行并验证输出
"""

import pytest
from pathlib import Path


class TestPhase0Dependencies:
    def test_check_dependencies_with_raw_data(self, phase0):
        """有 raw CSV 时，依赖检查应返回 True"""
        assert phase0.check_dependencies() is True


class TestPhase0Execution:
    def test_execute_runs_preprocessing(self, phase0):
        """
        execute() 真实运行预处理流水线。
        验证返回结果包含必要字段。
        """
        results = phase0.execute()

        assert 'num_users' in results
        assert 'num_items' in results
        assert 'num_interactions' in results
        assert results['num_users'] > 0
        assert results['num_items'] > 0
        assert results['num_interactions'] > 0

    def test_execute_creates_metadata_file(self, phase0, phase0_config):
        """execute() 应在 processed_dir 下生成 metadata.pkl"""
        phase0.execute()
        metadata_path = Path(phase0_config['dataset']['processed_dir']) / 'metadata.pkl'
        assert metadata_path.exists(), f"metadata.pkl not found at {metadata_path}"

    def test_execute_creates_sequence_files(self, phase0, phase0_config):
        """execute() 应生成 train/val/test_sequences.pkl"""
        phase0.execute()
        proc = Path(phase0_config['dataset']['processed_dir'])
        assert (proc / 'train_sequences.pkl').exists()
        assert (proc / 'val_sequences.pkl').exists()
        assert (proc / 'test_sequences.pkl').exists()

    def test_execute_is_completed_after_run(self, phase0):
        """execute() 完成后 is_completed() 应返回 True"""
        phase0.execute()
        assert phase0.is_completed() is True

    def test_execute_saves_results(self, phase0):
        """execute() 应将结果持久化到 results.json"""
        phase0.execute()
        loaded = phase0.load_results()
        assert loaded is not None
        assert 'num_users' in loaded

    def test_execute_duration_recorded(self, phase0):
        """execute() 应记录运行时间"""
        phase0.execute()
        duration = phase0.get_duration()
        assert duration is not None
        assert duration > 0
