"""
test_phase3_stb.py — Phase 3 真实执行测试

测试目标：STB 稳定交易偏差计算
- 依赖检查（item_embeddings.pt 存在性）
- execute() 真实调用 train_stb.main()
- 验证 stb_scores.npy 和 motivation_labels.npy 输出
"""

import pytest
from pathlib import Path
import numpy as np


class TestPhase3Dependencies:
    def test_check_dependencies_missing_embeddings(self, phase3):
        """item_embeddings.pt 不存在时应返回 False"""
        assert phase3.check_dependencies() is False

    def test_check_dependencies_with_embeddings(self, phase3, phase0, phase2):
        """Phase 2 执行后，依赖检查通过"""
        phase0.execute()
        phase2.execute()
        assert phase3.check_dependencies() is True


class TestPhase3Execution:
    def test_execute_runs_stb(self, phase3, phase0, phase2):
        """execute() 真实调用 train_stb.main()"""
        phase0.execute()
        phase2.execute()
        results = phase3.execute()

        assert 'stable_count' in results
        assert 'exploratory_count' in results
        assert 'other_count' in results
        assert 'stb_scores_path' in results
        assert 'motivation_labels_path' in results

    def test_execute_creates_scores_file(self, phase3, phase0, phase2, phase3_config):
        """execute() 应生成 stb_scores.npy"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        scores_path = Path(phase3_config['stb']['stb_scores_path'])
        assert scores_path.exists(), f"stb_scores.npy not found at {scores_path}"

    def test_execute_creates_labels_file(self, phase3, phase0, phase2, phase3_config):
        """execute() 应生成 motivation_labels.npy"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        labels_path = Path(phase3_config['stb']['motivation_labels_path'])
        assert labels_path.exists(), f"motivation_labels.npy not found at {labels_path}"

    def test_execute_label_counts_sum(self, phase3, phase0, phase2, phase3_config):
        """标签总数应等于 item 数"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        labels = np.load(phase3_config['stb']['motivation_labels_path'])
        scores = np.load(phase3_config['stb']['stb_scores_path'])
        assert len(labels) == len(scores)

    def test_execute_is_completed_after_run(self, phase3, phase0, phase2):
        """执行后 is_completed() 应返回 True"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        assert phase3.is_completed() is True

    def test_execute_saves_results(self, phase3, phase0, phase2):
        """执行后 results.json 应包含标签分布"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        loaded = phase3.load_results()
        assert loaded is not None
        assert 'stable_count' in loaded
