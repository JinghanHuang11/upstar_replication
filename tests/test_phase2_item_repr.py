"""
test_phase2_item_repr.py — Phase 2 真实执行测试

测试目标：Item-GNN 表征学习
- 依赖检查（metadata.pkl 存在性）
- execute() 真实调用 train_item_repr.main()
- 验证 item_embeddings.pt 输出
"""

import pytest
from pathlib import Path


class TestPhase2Dependencies:
    def test_check_dependencies_missing_metadata(self, phase2):
        """metadata.pkl 不存在时应返回 False"""
        assert phase2.check_dependencies() is False

    def test_check_dependencies_with_metadata(self, phase2, phase0):
        """Phase 0 执行后，依赖检查通过"""
        phase0.execute()
        assert phase2.check_dependencies() is True


class TestPhase2Execution:
    def test_execute_runs_item_repr(self, phase2, phase0):
        """execute() 真实调用 train_item_repr.main()"""
        phase0.execute()
        results = phase2.execute()

        assert 'embedding_dim' in results
        assert 'num_items' in results
        assert 'embedding_path' in results
        assert results['num_items'] > 0

    def test_execute_creates_embeddings_file(self, phase2, phase0, phase2_config):
        """execute() 应生成 item_embeddings.pt"""
        phase0.execute()
        phase2.execute()
        emb_path = Path(phase2_config['stb']['item_embeddings_path'])
        assert emb_path.exists(), f"item_embeddings.pt not found at {emb_path}"

    def test_execute_is_completed_after_run(self, phase2, phase0):
        """执行后 is_completed() 应返回 True"""
        phase0.execute()
        phase2.execute()
        assert phase2.is_completed() is True

    def test_execute_saves_results(self, phase2, phase0):
        """执行后 results.json 应包含 embedding 信息"""
        phase0.execute()
        phase2.execute()
        loaded = phase2.load_results()
        assert loaded is not None
        assert 'embedding_path' in loaded
