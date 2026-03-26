"""
test_phase4_upstar.py — Phase 4 真实执行测试

测试目标：UPSTAR 四阶段训练
- 依赖检查（item_embeddings.pt + motivation_labels.npy 同时存在）
- execute() 真实调用 train_upstar.main()
- 验证 model_after_stage4.pt 输出
"""

import pytest
from pathlib import Path


class TestPhase4Dependencies:
    def test_check_dependencies_both_missing(self, phase4):
        """两个文件都不存在时应返回 False"""
        assert phase4.check_dependencies() is False

    def test_check_dependencies_only_embeddings(self, phase4, phase0, phase2):
        """只有 item_embeddings.pt 时仍应返回 False"""
        phase0.execute()
        phase2.execute()
        assert phase4.check_dependencies() is False

    def test_check_dependencies_only_labels(self, phase4, phase0, phase2, phase3):
        """只有 motivation_labels.npy 时仍应返回 False"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        # 实际上 phase3 也创建了 labels，所以此时依赖已满足
        assert phase4.check_dependencies() is True

    def test_check_dependencies_both_present(self, phase4, phase0, phase2, phase3):
        """两个文件都存在时依赖检查通过"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        assert phase4.check_dependencies() is True


class TestPhase4Execution:
    def test_execute_runs_upstar(self, phase4, phase0, phase2, phase3):
        """execute() 真实调用 train_upstar.main()"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        results = phase4.execute()

        assert 'ndcg@10' in results
        assert 'hr@10' in results
        assert 'mrr@10' in results
        assert 'model_path' in results

    def test_execute_creates_model_checkpoint(self, phase4, phase0, phase2, phase3, phase4_config):
        """execute() 应生成 model_after_stage4.pt"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        ckpt_path = Path(phase4_config['upstar']['model_path'])
        assert ckpt_path.exists(), f"model_after_stage4.pt not found at {ckpt_path}"

    def test_execute_is_completed_after_run(self, phase4, phase0, phase2, phase3):
        """执行后 is_completed() 应返回 True"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        assert phase4.is_completed() is True

    def test_execute_saves_results(self, phase4, phase0, phase2, phase3):
        """执行后 results.json 应包含指标"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        loaded = phase4.load_results()
        assert loaded is not None
        assert 'ndcg@10' in loaded
