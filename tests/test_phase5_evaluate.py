"""
test_phase5_evaluate.py — Phase 5 真实执行测试

测试目标：Baseline vs UPSTAR 评估对比
- 依赖检查（baseline + upstar 模型文件同时存在）
- execute() 真实加载 Phase1/Phase4 结果并计算改进百分比
- 验证 comparison_table.txt 和 comparison.json 输出
"""

import pytest
from pathlib import Path
import json


class TestPhase5Dependencies:
    def test_check_dependencies_both_missing(self, phase5):
        """两个模型都不存在时应返回 False"""
        assert phase5.check_dependencies() is False

    def test_check_dependencies_only_baseline(self, phase5, phase0, phase1):
        """只有 baseline 模型时应返回 False"""
        phase0.execute()
        phase1.execute()
        assert phase5.check_dependencies() is False

    def test_check_dependencies_only_upstar(self, phase5, phase0, phase2, phase3, phase4):
        """只有 UPSTAR 模型时应返回 False"""
        phase0.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        assert phase5.check_dependencies() is False

    def test_check_dependencies_both_present(self, phase5, phase0, phase1, phase2, phase3, phase4):
        """两个模型都存在时依赖检查通过"""
        phase0.execute()
        phase1.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        assert phase5.check_dependencies() is True


class TestPhase5Execution:
    def test_execute_returns_comparison_structure(self, phase5, phase0, phase1, phase2, phase3, phase4):
        """execute() 结果应包含 baseline / upstar / comparison"""
        phase0.execute()
        phase1.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        results = phase5.execute()

        assert 'baseline' in results
        assert 'upstar' in results
        assert 'comparison' in results

    def test_execute_comparison_has_all_metrics(self, phase5, phase0, phase1, phase2, phase3, phase4):
        """comparison 应包含 ndcg@10 / hr@10 / mrr@10"""
        phase0.execute()
        phase1.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        results = phase5.execute()
        comparison = results['comparison']

        assert 'ndcg@10' in comparison
        assert 'hr@10' in comparison
        assert 'mrr@10' in comparison

    def test_execute_comparison_fields(self, phase5, phase0, phase1, phase2, phase3, phase4):
        """每个 metric 的 comparison 条目应包含 baseline / upstar / improvement_pct"""
        phase0.execute()
        phase1.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        results = phase5.execute()

        for metric in ['ndcg@10', 'hr@10', 'mrr@10']:
            entry = results['comparison'][metric]
            assert 'baseline' in entry
            assert 'upstar' in entry
            assert 'improvement_pct' in entry
            assert 'absolute_diff' in entry

    def test_execute_saves_comparison_table(self, phase5, phase0, phase1, phase2, phase3, phase4):
        """应生成 comparison_table.txt"""
        phase0.execute()
        phase1.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        phase5.execute()

        txt_file = phase5.results_dir / "comparison_table.txt"
        assert txt_file.exists(), f"comparison_table.txt not found at {txt_file}"
        content = txt_file.read_text()
        assert 'Baseline' in content
        assert 'UPSTAR' in content

    def test_execute_saves_comparison_json(self, phase5, phase0, phase1, phase2, phase3, phase4):
        """应生成 comparison.json"""
        phase0.execute()
        phase1.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        phase5.execute()

        json_file = phase5.results_dir / "comparison.json"
        assert json_file.exists(), f"comparison.json not found at {json_file}"

        with open(json_file) as f:
            data = json.load(f)
        assert 'ndcg@10' in data
        assert 'improvement_pct' in data['ndcg@10']

    def test_execute_is_completed_after_run(self, phase5, phase0, phase1, phase2, phase3, phase4):
        """执行后 is_completed() 应返回 True"""
        phase0.execute()
        phase1.execute()
        phase2.execute()
        phase3.execute()
        phase4.execute()
        phase5.execute()
        assert phase5.is_completed() is True
