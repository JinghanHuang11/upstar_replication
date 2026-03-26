"""
Baseline vs UPSTAR Comparison

Generates comparison reports for experimental results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare baseline and UPSTAR results
    """

    def __init__(
        self,
        baseline_results_dir: str,
        upstar_results_dir: str,
        output_dir: Optional[str] = None
    ):
        """
        Initialize comparator

        Args:
            baseline_results_dir: path to baseline results
            upstar_results_dir: path to UPSTAR results
            output_dir: directory to save comparison
        """
        self.baseline_dir = Path(baseline_results_dir)
        self.upstar_dir = Path(upstar_results_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/results/comparison")

        # Load results
        self.baseline_metrics = self._load_results(self.baseline_dir)
        self.upstar_metrics = self._load_results(self.upstar_dir)

    def _load_results(self, results_dir: Path) -> Dict[str, float]:
        """Load main results from directory"""
        main_results_path = results_dir / "main_results.json"

        if main_results_path.exists():
            with open(main_results_path, 'r') as f:
                data = json.load(f)
                return data.get('metrics', {})
        else:
            logger.warning(f"Main results not found: {main_results_path}")
            return {}

    def compare(self) -> Dict:
        """
        Compare baseline and UPSTAR

        Returns:
            comparison: dict with comparison data
        """
        # Get common metrics
        baseline_metrics = self.baseline_metrics
        upstar_metrics = self.upstar_metrics

        comparison = {
            'baseline': {},
            'upstar': {},
            'improvement': {},
            'relative_improvement': {}
        }

        # Main table metrics
        main_metrics = [
            'Precision@5', 'Precision@20',
            'NDCG@5', 'NDCG@20',
            'MRR@5', 'MRR@20'
        ]

        for metric in main_metrics:
            baseline_val = baseline_metrics.get(metric, 0.0)
            upstar_val = upstar_metrics.get(metric, 0.0)

            comparison['baseline'][metric] = baseline_val * 100
            comparison['upstar'][metric] = upstar_val * 100
            comparison['improvement'][metric] = (upstar_val - baseline_val) * 100
            comparison['relative_improvement'][metric] = (
                ((upstar_val - baseline_val) / (baseline_val + 1e-8)) * 100
            )

        return comparison

    def generate_comparison_table(self, format: str = 'text') -> str:
        """
        Generate comparison table

        Args:
            format: 'text', 'markdown', or 'latex'

        Returns:
            table_str: formatted table
        """
        comparison = self.compare()

        if format == 'text':
            return self._generate_text_table(comparison)
        elif format == 'markdown':
            return self._generate_markdown_table(comparison)
        elif format == 'latex':
            return self._generate_latex_table(comparison)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_text_table(self, comparison: Dict) -> str:
        """Generate text format table"""
        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("Baseline vs UPSTAR Comparison (Ta-Feng)")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"{'Metric':<15} {'Baseline':<15} {'UPSTAR':<15} {'Improvement':<15} {'Rel. Imp.':<15}")
        lines.append("-" * 100)

        for metric in ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']:
            if metric in comparison['baseline']:
                baseline_val = comparison['baseline'][metric]
                upstar_val = comparison['upstar'][metric]
                improvement = comparison['improvement'][metric]
                rel_imp = comparison['relative_improvement'][metric]

                lines.append(
                    f"{metric:<15} {baseline_val:>6.2f}%       {upstar_val:>6.2f}%       "
                    f"{improvement:>7.2f}%      {rel_imp:>7.2f}%"
                )

        lines.append("=" * 100)
        lines.append("")

        return "\n".join(lines)

    def _generate_markdown_table(self, comparison: Dict) -> str:
        """Generate Markdown table"""
        lines = []
        lines.append("\n## Baseline vs UPSTAR Comparison (Ta-Feng)\n")
        lines.append("| Metric | Baseline | UPSTAR | Improvement | Rel. Imp. |")
        lines.append("|--------|----------|--------|-------------|-----------|")

        for metric in ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']:
            if metric in comparison['baseline']:
                baseline_val = comparison['baseline'][metric]
                upstar_val = comparison['upstar'][metric]
                improvement = comparison['improvement'][metric]
                rel_imp = comparison['relative_improvement'][metric]

                lines.append(
                    f"| {metric} | {baseline_val:.2f}% | {upstar_val:.2f}% | "
                    f"{improvement:+.2f}% | {rel_imp:+.2f}% |"
                )

        return "\n".join(lines)

    def _generate_latex_table(self, comparison: Dict) -> str:
        """Generate LaTeX table"""
        lines = []
        lines.append("% Baseline vs UPSTAR Comparison (Ta-Feng)")
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lccccc}")
        lines.append("\\hline")
        lines.append("Metric & Baseline (\\%) & UPSTAR (\\%) & Improvement (\\%) & Rel. Imp. (\\%) \\\\")
        lines.append("\\hline")

        for metric in ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']:
            if metric in comparison['baseline']:
                baseline_val = comparison['baseline'][metric]
                upstar_val = comparison['upstar'][metric]
                improvement = comparison['improvement'][metric]
                rel_imp = comparison['relative_improvement'][metric]

                latex_metric = metric.replace('@', '\\@')
                lines.append(f"{latex_metric} & {baseline_val:.2f} & {upstar_val:.2f} & "
                             f"{improvement:+.2f} & {rel_imp:+.2f} \\\\")

        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Baseline vs UPSTAR Comparison on Ta-Feng Dataset}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def save_comparison(self):
        """Save comparison in all formats"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate comparison
        comparison = self.compare()

        # Save JSON
        json_path = self.output_dir / 'comparison.json'
        with open(json_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison to {json_path}")

        # Save CSV
        csv_path = self.output_dir / 'comparison.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Baseline (%)', 'UPSTAR (%)', 'Improvement (%)', 'Rel. Imp. (%)'])

            for metric in ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']:
                if metric in comparison['baseline']:
                    writer.writerow([
                        metric,
                        f"{comparison['baseline'][metric]:.2f}",
                        f"{comparison['upstar'][metric]:.2f}",
                        f"{comparison['improvement'][metric]:.2f}",
                        f"{comparison['relative_improvement'][metric]:.2f}"
                    ])
        logger.info(f"Saved CSV to {csv_path}")

        # Save TXT
        txt_path = self.output_dir / 'comparison.txt'
        with open(txt_path, 'w') as f:
            f.write(self.generate_comparison_table(format='text'))
        logger.info(f"Saved TXT to {txt_path}")

        # Save Markdown
        md_path = self.output_dir / 'comparison.md'
        with open(md_path, 'w') as f:
            f.write(self.generate_comparison_table(format='markdown'))
        logger.info(f"Saved Markdown to {md_path}")

        # Save LaTeX
        latex_path = self.output_dir / 'comparison.tex'
        with open(latex_path, 'w') as f:
            f.write(self.generate_comparison_table(format='latex'))
        logger.info(f"Saved LaTeX to {latex_path}")

        # Save improvement analysis
        self._save_improvement_analysis()

        logger.info(f"\nAll comparison results saved to {self.output_dir}")

    def _save_improvement_analysis(self):
        """Save detailed improvement analysis"""
        comparison = self.compare()

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("Improvement Analysis")
        lines.append("=" * 80)
        lines.append("")

        # Overall improvement
        main_metrics = ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']

        improvements = [comparison['improvement'][m] for m in main_metrics if m in comparison['improvement']]
        avg_improvement = sum(improvements) / len(improvements)

        lines.append(f"Average Improvement (Main Metrics): {avg_improvement:.2f}%")
        lines.append("")

        # Best and worst improvements
        best_metric = max(comparison['improvement'].items(), key=lambda x: x[1])
        worst_metric = min(comparison['improvement'].items(), key=lambda x: x[1])

        lines.append(f"Best Improvement: {best_metric[0]} ({best_metric[1]:+.2f}%)")
        lines.append(f"Worst Improvement: {worst_metric[0]} ({worst_metric[1]:+.2f}%)")
        lines.append("")

        # Significant improvements (>5%)
        lines.append("Significant Improvements (>5%):")
        for metric, imp in comparison['improvement'].items():
            if imp > 5.0:
                lines.append(f"  {metric}: {imp:+.2f}%")
        lines.append("")

        # Degradations
        lines.append("Degradations (<0%):")
        has_degradation = False
        for metric, imp in comparison['improvement'].items():
            if imp < 0:
                lines.append(f"  {metric}: {imp:+.2f}%")
                has_degradation = True
        if not has_degradation:
            lines.append("  None")
        lines.append("")

        lines.append("=" * 80)

        # Save
        analysis_path = self.output_dir / 'improvement_analysis.txt'
        with open(analysis_path, 'w') as f:
            f.write("\n".join(lines))
        logger.info(f"Saved improvement analysis to {analysis_path}")

    def generate_executive_summary(self) -> str:
        """Generate executive summary"""
        comparison = self.compare()

        summary = []
        summary.append("\n" + "=" * 80)
        summary.append("EXECUTIVE SUMMARY: Baseline vs UPSTAR")
        summary.append("=" * 80)
        summary.append("")

        # Key results
        summary.append("Key Metrics:")
        summary.append("-" * 40)

        key_metrics = {
            'NDCG@10': 'Primary ranking metric',
            'NDCG@20': 'Long-tail performance',
            'Precision@5': 'Top-5 accuracy',
            'Recall@20': 'Recall in top-20'
        }

        for metric, description in key_metrics.items():
            # Try main metrics, then full metrics
            if metric in comparison['baseline']:
                baseline_val = comparison['baseline'][metric]
                upstar_val = comparison['upstar'][metric]
                improvement = comparison['improvement'][metric]
                summary.append(f"  {metric}: {baseline_val:.2f}% → {upstar_val:.2f}% ({improvement:+.2f}%)")
            elif metric in self.baseline_metrics:
                baseline_val = self.baseline_metrics[metric] * 100
                upstar_val = self.upstar_metrics[metric] * 100
                improvement = upstar_val - baseline_val
                summary.append(f"  {metric}: {baseline_val:.2f}% → {upstar_val:.2f}% ({improvement:+.2f}%)")

        summary.append("")

        # Overall assessment
        improvements = [comparison['improvement'][m] for m in main_metrics if m in comparison['improvement']]
        avg_improvement = sum(improvements) / len(improvements)

        summary.append("Overall Assessment:")
        summary.append("-" * 40)
        summary.append(f"  Average improvement: {avg_improvement:.2f}%")

        if avg_improvement > 5:
            summary.append("  Assessment: ✅ SIGNIFICANT IMPROVEMENT")
        elif avg_improvement > 2:
            summary.append("  Assessment: ⚠️  MODERATE IMPROVEMENT")
        elif avg_improvement > 0:
            summary.append("  Assessment: ✅ MARGINAL IMPROVEMENT")
        else:
            summary.append("  Assessment: ❌ NO IMPROVEMENT")

        summary.append("")
        summary.append("=" * 80)

        return "\n".join(summary)


def compare_results(
    baseline_dir: str,
    upstar_dir: str,
    output_dir: str = "outputs/results/comparison"
):
    """
    Convenience function to compare baseline and UPSTAR

    Args:
        baseline_dir: path to baseline results
        upstar_dir: path to UPSTAR results
        output_dir: where to save comparison
    """
    comparator = ModelComparator(baseline_dir, upstar_dir, output_dir)
    comparator.save_comparison()

    # Print executive summary
    print(comparator.generate_executive_summary())

    return comparator


if __name__ == '__main__':
    print("Testing Model Comparator...")

    # Create dummy results
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baseline"
        upstar_dir = Path(tmpdir) / "upstar"

        baseline_dir.mkdir()
        upstar_dir.mkdir()

        # Create dummy results
        baseline_results = {
            "model_type": "baseline",
            "metrics": {
                "Precision@5": 0.0523,
                "Precision@20": 0.1234,
                "NDCG@5": 0.0612,
                "NDCG@20": 0.1456,
                "MRR@5": 0.0412,
                "MRR@20": 0.0891,
                "NDCG@10": 0.0967
            }
        }

        upstar_results = {
            "model_type": "upstar",
            "metrics": {
                "Precision@5": 0.0567,
                "Precision@20": 0.1312,
                "NDCG@5": 0.0667,
                "NDCG@20": 0.1523,
                "MRR@5": 0.0456,
                "MRR@20": 0.0945,
                "NDCG@10": 0.1034
            }
        }

        # Save
        with open(baseline_dir / "main_results.json", 'w') as f:
            json.dump(baseline_results, f)

        with open(upstar_dir / "main_results.json", 'w') as f:
            json.dump(upstar_results, f)

        # Test comparator
        comparator = ModelComparator(str(baseline_dir), str(upstar_dir), str(tmpdir / "comparison"))
        comparator.save_comparison()

        print("\nComparison generated successfully!")
        print(comparator.generate_executive_summary())

    print("\nModel comparator module loaded successfully!")
