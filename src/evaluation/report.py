"""
Report Generation for UPSTAR Evaluation

Generates:
- Main results table (P@5, P@20, NDCG@5, NDCG@20, MRR@5, MRR@20)
- Full table (all k values)
- Multiple formats: JSON, CSV, TXT, LaTeX
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate evaluation reports in multiple formats
    """

    def __init__(
        self,
        metrics: Dict[str, float],
        cv_stats: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize report generator

        Args:
            metrics: dict of metric names and values (0-1 scale)
            cv_stats: optional cross-validation stats with mean and std
        """
        self.metrics = metrics
        self.cv_stats = cv_stats

    def generate_main_table(self, format: str = 'text') -> str:
        """
        Generate main results table

        Metrics: P@5, P@20, NDCG@5, NDCG@20, MRR@5, MRR@20

        Args:
            format: 'text', 'latex', or 'markdown'

        Returns:
            table_str: formatted table string
        """
        main_metrics = ['Precision@5', 'Precision@20',
                       'NDCG@5', 'NDCG@20',
                       'MRR@5', 'MRR@20']

        if format == 'text':
            return self._generate_text_table(main_metrics, "Main Results")
        elif format == 'latex':
            return self._generate_latex_table(main_metrics, "Main Results")
        elif format == 'markdown':
            return self._generate_markdown_table(main_metrics, "Main Results")
        else:
            raise ValueError(f"Unknown format: {format}")

    def generate_full_table(self, format: str = 'text') -> str:
        """
        Generate full results table (all k values)

        Args:
            format: 'text', 'latex', or 'markdown'

        Returns:
            table_str: formatted table string
        """
        # Get all metric names
        if self.cv_stats:
            metric_names = sorted(self.cv_stats.keys())
        else:
            metric_names = sorted(self.metrics.keys())

        if format == 'text':
            return self._generate_text_table(metric_names, "Full Results")
        elif format == 'latex':
            return self._generate_latex_table(metric_names, "Full Results")
        elif format == 'markdown':
            return self._generate_markdown_table(metric_names, "Full Results")
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_text_table(
        self,
        metric_names: List[str],
        title: str
    ) -> str:
        """Generate text format table"""
        lines = []
        lines.append(f"\n{title}")
        lines.append("=" * 80)

        if self.cv_stats:
            # Cross-validation results
            lines.append(f"{'Metric':<15} {'Mean':<15} {'Std':<15}")
            lines.append("-" * 80)

            for metric_name in metric_names:
                if metric_name in self.cv_stats:
                    mean_val = self.cv_stats[metric_name]['mean'] * 100
                    std_val = self.cv_stats[metric_name]['std'] * 100
                    lines.append(f"{metric_name:<15} {mean_val:6.2f}%        {std_val:6.2f}%")
        else:
            # Single run results
            lines.append(f"{'Metric':<15} {'Value':<15}")
            lines.append("-" * 80)

            for metric_name in metric_names:
                if metric_name in self.metrics:
                    value = self.metrics[metric_name] * 100
                    lines.append(f"{metric_name:<15} {value:6.2f}%")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _generate_latex_table(
        self,
        metric_names: List[str],
        title: str
    ) -> str:
        """Generate LaTeX format table"""
        lines = []

        lines.append(f"% {title}")
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")

        if self.cv_stats:
            # Cross-validation results
            lines.append("\\begin{tabular}{lcc}")
            lines.append("\\hline")
            lines.append("Metric & Mean (\\%) & Std (\\%) \\\\")
            lines.append("\\hline")

            for metric_name in metric_names:
                if metric_name in self.cv_stats:
                    mean_val = self.cv_stats[metric_name]['mean'] * 100
                    std_val = self.cv_stats[metric_name]['std'] * 100
                    # Replace @ with \\@
                    latex_name = metric_name.replace('@', '\\@')
                    lines.append(f"{latex_name} & {mean_val:.2f} & {std_val:.2f} \\\\")

            lines.append("\\hline")
            lines.append("\\end{tabular}")
            lines.append(f"\\caption{{{title}}}")
        else:
            # Single run results
            lines.append("\\begin{tabular}{lc}")
            lines.append("\\hline")
            lines.append("Metric & Value (\\%) \\\\")
            lines.append("\\hline")

            for metric_name in metric_names:
                if metric_name in self.metrics:
                    value = self.metrics[metric_name] * 100
                    latex_name = metric_name.replace('@', '\\@')
                    lines.append(f"{latex_name} & {value:.2f} \\\\")

            lines.append("\\hline")
            lines.append("\\end{tabular}")
            lines.append(f"\\caption{{{title}}}")

        lines.append("\\end{table}")

        return "\n".join(lines)

    def _generate_markdown_table(
        self,
        metric_names: List[str],
        title: str
    ) -> str:
        """Generate Markdown format table"""
        lines = []
        lines.append(f"\n## {title}\n")

        if self.cv_stats:
            # Cross-validation results
            lines.append("| Metric | Mean (%) | Std (%) |")
            lines.append("|--------|----------|----------|")

            for metric_name in metric_names:
                if metric_name in self.cv_stats:
                    mean_val = self.cv_stats[metric_name]['mean'] * 100
                    std_val = self.cv_stats[metric_name]['std'] * 100
                    lines.append(f"| {metric_name} | {mean_val:.2f} | {std_val:.2f} |")
        else:
            # Single run results
            lines.append("| Metric | Value (%) |")
            lines.append("|--------|-----------|")

            for metric_name in metric_names:
                if metric_name in self.metrics:
                    value = self.metrics[metric_name] * 100
                    lines.append(f"| {metric_name} | {value:.2f} |")

        return "\n".join(lines)

    def save_json(self, output_path: str):
        """Save results as JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to percentage
        if self.cv_stats:
            results = {}
            for metric_name, stats in self.cv_stats.items():
                results[metric_name] = {
                    'mean': stats['mean'] * 100,
                    'std': stats['std'] * 100
                }
        else:
            results = {k: v * 100 for k, v in self.metrics.items()}

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved JSON to {output_path}")

    def save_csv(self, output_path: str):
        """Save results as CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            if self.cv_stats:
                # Cross-validation results
                writer.writerow(['Metric', 'Mean (%)', 'Std (%)'])
                for metric_name, stats in self.cv_stats.items():
                    writer.writerow([
                        metric_name,
                        f"{stats['mean'] * 100:.2f}",
                        f"{stats['std'] * 100:.2f}"
                    ])
            else:
                # Single run results
                writer.writerow(['Metric', 'Value (%)'])
                for metric_name, value in self.metrics.items():
                    writer.writerow([metric_name, f"{value * 100:.2f}"])

        logger.info(f"Saved CSV to {output_path}")

    def save_txt(self, output_path: str, main_only: bool = False):
        """Save results as TXT"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            if main_only:
                f.write(self.generate_main_table(format='text'))
            else:
                f.write(self.generate_full_table(format='text'))

        logger.info(f"Saved TXT to {output_path}")

    def save_all_formats(self, output_dir: str):
        """Save results in all formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main table
        self.save_json(output_dir / 'main_results.json')
        self.save_csv(output_dir / 'main_results.csv')
        self.save_txt(output_dir / 'main_results.txt', main_only=True)

        # Save full table
        self.save_txt(output_dir / 'full_results.txt', main_only=False)

        # Save LaTeX (optional)
        latex_path = output_dir / 'main_results.tex'
        with open(latex_path, 'w') as f:
            f.write(self.generate_main_table(format='latex'))
        logger.info(f"Saved LaTeX to {latex_path}")

        # Save Markdown
        md_path = output_dir / 'main_results.md'
        with open(md_path, 'w') as f:
            f.write(self.generate_main_table(format='markdown'))
        logger.info(f"Saved Markdown to {md_path}")

        logger.info(f"\nAll results saved to {output_dir}")


def generate_report(
    metrics: Dict[str, float],
    output_dir: str,
    cv_stats: Optional[Dict[str, Dict[str, float]]] = None
):
    """
    Convenience function to generate all reports

    Args:
        metrics: dict of metric names and values (0-1 scale)
        output_dir: directory to save reports
        cv_stats: optional cross-validation stats
    """
    generator = ReportGenerator(metrics, cv_stats)
    generator.save_all_formats(output_dir)


if __name__ == '__main__':
    print("Testing Report Generator...")

    # Test with single run metrics
    metrics = {
        'Precision@5': 0.0523,
        'Precision@20': 0.1234,
        'NDCG@5': 0.0612,
        'NDCG@20': 0.1456,
        'MRR@5': 0.0412,
        'MRR@20': 0.0891
    }

    generator = ReportGenerator(metrics)

    print("\nText table:")
    print(generator.generate_main_table(format='text'))

    print("\nLaTeX table:")
    print(generator.generate_main_table(format='latex'))

    print("\nMarkdown table:")
    print(generator.generate_main_table(format='markdown'))

    # Test with CV stats
    cv_stats = {
        'NDCG@10': {'mean': 0.0967, 'std': 0.0023},
        'Precision@5': {'mean': 0.0523, 'std': 0.0015}
    }

    generator_cv = ReportGenerator({}, cv_stats)

    print("\nCV text table:")
    print(generator_cv.generate_main_table(format='text'))

    print("\nReport generator module loaded successfully!")
