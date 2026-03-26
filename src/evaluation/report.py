"""
Report Generation for UPSTAR Evaluation

Generates:
- Main results table (P@5, P@20, NDCG@5, NDCG@20, MRR@5, MRR@20) — paper Table 2/3
- Additional engineering metrics (@1, @10, @15, Recall@*, etc.)
- Multiple formats: JSON, CSV, TXT, LaTeX

Input metrics dict is structured as:
    {
        'main_metrics':      {...},   # paper Table 2/3: P@5, P@20, NDCG@5, NDCG@20, MRR@5, MRR@20
        'additional_metrics': {...},   # engineering extras: @1, @10, @15, Recall@*, etc.
        '_flat':             {...}    # backward-compat: all metrics merged
    }
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Paper main metrics — must match PAPER_MAIN_METRICS in evaluator.py
PAPER_MAIN_METRICS = ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']


class ReportGenerator:
    """
    Generate evaluation reports in multiple formats
    """

    def __init__(
        self,
        metrics: Union[Dict[str, float], Dict[str, Dict[str, float]]],
        cv_stats: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize report generator

        Args:
            metrics: either a flat dict {metric_name: value} (backward compat),
                     or a structured dict with 'main_metrics' and 'additional_metrics'.
                     Backward-compat: a flat dict is treated as all-main.
            cv_stats: optional cross-validation stats with mean and std.
                       For structured metrics, this should also be structured:
                       {'main_metrics': {name: {mean, std}}, 'additional_metrics': {name: {mean, std}}}
        """
        # Detect structured vs flat input
        if 'main_metrics' in metrics or 'additional_metrics' in metrics:
            self.main_metrics = metrics.get('main_metrics', {})
            self.additional_metrics = metrics.get('additional_metrics', {})
            self.metrics = {**self.main_metrics, **self.additional_metrics}  # flat merged
        else:
            # Backward compat: treat all as main
            self.main_metrics = dict(metrics)
            self.additional_metrics = {}
            self.metrics = dict(metrics)
        self.cv_stats = cv_stats

    def generate_main_table(self, format: str = 'text') -> str:
        """
        Generate main results table (paper Table 2/3 format)

        Always uses: P@5, P@20, NDCG@5, NDCG@20, MRR@5, MRR@20

        Args:
            format: 'text', 'latex', or 'markdown'

        Returns:
            table_str: formatted table string
        """
        if format == 'text':
            return self._generate_text_table(PAPER_MAIN_METRICS, "Main Results (Paper Table)")
        elif format == 'latex':
            return self._generate_latex_table(PAPER_MAIN_METRICS, "Main Results (Paper Table)")
        elif format == 'markdown':
            return self._generate_markdown_table(PAPER_MAIN_METRICS, "Main Results (Paper Table)")
        else:
            raise ValueError(f"Unknown format: {format}")

    def generate_full_table(self, format: str = 'text') -> str:
        """
        Generate full results table (main + additional)

        Shows paper main table followed by additional engineering metrics.

        Args:
            format: 'text', 'latex', or 'markdown'

        Returns:
            table_str: formatted table string
        """
        if format == 'text':
            lines = []
            lines.append(self._generate_text_table(PAPER_MAIN_METRICS, "Main Results (Paper Table)"))
            if self.additional_metrics:
                lines.append("\n")
                additional_names = sorted(self.additional_metrics.keys())
                lines.append(self._generate_text_table(additional_names, "Additional Engineering Metrics"))
            return "\n".join(lines)
        elif format == 'latex':
            sections = []
            sections.append(self._generate_latex_table(PAPER_MAIN_METRICS, "Main Results (Paper Table)"))
            if self.additional_metrics:
                additional_names = sorted(self.additional_metrics.keys())
                sections.append("\n" + self._generate_latex_table(additional_names, "Additional Engineering Metrics"))
            return "\n".join(sections)
        elif format == 'markdown':
            sections = []
            sections.append(self._generate_markdown_table(PAPER_MAIN_METRICS, "Main Results (Paper Table)"))
            if self.additional_metrics:
                additional_names = sorted(self.additional_metrics.keys())
                sections.append("\n" + self._generate_markdown_table(additional_names, "Additional Engineering Metrics"))
            return "\n".join(sections)
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

            # cv_stats is either structured {'main_metrics':{...},'additional_metrics':{...}}
            # or flat {metric_name: {mean, std}}
            if 'main_metrics' in self.cv_stats or 'additional_metrics' in self.cv_stats:
                cv_data = {
                    **self.cv_stats.get('main_metrics', {}),
                    **self.cv_stats.get('additional_metrics', {})
                }
            else:
                cv_data = self.cv_stats

            for metric_name in metric_names:
                if metric_name in cv_data:
                    mean_val = cv_data[metric_name]['mean'] * 100
                    std_val = cv_data[metric_name]['std'] * 100
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
            if 'main_metrics' in self.cv_stats or 'additional_metrics' in self.cv_stats:
                cv_data = {
                    **self.cv_stats.get('main_metrics', {}),
                    **self.cv_stats.get('additional_metrics', {})
                }
            else:
                cv_data = self.cv_stats

            lines.append("\\begin{tabular}{lcc}")
            lines.append("\\hline")
            lines.append("Metric & Mean (\\%) & Std (\\%) \\\\")
            lines.append("\\hline")

            for metric_name in metric_names:
                if metric_name in cv_data:
                    mean_val = cv_data[metric_name]['mean'] * 100
                    std_val = cv_data[metric_name]['std'] * 100
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
            if 'main_metrics' in self.cv_stats or 'additional_metrics' in self.cv_stats:
                cv_data = {
                    **self.cv_stats.get('main_metrics', {}),
                    **self.cv_stats.get('additional_metrics', {})
                }
            else:
                cv_data = self.cv_stats

            lines.append("| Metric | Mean (%) | Std (%) |")
            lines.append("|--------|----------|----------|")

            for metric_name in metric_names:
                if metric_name in cv_data:
                    mean_val = cv_data[metric_name]['mean'] * 100
                    std_val = cv_data[metric_name]['std'] * 100
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
        """Save results as JSON — structured with main + additional sections"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cv_stats:
            # Cross-validation: structured with main + additional
            def convert_cv(stats_dict):
                return {k: {'mean': v['mean'] * 100, 'std': v['std'] * 100} for k, v in stats_dict.items()}

            # Try to split cv_stats into main / additional
            if isinstance(self.cv_stats, dict) and 'main_metrics' in self.cv_stats:
                cv_main = convert_cv(self.cv_stats.get('main_metrics', {}))
                cv_additional = convert_cv(self.cv_stats.get('additional_metrics', {}))
            else:
                # Backward compat: split by metric name
                cv_main = convert_cv({k: v for k, v in self.cv_stats.items() if k in PAPER_MAIN_METRICS})
                cv_additional = convert_cv({k: v for k, v in self.cv_stats.items() if k not in PAPER_MAIN_METRICS})

            results = {
                'main_metrics': cv_main,
                'additional_metrics': cv_additional
            }
        else:
            # Single run: main + additional (already in percentage)
            results = {
                'main_metrics':  {k: v * 100 for k, v in self.main_metrics.items()},
                'additional_metrics': {k: v * 100 for k, v in self.additional_metrics.items()}
            }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved JSON to {output_path}")

    def save_csv(self, output_path: str):
        """Save results as CSV — main table first, then additional metrics"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Main table (paper format)
            writer.writerow(['# Paper Main Table'])
            if self.cv_stats:
                # CV main — structured or flat
                if 'main_metrics' in self.cv_stats:
                    cv_main = {k: v for k, v in self.cv_stats['main_metrics'].items()
                               if k in PAPER_MAIN_METRICS}
                    cv_add = dict(self.cv_stats.get('additional_metrics', {}))
                else:
                    cv_main = {k: v for k, v in self.cv_stats.items()
                               if k in PAPER_MAIN_METRICS}
                    cv_add = {k: v for k, v in self.cv_stats.items()
                              if k not in PAPER_MAIN_METRICS}
                writer.writerow(['Metric', 'Mean (%)', 'Std (%)'])
                for metric_name in PAPER_MAIN_METRICS:
                    if metric_name in cv_main:
                        s = cv_main[metric_name]
                        writer.writerow([metric_name, f"{s['mean']*100:.2f}", f"{s['std']*100:.2f}"])
                if cv_add:
                    writer.writerow([])
                    writer.writerow(['# Additional Engineering Metrics'])
                    writer.writerow(['Metric', 'Mean (%)', 'Std (%)'])
                    for metric_name, s in sorted(cv_add.items()):
                        writer.writerow([metric_name, f"{s['mean']*100:.2f}", f"{s['std']*100:.2f}"])
            else:
                # Single run main
                writer.writerow(['Metric', 'Value (%)'])
                for metric_name in PAPER_MAIN_METRICS:
                    if metric_name in self.main_metrics:
                        writer.writerow([metric_name, f"{self.main_metrics[metric_name]*100:.2f}"])
                # Single run additional
                if self.additional_metrics:
                    writer.writerow([])
                    writer.writerow(['# Additional Engineering Metrics'])
                    writer.writerow(['Metric', 'Value (%)'])
                    for metric_name, value in sorted(self.additional_metrics.items()):
                        writer.writerow([metric_name, f"{value*100:.2f}"])

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

    # Test with structured metrics (new format)
    structured_metrics = {
        'main_metrics': {
            'Precision@5': 0.0523,
            'Precision@20': 0.1234,
            'NDCG@5': 0.0612,
            'NDCG@20': 0.1456,
            'MRR@5': 0.0412,
            'MRR@20': 0.0891
        },
        'additional_metrics': {
            'NDCG@10': 0.0967,
            'Recall@5': 0.0523,
            'Recall@20': 0.1234,
        }
    }

    generator = ReportGenerator(structured_metrics)

    print("\n=== Text main table ===")
    print(generator.generate_main_table(format='text'))

    print("\n=== Text full table ===")
    print(generator.generate_full_table(format='text'))

    print("\n=== LaTeX main table ===")
    print(generator.generate_main_table(format='latex'))

    print("\n=== Markdown main table ===")
    print(generator.generate_main_table(format='markdown'))

    # Test backward compat: flat dict
    print("\n=== Backward compat (flat dict) ===")
    flat_metrics = {
        'Precision@5': 0.0523,
        'Precision@20': 0.1234,
        'NDCG@5': 0.0612,
        'NDCG@20': 0.1456,
        'MRR@5': 0.0412,
        'MRR@20': 0.0891,
        'NDCG@10': 0.0967
    }
    gen_flat = ReportGenerator(flat_metrics)
    print(gen_flat.generate_main_table(format='text'))

    # Test with CV stats (structured)
    cv_stats_structured = {
        'main_metrics': {
            'NDCG@5':  {'mean': 0.0612, 'std': 0.0023},
            'NDCG@20': {'mean': 0.1456, 'std': 0.0041},
            'Precision@5': {'mean': 0.0523, 'std': 0.0015}
        },
        'additional_metrics': {
            'NDCG@10': {'mean': 0.0967, 'std': 0.0030},
        }
    }
    generator_cv = ReportGenerator({}, cv_stats_structured)
    print("\n=== CV text main table ===")
    print(generator_cv.generate_main_table(format='text'))

    print("\nReport generator module loaded successfully!")
