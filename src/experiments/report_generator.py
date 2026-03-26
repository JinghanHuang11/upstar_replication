"""
Experiment Report Generator

Generates comprehensive reports in Markdown and HTML formats
"""

import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ExperimentReport:
    """Generate experiment reports in multiple formats"""

    def __init__(self, results: Dict[int, Dict[str, Any]],
                 config: Dict[str, Any],
                 output_dir: Path):
        """
        Initialize report generator

        Args:
            results: Results from all phases
            config: Experiment configuration
            output_dir: Output directory for reports
        """
        self.results = results
        self.config = config
        self.output_dir = Path(output_dir)
        self.report_dir = self.output_dir / "report"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self):
        """Generate all report formats"""
        logger.info("Generating experiment reports...")

        self.save_markdown()
        self.save_html()
        self.save_json()

        logger.info(f"Reports saved to {self.report_dir}")

    def save_markdown(self):
        """Generate Markdown report"""
        report_path = self.report_dir / "report.md"

        content = self._generate_markdown_content()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Markdown report saved to {report_path}")

    def save_html(self):
        """Generate HTML report"""
        report_path = self.report_dir / "report.html"

        content = self._generate_html_content()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"HTML report saved to {report_path}")

    def save_json(self):
        """Generate JSON report"""
        report_path = self.report_dir / "report.json"

        report_data = {
            'experiment_name': self.config.get('name', 'experiment'),
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': self.results
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON report saved to {report_path}")

    def _generate_markdown_content(self) -> str:
        """Generate Markdown content"""
        lines = []

        # Title
        lines.append("# UPSTAR 实验报告\n")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**实验名称**: {self.config.get('name', 'experiment')}\n")

        # Table of Contents
        lines.append("## 目录\n")
        lines.append("- [实验配置](#实验配置)\n")
        lines.append("- [阶段结果](#阶段结果)\n")
        lines.append("- [性能对比](#性能对比)\n")
        lines.append("- [结论](#结论)\n")

        # Configuration
        lines.append("\n## 实验配置\n")
        lines.append(self._format_config())

        # Phase Results
        lines.append("\n## 阶段结果\n")
        lines.append(self._format_phase_results())

        # Performance Comparison
        lines.append("\n## 性能对比\n")
        lines.append(self._format_comparison())

        # Conclusion
        lines.append("\n## 结论\n")
        lines.append(self._generate_conclusion())

        return "\n".join(lines)

    def _format_config(self) -> str:
        """Format configuration section"""
        lines = []

        # Dataset info
        dataset = self.config.get('dataset', {})
        lines.append("### 数据集\n")
        lines.append(f"- **名称**: {dataset.get('name', 'N/A')}\n")
        lines.append(f"- **原始目录**: {dataset.get('raw_dir', 'N/A')}\n")
        lines.append(f"- **处理目录**: {dataset.get('processed_dir', 'N/A')}\n")

        # Add dataset statistics if available
        if 0 in self.results:
            results_0 = self.results[0]
            lines.append("\n**数据集统计**:\n")
            lines.append(f"- 用户数: {results_0.get('num_users', 'N/A'):,}\n")
            lines.append(f"- 物品数: {results_0.get('num_items', 'N/A'):,}\n")
            lines.append(f"- 交互数: {results_0.get('num_interactions', 'N/A'):,}\n")
            lines.append(f"- 训练序列: {results_0.get('train_sequences', 'N/A'):,}\n")
            lines.append(f"- 验证序列: {results_0.get('val_sequences', 'N/A'):,}\n")
            lines.append(f"- 测试序列: {results_0.get('test_sequences', 'N/A'):,}\n")

        return "\n".join(lines)

    def _format_phase_results(self) -> str:
        """Format phase results section"""
        lines = []

        phase_names = {
            0: "Phase 0: 数据预处理",
            1: "Phase 1: Baseline训练",
            2: "Phase 2: Item表示学习",
            3: "Phase 3: STB计算",
            4: "Phase 4: UPSTAR训练",
            5: "Phase 5: 评估与对比"
        }

        for phase_num in sorted(self.results.keys()):
            if phase_num not in self.results:
                continue

            lines.append(f"\n### {phase_names.get(phase_num, f'Phase {phase_num}')}\n")

            results = self.results[phase_num]

            # Format results based on phase
            if phase_num == 0:  # Preprocessing
                lines.append(f"- ✅ 数据预处理完成\n")
                lines.append(f"- 用户数: {results.get('num_users', 0):,}\n")
                lines.append(f"- 物品数: {results.get('num_items', 0):,}\n")

            elif phase_num == 1:  # Baseline
                lines.append(f"- ✅ Baseline训练完成\n")
                lines.append(f"- **NDCG@10**: {results.get('ndcg@10', 0):.4f}\n")
                lines.append(f"- **HR@10**: {results.get('hr@10', 0):.4f}\n")
                lines.append(f"- **MRR@10**: {results.get('mrr@10', 0):.4f}\n")
                lines.append(f"- 最佳epoch: {results.get('best_epoch', 0)}\n")

            elif phase_num == 2:  # Item representation
                lines.append(f"- ✅ Item表示学习完成\n")
                lines.append(f"- Embedding维度: {results.get('embedding_dim', 0)}\n")
                lines.append(f"- 物品数: {results.get('num_items', 0):,}\n")

            elif phase_num == 3:  # STB
                lines.append(f"- ✅ STB计算完成\n")
                lines.append(f"- Stable物品: {results.get('stable_count', 0):,}\n")
                lines.append(f"- Exploratory物品: {results.get('exploratory_count', 0):,}\n")
                lines.append(f"- Other物品: {results.get('other_count', 0):,}\n")

            elif phase_num == 4:  # UPSTAR
                lines.append(f"- ✅ UPSTAR训练完成\n")
                lines.append(f"- **NDCG@10**: {results.get('ndcg@10', 0):.4f}\n")
                lines.append(f"- **HR@10**: {results.get('hr@10', 0):.4f}\n")
                lines.append(f"- **MRR@10**: {results.get('mrr_10', 0):.4f}\n")

        return "\n".join(lines)

    def _format_comparison(self) -> str:
        """Format comparison section"""
        lines = []

        if 5 not in self.results:
            lines.append("*Phase 5未完成，无法生成对比*\n")
            return "\n".join(lines)

        comparison = self.results[5].get('comparison', {})

        if not comparison:
            lines.append("*无对比数据*\n")
            return "\n".join(lines)

        lines.append("### Baseline vs UPSTAR\n")
        lines.append("\n")
        lines.append("| 指标 | Baseline | UPSTAR | 提升 |\n")
        lines.append("|------|----------|--------|------|\n")

        for metric_name, values in comparison.items():
            metric_display = metric_name.upper().replace('@', '@')
            baseline = values.get('baseline', 0.0)
            upstar = values.get('upstar', 0.0)
            improvement = values.get('improvement_pct', 0.0)

            improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            lines.append(f"| {metric_display} | {baseline:.4f} | {upstar:.4f} | {improvement_str} |\n")

        return "\n".join(lines)

    def _generate_conclusion(self) -> str:
        """Generate conclusion section"""
        lines = []

        if 5 not in self.results:
            lines.append("*实验未完成，无法生成结论*\n")
            return "\n".join(lines)

        comparison = self.results[5].get('comparison', {})

        if not comparison:
            lines.append("*无对比数据*\n")
            return "\n".join(lines)

        # Calculate average improvement
        improvements = [v.get('improvement_pct', 0) for v in comparison.values()]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        lines.append(f"**总体性能**: UPSTAR相对于Baseline")
        lines.append(f"平均提升 **{avg_improvement:.2f}%**\n")

        lines.append("\n**主要发现**:\n")

        # Analyze each metric
        for metric_name, values in comparison.items():
            improvement = values.get('improvement_pct', 0)
            if improvement > 0:
                lines.append(f"- ✅ {metric_name.upper()}: 提升 {improvement:.2f}%\n")
            else:
                lines.append(f"- ⚠️  {metric_name.upper()}: 下降 {abs(improvement):.2f}%\n")

        return "\n".join(lines)

    def _generate_html_content(self) -> str:
        """Generate HTML content"""
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UPSTAR 实验报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-positive {{
            color: #27ae60;
            font-weight: bold;
        }}
        .metric-negative {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
        }}
        ul {{
            line-height: 1.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>UPSTAR 实验报告</h1>
        <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>实验名称: {self.config.get('name', 'experiment')}</p>

        <h2>目录</h2>
        <ul>
            <li><a href="#config">实验配置</a></li>
            <li><a href="#results">阶段结果</a></li>
            <li><a href="#comparison">性能对比</a></li>
            <li><a href="#conclusion">结论</a></li>
        </ul>

        <h2 id="config">实验配置</h2>
        {self._format_config_html()}

        <h2 id="results">阶段结果</h2>
        {self._format_results_html()}

        <h2 id="comparison">性能对比</h2>
        {self._format_comparison_html()}

        <h2 id="conclusion">结论</h2>
        {self._format_conclusion_html()}
    </div>
</body>
</html>"""
        return html

    def _format_config_html(self) -> str:
        """Format configuration as HTML"""
        dataset = self.config.get('dataset', {})
        html = f"""
        <h3>数据集</h3>
        <ul>
            <li><strong>名称</strong>: {dataset.get('name', 'N/A')}</li>
            <li><strong>原始目录</strong>: {dataset.get('raw_dir', 'N/A')}</li>
            <li><strong>处理目录</strong>: {dataset.get('processed_dir', 'N/A')}</li>
        </ul>
        """
        return html

    def _format_results_html(self) -> str:
        """Format results as HTML"""
        html = ""

        for phase_num in sorted(self.results.keys()):
            results = self.results[phase_num]
            html += f"<h3>Phase {phase_num}</h3>"
            html += "<ul>"

            for key, value in results.items():
                if isinstance(value, float):
                    html += f"<li>{key}: {value:.4f}</li>"
                elif isinstance(value, int):
                    html += f"<li>{key}: {value:,}</li>"
                else:
                    html += f"<li>{key}: {value}</li>"

            html += "</ul>"

        return html

    def _format_comparison_html(self) -> str:
        """Format comparison as HTML"""
        if 5 not in self.results:
            return "<p>Phase 5未完成，无法生成对比</p>"

        comparison = self.results[5].get('comparison', {})

        if not comparison:
            return "<p>无对比数据</p>"

        html = "<table>"
        html += "<tr><th>指标</th><th>Baseline</th><th>UPSTAR</th><th>提升</th></tr>"

        for metric_name, values in comparison.items():
            metric_display = metric_name.upper()
            baseline = values.get('baseline', 0.0)
            upstar = values.get('upstar', 0.0)
            improvement = values.get('improvement_pct', 0.0)

            improvement_class = "metric-positive" if improvement > 0 else "metric-negative"
            improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"

            html += f"<tr>"
            html += f"<td>{metric_display}</td>"
            html += f"<td>{baseline:.4f}</td>"
            html += f"<td>{upstar:.4f}</td>"
            html += f"<td class='{improvement_class}'>{improvement_str}</td>"
            html += f"</tr>"

        html += "</table>"
        return html

    def _format_conclusion_html(self) -> str:
        """Format conclusion as HTML"""
        if 5 not in self.results:
            return "<p>实验未完成，无法生成结论</p>"

        comparison = self.results[5].get('comparison', {})

        if not comparison:
            return "<p>无对比数据</p>"

        improvements = [v.get('improvement_pct', 0) for v in comparison.values()]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        html = f"<p><strong>总体性能</strong>: UPSTAR相对于Baseline平均提升 <span class='metric-positive'>{avg_improvement:.2f}%</span></p>"
        html += "<h3>主要发现</h3><ul>"

        for metric_name, values in comparison.items():
            improvement = values.get('improvement_pct', 0)
            if improvement > 0:
                html += f"<li class='metric-positive'>{metric_name.upper()}: 提升 {improvement:.2f}%</li>"
            else:
                html += f"<li class='metric-negative'>{metric_name.upper()}: 下降 {abs(improvement):.2f}%</li>"

        html += "</ul>"
        return html