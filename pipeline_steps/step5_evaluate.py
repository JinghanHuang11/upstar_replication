#!/usr/bin/env python
"""
Step 5: Evaluation and Comparison

Compares baseline and UPSTAR results, generates comparison report.

Output:
    - outputs/comparison/comparison.txt
    - outputs/comparison/improvement_analysis.txt

Usage:
    python -m pipeline_steps.step5_evaluate
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_steps.base_step import BaseStep


class Step5Evaluate(BaseStep):
    """Step 5: Evaluation and comparison"""

    def __init__(self):
        output_dir = "outputs/comparison"
        super().__init__(None, output_dir)
        self.baseline_dir = Path("outputs/phase1_baseline/results")
        self.upstar_dir = Path("outputs/phase4_upstar/results")

    def check_prerequisites(self) -> bool:
        """Check if baseline and UPSTAR results exist"""
        self.logger.info("Checking prerequisites...")

        # Check baseline results
        baseline_results = self.baseline_dir / 'cv_results.json'
        if not baseline_results.exists():
            self.logger.error(f"  ✗ Baseline results not found: {baseline_results}")
            self.logger.error("    Run: python -m pipeline_steps.step1_baseline")
            return False
        self.logger.info(f"  ✓ Baseline results: {baseline_results}")

        # Check UPSTAR results
        upstar_results = self.upstar_dir / 'cv_results.json'
        if not upstar_results.exists():
            self.logger.error(f"  ✗ UPSTAR results not found: {upstar_results}")
            self.logger.error("    Run: python -m pipeline_steps.step4_upstar")
            return False
        self.logger.info(f"  ✓ UPSTAR results: {upstar_results}")

        return True

    def check_output_exists(self) -> bool:
        """Check if comparison already exists"""
        if not self.output_dir:
            return False

        comparison_file = self.output_dir / 'comparison.txt'
        if comparison_file.exists():
            self.logger.info(f"  ✓ Output exists: {comparison_file}")
            return True
        else:
            self.logger.info(f"  → Output not found, will run comparison")
            return False

    def run(self) -> bool:
        """Run evaluation and comparison"""
        try:
            from src.evaluation.comparison import ModelComparator
            import json

            self.logger.info("Generating comparison...")

            # Create comparator
            comparator = ModelComparator(
                baseline_results_dir=str(self.baseline_dir),
                upstar_results_dir=str(self.upstar_dir),
                output_dir=str(self.output_dir)
            )

            # Save comparison
            comparator.save_comparison()

            # Print executive summary
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("COMPARISON RESULTS")
            self.logger.info("=" * 80)
            self.logger.info("")
            self.logger.info(comparator.generate_executive_summary())

            # Show detailed comparison
            comparison_file = self.output_dir / 'comparison.txt'
            if comparison_file.exists():
                self.logger.info("")
                self.logger.info("✓ Comparison complete!")
                self.logger.info(f"Results saved to: {comparison_file}")
                self.logger.info("")
                self.logger.info("View full comparison:")
                self.logger.info(f"  cat {comparison_file}")
                self.logger.info("")
                self.logger.info("View improvement analysis:")
                self.logger.info(f"  cat {self.output_dir / 'improvement_analysis.txt'}")

                # Load and display summary
                with open(comparison_file, 'r') as f:
                    content = f.read()
                    # Extract just the summary table
                    lines = content.split('\n')
                    in_table = False
                    for line in lines:
                        if 'Metric' in line and 'Baseline' in line:
                            in_table = True
                        if in_table:
                            self.logger.info(line)
                            if line.strip() == '':
                                in_table = False

                return True
            else:
                self.logger.error("Comparison completed but output file not found")
                return False

        except Exception as e:
            self.logger.error(f"Comparison failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Step 5: Evaluation and Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comparison
  python -m pipeline_steps.step5_evaluate
        """
    )

    args = parser.parse_args()

    # Create and run step
    step = Step5Evaluate()
    success = step.execute()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
