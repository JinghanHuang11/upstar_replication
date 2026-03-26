"""
Experiment Manager - Orchestrates complete UPSTAR experiment pipeline

Manages all phases from data preprocessing to final evaluation
"""

import logging
import yaml
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.experiments.phases import (
    Phase0_Preprocess,
    Phase1_Baseline,
    Phase2_ItemRepr,
    Phase3_STB,
    Phase4_UPSTAR,
    Phase5_Evaluate
)

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages complete UPSTAR experiment pipeline"""

    def __init__(self, config_path: str, output_base_dir: str = "outputs/experiments"):
        """
        Initialize experiment manager

        Args:
            config_path: Path to experiment configuration YAML
            output_base_dir: Base directory for experiment outputs
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Create experiment output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{self.config.get('name', 'experiment')}_{timestamp}"
        self.output_dir = Path(output_base_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize phases
        self.phases = [
            Phase0_Preprocess(self.config, self.output_dir),
            Phase1_Baseline(self.config, self.output_dir),
            Phase2_ItemRepr(self.config, self.output_dir),
            Phase3_STB(self.config, self.output_dir),
            Phase4_UPSTAR(self.config, self.output_dir),
            Phase5_Evaluate(self.config, self.output_dir),
        ]

        self.results = {}

        logger.info(f"Experiment manager initialized")
        logger.info(f"  Config: {self.config_path}")
        logger.info(f"  Output directory: {self.output_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def run(self,
            start_phase: int = 0,
            stop_phase: Optional[int] = None,
            skip_completed: bool = True,
            resume: bool = False):
        """
        Run experiment pipeline

        Args:
            start_phase: Phase to start from (0-5)
            stop_phase: Phase to stop at (None = run all)
            skip_completed: Skip already completed phases
            resume: Resume from last incomplete phase
        """
        logger.info("="*80)
        logger.info("STARTING EXPERIMENT PIPELINE")
        logger.info("="*80)

        if resume:
            start_phase = self._find_resume_phase()
            logger.info(f"Resuming from phase {start_phase}")

        total_start_time = time.time()

        for i, phase in enumerate(self.phases):
            if i < start_phase:
                logger.info(f"Skipping phase {i} ({phase.phase_name})")
                continue

            if stop_phase is not None and i > stop_phase:
                logger.info(f"Stopping at phase {stop_phase}")
                break

            # Check if already completed
            if skip_completed and phase.is_completed():
                logger.info(f"Phase {i} ({phase.phase_name}) already completed, skipping...")
                self.results[i] = phase.load_results()
                continue

            # Execute phase
            try:
                self.results[i] = phase.execute()
            except Exception as e:
                logger.error(f"Phase {i} failed: {str(e)}")
                logger.info("You can resume from this phase later using --resume")
                raise

        total_duration = time.time() - total_start_time

        logger.info("="*80)
        logger.info("EXPERIMENT PIPELINE COMPLETE")
        logger.info(f"Total duration: {total_duration/3600:.2f} hours")
        logger.info("="*80)

        # Save summary
        self._save_summary(total_duration)

    def _find_resume_phase(self) -> int:
        """Find the first incomplete phase"""
        for i, phase in enumerate(self.phases):
            if not phase.is_completed():
                return i
        return len(self.phases) - 1

    def _save_summary(self, total_duration: float):
        """Save experiment summary"""
        summary = {
            'experiment_name': self.config.get('name', 'experiment'),
            'config_path': str(self.config_path),
            'output_dir': str(self.output_dir),
            'total_duration_hours': total_duration / 3600,
            'timestamp': datetime.now().isoformat(),
            'phases': {}
        }

        for i, phase in enumerate(self.phases):
            phase_info = {
                'name': phase.phase_name,
                'completed': phase.is_completed(),
                'duration_seconds': phase.get_duration()
            }

            if i in self.results:
                phase_info['results'] = self.results[i]

            summary['phases'][i] = phase_info

        # Save summary
        summary_file = self.output_dir / "experiment_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Experiment summary saved to {summary_file}")

    def get_results(self) -> Dict[int, Dict[str, Any]]:
        """Get results from all completed phases"""
        return self.results

    def get_comparison(self) -> Optional[Dict[str, Any]]:
        """Get baseline vs UPSTAR comparison"""
        if 5 in self.results:
            return self.results[5].get('comparison')
        return None

    def print_summary(self):
        """Print experiment summary to console"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80 + "\n")

        print(f"Output directory: {self.output_dir}\n")

        print("Phase Status:")
        print("-" * 80)
        for i, phase in enumerate(self.phases):
            status = "✅ Complete" if phase.is_completed() else "⏳ Pending"
            duration = phase.get_duration()
            duration_str = f" ({duration:.1f}s)" if duration else ""

            print(f"  Phase {i}: {phase.phase_name:<30} {status:<15} {duration_str}")

        print("\n" + "-" * 80)

        # Print comparison if available
        comparison = self.get_comparison()
        if comparison:
            print("\nPerformance Comparison:")
            print("-" * 80)
            print(f"{'Metric':<15} {'Baseline':<12} {'UPSTAR':<12} {'Improvement':<12}")
            print("-" * 80)

            for metric_name, values in comparison.items():
                metric_display = metric_name.upper()
                baseline = values['baseline']
                upstar = values['upstar']
                improvement = values['improvement_pct']

                print(f"{metric_display:<15} {baseline:<12.4f} {upstar:<12.4f} {improvement:+.2f}%")

        print("\n" + "="*80)

    def estimate_remaining_time(self, current_phase: int) -> Optional[float]:
        """
        Estimate remaining time based on completed phases

        Returns:
            Estimated remaining time in seconds
        """
        # Get durations of completed phases
        durations = []
        for i, phase in enumerate(self.phases):
            if i < current_phase and phase.is_completed():
                duration = phase.get_duration()
                if duration:
                    durations.append(duration)

        if not durations:
            return None

        # Simple estimate: average duration * remaining phases
        avg_duration = sum(durations) / len(durations)
        remaining_phases = len(self.phases) - current_phase

        return avg_duration * remaining_phases