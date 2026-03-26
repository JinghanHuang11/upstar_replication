"""
Base class for experiment phases

Each phase is self-contained with:
- Dependency checking
- Execution logic
- Result loading/saving
- Status tracking
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import time
import json

logger = logging.getLogger(__name__)


class ExperimentPhase(ABC):
    """Base class for an experiment phase"""

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.phase_dir = self.output_dir / self.phase_name
        self.checkpoint_dir = self.phase_dir / "checkpoints"
        self.log_dir = self.phase_dir / "logs"
        self.results_dir = self.phase_dir / "results"

        # Create directories
        self.phase_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def phase_name(self) -> str:
        """Name of the phase (e.g., 'phase0_preprocess')"""
        pass

    @property
    @abstractmethod
    def phase_number(self) -> int:
        """Phase number (0-5)"""
        pass

    @abstractmethod
    def check_dependencies(self) -> bool:
        """Check if all dependencies are satisfied"""
        pass

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the phase and return results"""
        pass

    def is_completed(self) -> bool:
        """Check if this phase has been completed"""
        status_file = self.phase_dir / "status.json"
        if not status_file.exists():
            return False

        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            return status.get('completed', False)
        except:
            return False

    def load_results(self) -> Optional[Dict[str, Any]]:
        """Load results from completed phase"""
        if not self.is_completed():
            return None

        results_file = self.results_dir / "results.json"
        if not results_file.exists():
            return None

        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except:
            return None

    def save_status(self, completed: bool, message: str = ""):
        """Save phase status"""
        status = {
            'phase': self.phase_name,
            'phase_number': self.phase_number,
            'completed': completed,
            'message': message,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        status_file = self.phase_dir / "status.json"
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

    def save_results(self, results: Dict[str, Any]):
        """Save phase results"""
        results_file = self.results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def get_duration(self) -> Optional[float]:
        """Get phase execution duration in seconds"""
        status_file = self.phase_dir / "status.json"
        if not status_file.exists():
            return None

        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            return status.get('duration_seconds')
        except:
            return None

    def execute(self) -> Dict[str, Any]:
        """Execute the phase with timing and error handling"""
        logger.info(f"{'='*80}")
        logger.info(f"Starting {self.phase_name}")
        logger.info(f"{'='*80}")

        # Check dependencies
        if not self.check_dependencies():
            error_msg = f"Dependencies not satisfied for {self.phase_name}"
            logger.error(error_msg)
            self.save_status(False, error_msg)
            raise RuntimeError(error_msg)

        # Check if already completed
        if self.is_completed():
            logger.info(f"{self.phase_name} already completed, loading results...")
            results = self.load_results()
            if results:
                duration = self.get_duration()
                logger.info(f"  Duration: {duration:.2f}s" if duration else "")
                return results

        # Run the phase
        start_time = time.time()
        try:
            results = self.run()
            duration = time.time() - start_time

            # Save results and status
            self.save_results(results)
            self.save_status(True, f"Completed in {duration:.2f}s")

            # Update duration in status
            status_file = self.phase_dir / "status.json"
            with open(status_file, 'r') as f:
                status = json.load(f)
            status['duration_seconds'] = duration
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)

            logger.info(f"{self.phase_name} completed in {duration:.2f}s")
            return results

        except Exception as e:
            logger.error(f"{self.phase_name} failed: {str(e)}")
            self.save_status(False, f"Failed: {str(e)}")
            raise