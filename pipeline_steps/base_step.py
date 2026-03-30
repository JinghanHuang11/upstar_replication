"""
Base class for all pipeline steps.

Provides common functionality for:
- Logging
- Config validation
- Output directory management
- Prerequisite checking
- Error handling
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
import time


class BaseStep:
    """Base class for all pipeline steps"""

    def __init__(self, config_path: str = None, output_dir: str = None):
        """
        Initialize pipeline step.

        Args:
            config_path: Path to config file
            output_dir: Path to output directory
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir) if output_dir else None
        self.config = None

        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

        # Add console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(handler)

    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load config from file"""
        config_path = config_path or self.config_path
        if not config_path:
            raise ValueError("Config path not provided")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        return self.config

    def check_prerequisites(self) -> bool:
        """
        Check if prerequisites are satisfied.

        Returns:
            True if all prerequisites are satisfied
        """
        raise NotImplementedError("Subclasses must implement check_prerequisites")

    def check_output_exists(self) -> bool:
        """
        Check if output already exists.

        Returns:
            True if output exists
        """
        raise NotImplementedError("Subclasses must implement check_output_exists")

    def run(self) -> bool:
        """
        Execute the pipeline step.

        Returns:
            True if successful
        """
        raise NotImplementedError("Subclasses must implement run")

    def create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {self.output_dir}")

    def get_elapsed_time_str(self, start_time: float) -> str:
        """Get formatted elapsed time string"""
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def execute(self) -> bool:
        """
        Main execution method with logging and error handling.

        Returns:
            True if successful
        """
        start_time = time.time()

        self.logger.info("=" * 80)
        self.logger.info(f"Running: {self.__class__.__name__}")
        self.logger.info("=" * 80)

        try:
            # Check prerequisites
            self.logger.info("Checking prerequisites...")
            if not self.check_prerequisites():
                self.logger.error("Prerequisites not satisfied")
                return False

            # Create output directory
            self.create_output_dir()

            # Run the step
            self.logger.info("Starting execution...")
            success = self.run()

            if success:
                elapsed = self.get_elapsed_time_str(start_time)
                self.logger.info("=" * 80)
                self.logger.info(f"✓ Step completed successfully in {elapsed}")
                self.logger.info("=" * 80)
            else:
                self.logger.error("Step failed")

            return success

        except Exception as e:
            elapsed = self.get_elapsed_time_str(start_time)
            self.logger.error(f"✗ Step failed after {elapsed}: {e}")
            import traceback
            traceback.print_exc()
            return False
