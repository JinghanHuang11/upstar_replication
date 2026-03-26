"""
Logger configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Get a logger with console and optional file handler

    Args:
        name: logger name
        log_file: optional log file path
        level: logging level

    Returns:
        configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class TensorboardLogger:
    """Simple wrapper for TensorBoard logging"""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        Path(log_dir).parent.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value"""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram"""
        self.writer.add_histogram(tag, values, step)

    def log_graph(self, model, input_data):
        """Log model graph"""
        self.writer.add_graph(model, input_data)

    def close(self):
        """Close the writer"""
        self.writer.close()