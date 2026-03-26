"""
IO utilities for caching and loading data
"""

import pickle
import torch
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def save_pickle(data: Any, filepath: str):
    """Save data to pickle file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved pickle to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load data from pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Loaded pickle from {filepath}")
    return data


def save_torch(data: Any, filepath: str):
    """Save data to torch file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, filepath)
    logger.info(f"Saved torch file to {filepath}")


def load_torch(filepath: str) -> Any:
    """Load data from torch file"""
    data = torch.load(filepath)
    logger.info(f"Loaded torch file from {filepath}")
    return data


def cache_exists(filepath: str) -> bool:
    """Check if cache file exists"""
    return Path(filepath).exists()


def get_cache_path(config: Dict, filename: str) -> str:
    """Get full path for cache file"""
    cache_dir = Path(config['dataset']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / filename)


def get_processed_path(config: Dict, filename: str) -> str:
    """Get full path for processed file"""
    processed_dir = Path(config['dataset']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    return str(processed_dir / filename)


def get_checkpoint_path(config: Dict, filename: str) -> str:
    """Get full path for checkpoint file"""
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / filename)