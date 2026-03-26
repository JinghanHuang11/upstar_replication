"""
Utility functions
"""

import random
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any
import yaml
import json


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str):
    """Save configuration to YAML file"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_json(data: Any, save_path: str):
    """Save data to JSON file"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(load_path: str) -> Any:
    """Load data from JSON file"""
    with open(load_path, 'r') as f:
        data = json.load(f)
    return data


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get available device (GPU if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(path: str):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)