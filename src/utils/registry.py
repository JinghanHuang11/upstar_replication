"""
Model and component registry for easy configuration
"""

from typing import Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class Registry:
    """Registry for models, datasets, and other components"""

    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, key: str):
        """Decorator to register a component"""
        def decorator(component):
            self._registry[key] = component
            logger.info(f"Registered {key} in {self.name}")
            return component
        return decorator

    def get(self, key: str) -> Any:
        """Get a registered component"""
        if key not in self._registry:
            raise KeyError(f"{key} not found in {self.name} registry")
        return self._registry[key]

    def list_keys(self) -> list:
        """List all registered keys"""
        return list(self._registry.keys())

    def contains(self, key: str) -> bool:
        """Check if key is registered"""
        return key in self._registry


# Create registries for different component types
MODEL_REGISTRY = Registry('models')
DATASET_REGISTRY = Registry('datasets')
LOSS_REGISTRY = Registry('losses')
METRIC_REGISTRY = Registry('metrics')


def register_model(name: str):
    """Decorator to register a model"""
    return MODEL_REGISTRY.register(name)


def register_dataset(name: str):
    """Decorator to register a dataset"""
    return DATASET_REGISTRY.register(name)


def register_loss(name: str):
    """Decorator to register a loss function"""
    return LOSS_REGISTRY.register(name)


def register_metric(name: str):
    """Decorator to register a metric"""
    return METRIC_REGISTRY.register(name)