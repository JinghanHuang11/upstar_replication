"""
Convenience function to load item embeddings
"""

import torch
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_item_embeddings(
    config: Optional[dict] = None,
    checkpoint_dir: Optional[str] = None
) -> torch.Tensor:
    """
    Load learned item embeddings (GRAPH-ENHANCED)

    This loads the FINAL item embeddings AFTER GNN message passing,
    which include graph structural information.

    Args:
        config: configuration dict (optional)
        checkpoint_dir: path to checkpoint directory (optional)

    Returns:
        item_embeddings: [num_items, embed_dim] - GRAPH-ENHANCED embeddings

    Example usage:
        >>> from src.models.load_item_embeddings import get_item_embeddings
        >>> embeddings = get_item_embeddings(checkpoint_dir='outputs/phase2_item_repr/checkpoints/')
        >>> print(embeddings.shape)  # [num_items, 128]

    NOTE:
        These embeddings are the output of ItemGNN.forward(edge_index),
        NOT the raw embedding table weights.
        They include graph structural information through GNN propagation.
    """
    if checkpoint_dir is None and config is not None:
        # Try common paths
        possible_dirs = [
            config['logging'].get('checkpoint_dir', ''),
            'outputs/phase2_item_repr/checkpoints/',
            'outputs/phase3_stb/checkpoints/'  # STB config might point here
        ]
        for dir_path in possible_dirs:
            if dir_path and (Path(dir_path) / 'item_embeddings.pt').exists():
                checkpoint_dir = dir_path
                break

    if checkpoint_dir is None:
        checkpoint_dir = 'outputs/phase2_item_repr/checkpoints/'

    embedding_path = Path(checkpoint_dir) / 'item_embeddings.pt'

    if not embedding_path.exists():
        # Try absolute path
        if not embedding_path.is_absolute():
            embedding_path = Path(__file__).parent.parent.parent / checkpoint_dir / 'item_embeddings.pt'

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Item embeddings not found at {embedding_path}\n"
            f"Please run item representation learning first:\n"
            f"  bash scripts/run_item_repr.sh"
        )

    embeddings = torch.load(embedding_path)

    logger.info(f"Loaded item embeddings from {embedding_path}")
    logger.info(f"  Shape: {embeddings.shape}")

    return embeddings


if __name__ == '__main__':
    # Test loading
    print("Testing item embedding loading...")

    try:
        embeddings = get_item_embeddings()
        print(f"Successfully loaded embeddings: {embeddings.shape}")
    except FileNotFoundError as e:
        print(e)
        print("\nPlease train item representations first:")
        print("  bash scripts/run_item_repr.sh")