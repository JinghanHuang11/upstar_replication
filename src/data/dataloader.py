"""
PyTorch DataLoader for sequential recommendation
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    Collate function for DataLoader

    Args:
        batch: list of dicts from SequentialDataset

    Returns:
        batched dict of tensors
    """
    # Stack all tensors
    batch_dict = {}

    keys = batch[0].keys()
    for key in keys:
        tensors = [sample[key] for sample in batch]
        batch_dict[key] = torch.stack(tensors, dim=0)

    return batch_dict


def get_dataloader(
    sequences: Dict,
    max_seq_length: int,
    batch_size: int,
    split: str = 'train',
    num_workers: int = 0,
    num_items: int = None,
    use_negative_sampling: bool = False,
    num_neg: int = 100
) -> DataLoader:
    """
    Get DataLoader for sequential recommendation

    Args:
        sequences: user sequences
        max_seq_length: maximum sequence length
        batch_size: batch size
        split: 'train', 'val', or 'test'
        num_workers: number of workers for data loading
        num_items: total number of items (required for negative sampling)
        use_negative_sampling: whether to use negative sampling
        num_neg: number of negative samples

    Returns:
        DataLoader
    """
    from src.data.dataset import SequentialDataset, NegativeSamplingDataset

    # Create dataset
    if use_negative_sampling and split == 'train':
        if num_items is None:
            raise ValueError("num_items required for negative sampling")

        dataset = NegativeSamplingDataset(
            sequences=sequences,
            num_items=num_items,
            max_seq_length=max_seq_length,
            num_neg=num_neg
        )
    else:
        dataset = SequentialDataset(
            sequences=sequences,
            max_seq_length=max_seq_length,
            split=split
        )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    logger.info(f"Created {split} dataloader: {len(dataloader)} batches")

    return dataloader


if __name__ == '__main__':
    import pickle

    print("Testing DataLoader...")

    # Create dummy data
    sequences = {
        i: {'items': list(range(i, i+10)), 'target': i+10}
        for i in range(100)
    }

    dataloader = get_dataloader(
        sequences=sequences,
        max_seq_length=20,
        batch_size=16,
        split='train',
        num_workers=0
    )

    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Items shape: {batch['items'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Seq lengths: {batch['seq_length'][:5]}")
        break

    print("\nDataLoader test passed!")