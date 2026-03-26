"""
UPSTAR Dataset: Split user sequences into three paths based on motivation

Constructs:
- Stable preference subsequence
- Exploratory intent subsequence
- Entire original sequence
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UPSTARDataset(Dataset):
    """
    Dataset for UPSTAR three-path model

    For each user sequence, create three subsequences:
    1. Stable preference subsequence (items with stable motivation)
    2. Exploratory intent subsequence (items with exploratory motivation)
    3. Entire original sequence (all items)
    """

    def __init__(
        self,
        sequences: Dict[int, Dict],
        motivation_labels: np.ndarray,
        max_seq_length: int = 50,
        split: str = 'train'
    ):
        """
        Args:
            sequences: {user_idx: {'items': [(item, timestamp), ...], 'target': int}}
            motivation_labels: [num_items] - 0=exploratory, 1=stable, 2=uncategorized
            max_seq_length: maximum sequence length
            split: 'train', 'val', or 'test'
        """
        self.max_seq_length = max_seq_length
        self.split = split
        self.motivation_labels = motivation_labels

        # Build samples
        self.samples = []
        for user_idx, data in sequences.items():
            # Support both old format [item1, item2, ...] and new format [(item, ts), ...]
            raw_items = data['items']
            if raw_items and isinstance(raw_items[0], tuple):
                # New format: list of (item, timestamp) tuples
                items = [item for item, _ in raw_items]
            else:
                # Old format: plain list of items
                items = raw_items

            target = data.get('target')

            if len(items) < 1:
                continue

            if target is not None or split == 'test':
                self.samples.append({
                    'user_idx': user_idx,
                    'items': items,
                    'target': target
                })

        logger.info(f"Created {split} dataset with {len(self)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def split_sequence_by_motivation(
        self,
        items: List[int]
    ) -> Dict[str, List[int]]:
        """
        Split sequence into three subsequences based on motivation

        Args:
            items: list of item indices

        Returns:
            {
                'stable': [...],        # items with stable motivation
                'exploratory': [...],   # items with exploratory motivation
                'entire': [...]         # all items
            }
        """
        stable_items = []
        exploratory_items = []
        entire_items = []

        for item in items:
            motivation = self.motivation_labels[item]

            # Add to entire sequence
            entire_items.append(item)

            # Add to specific subsequence
            if motivation == 1:  # Stable
                stable_items.append(item)
            elif motivation == 0:  # Exploratory
                exploratory_items.append(item)
            # motivation == 2 (uncategorized): only in entire sequence

        return {
            'stable': stable_items,
            'exploratory': exploratory_items,
            'entire': entire_items
        }

    def pad_sequence(self, items: List[int]) -> Dict[str, torch.Tensor]:
        """
        Pad sequence to max_seq_length

        Returns:
            {
                'items': [max_seq_length],
                'mask': [max_seq_length],
                'seq_length': int
            }
        """
        seq_length = len(items)

        # Ensure minimum length of 1 (for empty subsequences)
        # This prevents pack_padded_sequence errors
        if seq_length == 0:
            items = [0]  # Use padding token
            seq_length = 1

        # Truncate if too long
        if seq_length > self.max_seq_length:
            items = items[-self.max_seq_length:]
            seq_length = self.max_seq_length

        # Pad
        items_tensor = torch.zeros(self.max_seq_length, dtype=torch.long)
        items_tensor[:seq_length] = torch.tensor(items, dtype=torch.long)

        # Mask (1 for valid, 0 for padding)
        mask = torch.zeros(self.max_seq_length, dtype=torch.bool)
        mask[:seq_length] = True

        return {
            'items': items_tensor,
            'mask': mask,
            'seq_length': seq_length
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        user_idx = sample['user_idx']
        items = sample['items']
        target = sample['target']

        # Split by motivation
        subsequences = self.split_sequence_by_motivation(items)

        # Pad each subsequence
        stable_data = self.pad_sequence(subsequences['stable'])
        exploratory_data = self.pad_sequence(subsequences['exploratory'])
        entire_data = self.pad_sequence(subsequences['entire'])

        output = {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'seq_stable': stable_data['items'],
            'mask_stable': stable_data['mask'],
            'len_stable': torch.tensor(stable_data['seq_length'], dtype=torch.long),
            'seq_exploratory': exploratory_data['items'],
            'mask_exploratory': exploratory_data['mask'],
            'len_exploratory': torch.tensor(exploratory_data['seq_length'], dtype=torch.long),
            'seq_entire': entire_data['items'],
            'mask_entire': entire_data['mask'],
            'len_entire': torch.tensor(entire_data['seq_length'], dtype=torch.long),
        }

        if target is not None:
            output['target'] = torch.tensor(target, dtype=torch.long)
            # Add motivation label for the target item
            output['motivation_labels'] = torch.tensor(
                self.motivation_labels[target], dtype=torch.long
            )

        return output


def upstar_collate_fn(batch):
    """
    Collate function for UPSTAR dataset

    Handles empty subsequences properly
    """
    batch_size = len(batch)

    # Stack all tensors
    output = {}
    keys = batch[0].keys()

    for key in keys:
        if key == 'target':
            # Handle optional target
            if batch[0][key] is not None:
                output[key] = torch.stack([item[key] for item in batch])
        else:
            output[key] = torch.stack([item[key] for item in batch])

    return output


def get_upstar_dataloader(
    sequences: Dict,
    motivation_labels: np.ndarray,
    max_seq_length: int,
    batch_size: int,
    split: str = 'train',
    num_workers: int = 0
):
    """
    Get DataLoader for UPSTAR model

    Args:
        sequences: user sequences
        motivation_labels: [num_items] motivation labels
        max_seq_length: maximum sequence length
        batch_size: batch size
        split: 'train', 'val', or 'test'
        num_workers: number of workers

    Returns:
        DataLoader
    """
    dataset = UPSTARDataset(
        sequences=sequences,
        motivation_labels=motivation_labels,
        max_seq_length=max_seq_length,
        split=split
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=upstar_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    logger.info(f"Created {split} dataloader: {len(dataloader)} batches")

    return dataloader


if __name__ == '__main__':
    # Test dataset
    print("Testing UPSTAR Dataset...")

    # Create dummy data
    sequences = {
        0: {'items': [1, 2, 3, 4, 5], 'target': 6},
        1: {'items': [2, 3, 4], 'target': 5},
        2: {'items': [1, 2, 3, 4, 5, 6], 'target': 7}
    }

    # Create dummy labels
    motivation_labels = np.array([1, 0, 1, 0, 2, 1, 0])  # 1=stable, 0=exploratory, 2=uncategorized

    dataset = UPSTARDataset(
        sequences=sequences,
        motivation_labels=motivation_labels,
        max_seq_length=10,
        split='train'
    )

    print(f"Dataset size: {len(dataset)}")

    # Test sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"seq_stable shape: {sample['seq_stable'].shape}")
    print(f"seq_exploratory shape: {sample['seq_exploratory'].shape}")
    print(f"seq_entire shape: {sample['seq_entire'].shape}")
    print(f"len_stable: {sample['len_stable']}")
    print(f"len_exploratory: {sample['len_exploratory']}")
    print(f"len_entire: {sample['len_entire']}")
    print(f"target: {sample['target']}")

    # Test dataloader
    dataloader = get_upstar_dataloader(
        sequences=sequences,
        motivation_labels=motivation_labels,
        max_seq_length=10,
        batch_size=2,
        split='train',
        num_workers=0
    )

    for batch in dataloader:
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch size: {batch['seq_stable'].shape[0]}")
        print(f"Stable seqs: {batch['len_stable']}")
        print(f"Exploratory seqs: {batch['len_exploratory']}")
        print(f"Entire seqs: {batch['len_entire']}")
        break

    print("\nUPSTAR Dataset test passed!")
