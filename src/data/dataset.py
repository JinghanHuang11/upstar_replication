"""
PyTorch Dataset for next-item recommendation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class SequentialDataset(Dataset):
    """
    Dataset for next-item recommendation

    For each user sequence, create next-item prediction samples:
    - Input: items[:-1]
    - Target: items[-1]
    """

    def __init__(
        self,
        sequences: Dict[int, Dict],
        max_seq_length: int = 50,
        split: str = 'train'
    ):
        """
        Args:
            sequences: {user_idx: {'items': [...], 'target': int}}
            max_seq_length: maximum sequence length
            split: 'train', 'val', or 'test'
        """
        self.max_seq_length = max_seq_length
        self.split = split

        # Build samples
        self.samples = []
        for user_idx, data in sequences.items():
            items = data['items']
            target = data.get('target')

            if len(items) < 1:
                continue

            # For training/val: we have explicit target
            # For test: we might not have target (will be handled during evaluation)
            if target is not None or split == 'test':
                self.samples.append({
                    'user_idx': user_idx,
                    'items': items,
                    'target': target
                })

        logger.info(f"Created {split} dataset with {len(self)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        user_idx = sample['user_idx']
        items = sample['items']
        target = sample['target']

        # **PAPER-ALIGNED FORMAT HANDLING**
        # Extract item indices if items are tuples (item_idx, timestamp)
        # This handles the new paper-aligned format: [(item, timestamp), ...]
        # Required for Phase 2/3 (session construction, item-time graph)
        # BACKWARD COMPATIBILITY: Also supports old format [item1, item2, ...]
        if len(items) > 0 and isinstance(items[0], (tuple, list)):
            # New format: [(item, timestamp), ...] → extract items only
            items = [item[0] for item in items]
        # Old format: [item1, item2, ...] → use as-is

        # Truncate sequence if too long
        if len(items) > self.max_seq_length:
            items = items[-self.max_seq_length:]

        # Convert to tensor
        seq_length = len(items)
        items_tensor = torch.zeros(self.max_seq_length, dtype=torch.long)
        items_tensor[:seq_length] = torch.tensor(items, dtype=torch.long)

        # Create mask for valid positions
        mask = torch.zeros(self.max_seq_length, dtype=torch.bool)
        mask[:seq_length] = True

        output = {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'items': items_tensor,
            'seq_length': torch.tensor(seq_length, dtype=torch.long),
            'mask': mask
        }

        if target is not None:
            output['target'] = torch.tensor(target, dtype=torch.long)

        return output


class NegativeSamplingDataset(Dataset):
    """
    Dataset with negative sampling for training

    For each positive sample, generate num_neg negative samples
    """

    def __init__(
        self,
        sequences: Dict[int, Dict],
        num_items: int,
        max_seq_length: int = 50,
        num_neg: int = 100
    ):
        """
        Args:
            sequences: user sequences
            num_items: total number of items
            max_seq_length: maximum sequence length
            num_neg: number of negative samples per positive
        """
        self.num_items = num_items
        self.max_seq_length = max_seq_length
        self.num_neg = num_neg

        # Build positive samples
        self.samples = []
        for user_idx, data in sequences.items():
            items = data['items']
            target = data['target']

            if target is not None and len(items) > 0:
                self.samples.append({
                    'user_idx': user_idx,
                    'items': items,
                    'target': target
                })

        logger.info(f"Created dataset with {len(self)} positive samples, {num_neg} negatives each")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        user_idx = sample['user_idx']
        items = sample['items']
        target = sample['target']

        # Truncate sequence
        if len(items) > self.max_seq_length:
            items = items[-self.max_seq_length:]

        seq_length = len(items)
        items_tensor = torch.zeros(self.max_seq_length, dtype=torch.long)
        items_tensor[:seq_length] = torch.tensor(items, dtype=torch.long)

        # Negative sampling
        neg_items = torch.randint(0, self.num_items, (self.num_neg,), dtype=torch.long)

        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'items': items_tensor,
            'seq_length': torch.tensor(seq_length, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'neg_items': neg_items
        }


if __name__ == '__main__':
    # Test dataset
    import pickle

    print("Testing SequentialDataset...")

    # Create dummy data
    sequences = {
        0: {'items': [1, 2, 3, 4], 'target': 5},
        1: {'items': [2, 3], 'target': 4},
        2: {'items': [1, 2, 3, 4, 5, 6], 'target': 7}
    }

    dataset = SequentialDataset(sequences, max_seq_length=10, split='train')

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Items shape: {sample['items'].shape}")
    print(f"Seq length: {sample['seq_length']}")
    print(f"Target: {sample['target']}")

    print("\nDataset test passed!")