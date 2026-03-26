"""
Build user sequences and split into train/val/test
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """Build user sequences and split dataset"""

    def __init__(self, config: Dict):
        self.config = config
        self.processed_dir = Path(config['dataset']['processed_dir'])

        # Split strategy
        self.split_method = config['dataset'].get('split_method', 'leave_one_out')
        self.split_ratio = config['dataset'].get('split_ratio', [0.8, 0.1, 0.1])

    def build_user_sequences(self, df: pd.DataFrame) -> Dict[int, List[Tuple[int, int]]]:
        """
        Build user sequences from interaction dataframe

        Args:
            df: DataFrame with columns [user_idx, item_idx, timestamp]

        Returns:
            user_sequences: {user_idx: [(item_idx1, timestamp1), (item_idx2, timestamp2), ...]}
            Each entry is a (item, timestamp) tuple preserving original order.
        """
        logger.info("Building user sequences (with timestamps)...")

        user_sequences = defaultdict(list)

        for _, row in df.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            timestamp = int(row['timestamp'])
            user_sequences[user_idx].append((item_idx, timestamp))

        # Sort each user's sequence by timestamp (already sorted in df)
        logger.info(f"Built sequences for {len(user_sequences)} users")

        return dict(user_sequences)

    def split_leave_one_out(
        self,
        user_sequences: Dict[int, List[Tuple[int, int]]]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Leave-one-out split:
        - Train: all items except last two
        - Val: second to last item
        - Test: last item

        Args:
            user_sequences: {user_idx: [(item, timestamp), ...]}

        Returns:
            train_sequences, val_sequences, test_sequences
            Each items field is a list of (item, timestamp) tuples.
        """
        logger.info("Splitting with leave-one-out strategy...")

        train_sequences = {}
        val_sequences = {}
        test_sequences = {}

        for user_idx, seq in user_sequences.items():
            if len(seq) < 3:
                # Need at least 3 items for train/val/test
                continue

            # Extract items only (not timestamps) for target
            all_items = [item for item, _ in seq]
            all_items_timestamps = seq

            # For training: use items[:-2] as input, items[-2] as target
            train_sequences[user_idx] = {
                'items': all_items_timestamps[:-2],
                'target': all_items[-2]
            }

            # For validation: use items[:-1] as input, items[-1] as target
            val_sequences[user_idx] = {
                'items': all_items_timestamps[:-1],
                'target': all_items[-1]
            }

            # For testing: use items[:-1] as input, last item as target
            test_sequences[user_idx] = {
                'items': all_items_timestamps[:-1],
                'target': all_items[-1]
            }

        logger.info(f"Train: {len(train_sequences)} users")
        logger.info(f"Val: {len(val_sequences)} users")
        logger.info(f"Test: {len(test_sequences)} users")

        return train_sequences, val_sequences, test_sequences

    def split_ratio(
        self,
        user_sequences: Dict[int, List[Tuple[int, int]]]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Ratio-based split:
        - Train: first split_ratio[0] of each user's sequence
        - Val: next split_ratio[1]
        - Test: last split_ratio[2]

        Args:
            user_sequences: {user_idx: [(item, timestamp), ...]}

        Returns:
            train_sequences, val_sequences, test_sequences
            Each items field is a list of (item, timestamp) tuples.
        """
        logger.info(f"Splitting with ratio {self.split_ratio}...")

        train_sequences = {}
        val_sequences = {}
        test_sequences = {}

        for user_idx, seq in user_sequences.items():
            n = len(seq)

            train_end = int(n * self.split_ratio[0])
            val_end = train_end + int(n * self.split_ratio[1])

            # Extract items for targets
            all_items = [item for item, _ in seq]

            # Train
            if train_end > 0:
                train_sequences[user_idx] = {
                    'items': seq[:train_end],
                    'target': all_items[train_end] if train_end < n else None
                }

            # Val
            if val_end > train_end and val_end < n:
                val_sequences[user_idx] = {
                    'items': seq[:val_end],
                    'target': all_items[val_end]
                }

            # Test
            if n > val_end:
                test_sequences[user_idx] = {
                    'items': seq[:n-1],
                    'target': all_items[n-1]
                }

        logger.info(f"Train: {len(train_sequences)} users")
        logger.info(f"Val: {len(val_sequences)} users")
        logger.info(f"Test: {len(test_sequences)} users")

        return train_sequences, val_sequences, test_sequences

    def save_sequences(
        self,
        train_sequences: Dict,
        val_sequences: Dict,
        test_sequences: Dict
    ):
        """Save sequences to disk"""
        logger.info("Saving sequences...")

        with open(self.processed_dir / 'train_sequences.pkl', 'wb') as f:
            pickle.dump(train_sequences, f)

        with open(self.processed_dir / 'val_sequences.pkl', 'wb') as f:
            pickle.dump(val_sequences, f)

        with open(self.processed_dir / 'test_sequences.pkl', 'wb') as f:
            pickle.dump(test_sequences, f)

        logger.info(f"Saved sequences to {self.processed_dir}")

    def run(self, df: pd.DataFrame):
        """Run sequence building and splitting"""
        logger.info("=" * 60)
        logger.info("Building sequences and splitting dataset")
        logger.info("=" * 60)

        # Build sequences
        user_sequences = self.build_user_sequences(df)

        # Split
        if self.split_method == 'leave_one_out':
            train_sequences, val_sequences, test_sequences = self.split_leave_one_out(user_sequences)
        else:
            train_sequences, val_sequences, test_sequences = self.split_ratio(user_sequences)

        # Save
        self.save_sequences(train_sequences, val_sequences, test_sequences)

        logger.info("Sequence building complete!")
        logger.info("=" * 60)


if __name__ == '__main__':
    import yaml
    import sys

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/baseline.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load processed interactions
    processed_dir = Path(config['dataset']['processed_dir'])
    df = pd.read_csv(processed_dir / 'interactions.csv')

    logger.info(f"Loaded {len(df)} interactions")

    # Build sequences
    builder = SequenceBuilder(config)
    builder.run(df)