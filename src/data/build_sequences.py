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

    def split_10fold_cv(
        self,
        user_sequences: Dict[int, List[Tuple[int, int]]],
        num_folds: int = 10,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        10-Fold Cross-Validation split (user-level)

        Strategy:
        1. Shuffle all users randomly
        2. Split into 10 equal folds
        3. For each fold: test users = fold_i users, train users = all other folds
        4. Each user's FULL sequence is used (no train/val/test split within user)

        Args:
            user_sequences: {user_idx: [(item, timestamp), ...]}
            num_folds: number of folds (default 10)
            random_seed: random seed for reproducibility

        Returns:
            train_sequences_list: list of 10 train dicts (one per fold)
            test_sequences_list: list of 10 test dicts (one per fold)
            Each dict: {user_idx: {'items': [(item, timestamp), ...], 'target': last_item}}
        """
        import numpy as np

        logger.info(f"Splitting with {num_folds}-fold cross-validation (user-level)...")

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Get all user indices and shuffle
        user_indices = list(user_sequences.keys())
        np.random.shuffle(user_indices)

        # Split into folds
        num_users = len(user_indices)
        fold_size = num_users // num_folds

        folds = []
        for i in range(num_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < num_folds - 1 else num_users
            fold_users = user_indices[start_idx:end_idx]
            folds.append(fold_users)

        # Log fold sizes
        logger.info(f"Created {num_folds} folds:")
        for i, fold in enumerate(folds):
            logger.info(f"  Fold {i + 1}: {len(fold)} users")

        # Create train/test splits for each fold
        train_sequences_list = []
        test_sequences_list = []

        for fold_idx in range(num_folds):
            # Test users for this fold
            test_users = folds[fold_idx]

            # Train users (all other folds)
            train_users = []
            for i, fold in enumerate(folds):
                if i != fold_idx:
                    train_users.extend(fold)

            # Create sequence dicts for this fold
            train_sequences = {}
            test_sequences = {}

            # Process train users: use full sequence, predict last item
            for user_idx in train_users:
                seq = user_sequences[user_idx]
                if len(seq) >= 2:  # Need at least 2 items: input + target
                    all_items = [item for item, _ in seq]
                    train_sequences[user_idx] = {
                        'items': seq[:-1],  # All but last item (with timestamps)
                        'target': all_items[-1]  # Last item as target
                    }

            # Process test users: use full sequence, predict last item
            for user_idx in test_users:
                seq = user_sequences[user_idx]
                if len(seq) >= 2:
                    all_items = [item for item, _ in seq]
                    test_sequences[user_idx] = {
                        'items': seq[:-1],
                        'target': all_items[-1]
                    }

            train_sequences_list.append(train_sequences)
            test_sequences_list.append(test_sequences)

            logger.info(f"Fold {fold_idx + 1}: "
                       f"Train={len(train_sequences)} users, "
                       f"Test={len(test_sequences)} users")

        return train_sequences_list, test_sequences_list

    def save_sequences(
        self,
        train_sequences: Dict,
        val_sequences: Dict = None,
        test_sequences: Dict = None,
        train_sequences_list: List[Dict] = None,
        test_sequences_list: List[Dict] = None
    ):
        """
        Save sequences to disk

        Args:
            train_sequences: single train split (for leave-one-out or ratio split)
            val_sequences: single val split (optional)
            test_sequences: single test split (optional)
            train_sequences_list: list of train splits (for CV, one per fold)
            test_sequences_list: list of test splits (for CV, one per fold)
        """
        logger.info("Saving sequences...")

        # Handle CV mode: save all folds
        if train_sequences_list is not None and test_sequences_list is not None:
            num_folds = len(train_sequences_list)
            logger.info(f"Saving {num_folds}-fold cross-validation splits...")

            cv_dir = self.processed_dir / 'cv_splits'
            cv_dir.mkdir(parents=True, exist_ok=True)

            for fold_idx in range(num_folds):
                fold_dir = cv_dir / f'fold_{fold_idx + 1}'
                fold_dir.mkdir(parents=True, exist_ok=True)

                with open(fold_dir / 'train_sequences.pkl', 'wb') as f:
                    pickle.dump(train_sequences_list[fold_idx], f)

                with open(fold_dir / 'test_sequences.pkl', 'wb') as f:
                    pickle.dump(test_sequences_list[fold_idx], f)

            # Also save metadata about CV split
            cv_metadata = {
                'num_folds': num_folds,
                'split_method': '10fold_cv',
                'folds': [{'train_size': len(train_sequences_list[i]),
                          'test_size': len(test_sequences_list[i])}
                         for i in range(num_folds)]
            }
            with open(cv_dir / 'cv_metadata.pkl', 'wb') as f:
                pickle.dump(cv_metadata, f)

            logger.info(f"Saved CV splits to {cv_dir}")

        # Handle single split mode (leave-one-out or ratio)
        else:
            with open(self.processed_dir / 'train_sequences.pkl', 'wb') as f:
                pickle.dump(train_sequences, f)

            if val_sequences is not None:
                with open(self.processed_dir / 'val_sequences.pkl', 'wb') as f:
                    pickle.dump(val_sequences, f)

            if test_sequences is not None:
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

        # Split based on method
        if self.split_method == '10fold_cv':
            # 10-fold cross-validation
            num_folds = self.config['dataset'].get('num_folds', 10)
            random_seed = self.config['dataset'].get('cv_random_seed', 42)
            train_sequences_list, test_sequences_list = self.split_10fold_cv(
                user_sequences,
                num_folds=num_folds,
                random_seed=random_seed
            )
            # Save CV splits
            self.save_sequences(
                train_sequences_list=train_sequences_list,
                test_sequences_list=test_sequences_list
            )
        elif self.split_method == 'leave_one_out':
            # Leave-one-out split
            train_sequences, val_sequences, test_sequences = self.split_leave_one_out(user_sequences)
            self.save_sequences(train_sequences, val_sequences, test_sequences)
        else:
            # Ratio-based split
            train_sequences, val_sequences, test_sequences = self.split_ratio(user_sequences)
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