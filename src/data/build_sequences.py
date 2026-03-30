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
    """Build user sequences and split dataset (cv10 mode only)"""

    def __init__(self, config: Dict):
        self.config = config
        self.processed_dir = Path(config['dataset']['processed_dir'])

        # Split strategy: ONLY 'cv10' (10-fold CV) supported
        # Paper-aligned: User-level 10-fold cross-validation
        self.split_method = 'cv10'

        # CV parameters
        self.num_folds = config['dataset'].get('num_folds', 10)
        self.random_seed = config['dataset'].get('cv_random_seed', 42)

    def build_user_sequences(self, df: pd.DataFrame) -> Dict[int, List[Tuple[int, int]]]:
        """
        Build user sequences from interaction dataframe

        **PAPER-ALIGNED FORMAT**: Outputs [(item, timestamp), ...] tuples
        - Required for Phase 2 (Item Graph): session construction via timestamps
        - Required for Phase 3 (Item-Time Graph): day-level time node construction

        Args:
            df: DataFrame with columns [user_idx, item_idx, timestamp]

        Returns:
            user_sequences: {user_idx: [(item_idx1, timestamp1), (item_idx2, timestamp2), ...]}
            Each entry is a (item, timestamp) tuple preserving original order.

        **Format Evolution**:
        - Old format (engineering compatibility): [item1, item2, ...]
        - New format (paper-aligned): [(item1, ts1), (item2, ts2), ...]
        - Both formats supported by downstream modules for backward compatibility
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



    def split_cv10(
        self,
        user_sequences: Dict[int, List[Tuple[int, int]]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        10-Fold Cross-Validation split (ONLY mode supported)

        Strategy:
        1. Shuffle all users randomly
        2. Split into 10 equal folds
        3. For each fold: test users = fold_i users, train users = all other folds
        4. Each user's FULL sequence is used (no train/val/test split within user)

        **PAPER-ALIGNED OUTPUT FORMAT**:
        - 'items': [(item, timestamp), ...] with timestamps preserved
        - 'target': item_id only (last item in sequence)

        Args:
            user_sequences: {user_idx: [(item, timestamp), ...]}

        Returns:
            train_sequences_list: list of 10 train dicts (one per fold)
            test_sequences_list: list of 10 test dicts (one per fold)
            Each dict: {user_idx: {'items': [(item, timestamp), ...], 'target': last_item}}
        """
        import numpy as np

        num_folds = self.num_folds
        random_seed = self.random_seed

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
        train_sequences_list: List[Dict],
        test_sequences_list: List[Dict]
    ):
        """
        Save CV10 sequences to disk (ONLY mode supported)

        Args:
            train_sequences_list: list of train splits (for CV, one per fold)
            test_sequences_list: list of test splits (for CV, one per fold)
        """
        logger.info("Saving sequences...")

        # CV mode: save all folds
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

        # Save CV metadata
        # **CONSUMERS**: Phase 1/4 CV training scripts need num_folds and fold sizes
        cv_metadata = {
            'num_folds': num_folds,
            'split_method': 'cv10',
            'folds': [{'train_size': len(train_sequences_list[i]),
                      'test_size': len(test_sequences_list[i])}
                     for i in range(num_folds)],
            'random_seed': self.random_seed,  # Fixed: use instance variable
            'total_users': sum(len(s) for s in train_sequences_list) + len(test_sequences_list[0]),
            'version': '1.1',
        }
        with open(cv_dir / 'cv_metadata.pkl', 'wb') as f:
            pickle.dump(cv_metadata, f)

        logger.info(f"Saved CV splits to {cv_dir}")

    def run(self, df: pd.DataFrame):
        """Run sequence building and splitting (cv10 mode only)"""
        logger.info("=" * 60)
        logger.info("Building sequences and splitting dataset (10-fold CV)")
        logger.info("=" * 60)

        # Build sequences
        user_sequences = self.build_user_sequences(df)

        # Split using cv10 (ONLY mode supported)
        train_sequences_list, test_sequences_list = self.split_cv10(user_sequences)

        # Save CV splits
        self.save_sequences(
            train_sequences_list=train_sequences_list,
            test_sequences_list=test_sequences_list
        )

        logger.info("Sequence building complete!")
        logger.info("=" * 60)


if __name__ == '__main__':
    import yaml
    import sys

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/tafeng_baseline.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load processed interactions
    processed_dir = Path(config['dataset']['processed_dir'])
    df = pd.read_csv(processed_dir / 'interactions.csv')

    logger.info(f"Loaded {len(df)} interactions")

    # Build sequences
    builder = SequenceBuilder(config)
    builder.run(df)