"""
Data preprocessing module
Loads raw data, builds vocabulary, and saves processed data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import pickle
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocessor for sequential recommendation datasets"""

    def __init__(self, config: Dict):
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.data_dir = Path(config['dataset']['data_dir'])
        self.processed_dir = Path(config['dataset']['processed_dir'])

        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Vocabulary
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}

        # Statistics
        self.num_users = 0
        self.num_items = 0
        self.num_interactions = 0

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from files
        Expected columns: user_id, item_id, timestamp

        Returns:
            DataFrame with columns [user_id, item_id, timestamp]
        """
        logger.info(f"Loading raw data from {self.data_dir}")

        # Try different file formats
        data_file = None

        # First try exact name match
        for ext in ['.csv', '.txt', '.dat']:
            candidate = self.data_dir / f"{self.dataset_name}{ext}"
            if candidate.exists():
                data_file = candidate
                break

        # Try with underscores (for Ta-Feng style naming)
        if data_file is None:
            # Replace spaces/underscores in dataset name
            name_variants = [
                self.dataset_name,
                self.dataset_name.replace('_', ''),
                self.dataset_name.replace('-', ''),
                self.dataset_name.replace(' ', ''),
                self.dataset_name.lower().replace('_', '').replace('-', '').replace(' ', ''),
                # Ta-Feng specific: try ta_feng for tafeng
                self.dataset_name.lower().replace('tafeng', 'ta_feng'),
            ]
            # Remove duplicates
            name_variants = list(dict.fromkeys(name_variants))

            for name in name_variants:
                for ext in ['.csv', '.txt', '.dat']:
                    candidate = self.data_dir / f"{name}{ext}"
                    if candidate.exists():
                        data_file = candidate
                        break
                if data_file:
                    break

        if data_file is None:
            # Try finding any CSV file in the directory (excluding directories)
            csv_files = list(self.data_dir.glob('*.csv'))
            for f in csv_files:
                if f.is_file() and f.name != '.gitkeep':
                    data_file = f
                    break

        if data_file is None:
            raise FileNotFoundError(f"No data file found in {self.data_dir}")

        logger.info(f"Loading from {data_file}")

        # Load based on extension
        if data_file.suffix == '.csv':
            df = pd.read_csv(data_file)
        else:
            # Try CSV format first, then whitespace-separated
            try:
                df = pd.read_csv(data_file, sep='\t')
            except:
                df = pd.read_csv(data_file, delim_whitespace=True)

        logger.info(f"Loaded {len(df)} rows")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Ensure required columns exist
        # Make column names case-insensitive by converting to lowercase for comparison
        df.columns = df.columns.str.lower().str.strip()
        required_cols = ['user_id', 'item_id', 'timestamp']
        for col in required_cols:
            if col not in df.columns:
                # Try common variations
                variations = {
                    'user_id': ['user', 'uid', 'customer', 'userid', 'customer_id', 'customer_no'],
                    'item_id': ['item', 'iid', 'product', 'itemid', 'product_id', 'item_no'],
                    'timestamp': ['time', 'date', 'ts', 'datetime', 'transaction_dt', 'trans_date']
                }

                found = False
                for var in variations.get(col, []):
                    if var in df.columns:
                        df = df.rename(columns={var: col})
                        found = True
                        break

                if not found:
                    # Create timestamp if missing
                    if col == 'timestamp':
                        logger.warning("No timestamp column found, creating artificial timestamps")
                        df['timestamp'] = np.arange(len(df))
                    else:
                        raise ValueError(f"Required column '{col}' not found in data")

        return df[required_cols]

    def build_vocabulary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build user and item vocabulary
        Map IDs to continuous indices
        """
        logger.info("Building vocabulary...")

        # Get unique users and items
        unique_users = df['user_id'].unique()
        unique_items = df['item_id'].unique()

        # Build mappings
        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}

        self.item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx2item = {idx: iid for iid, idx in self.item2idx.items()}

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        logger.info(f"Vocabulary built: {self.num_users} users, {self.num_items} items")

        # Map to indices
        df['user_idx'] = df['user_id'].map(self.user2idx)
        df['item_idx'] = df['item_id'].map(self.item2idx)

        return df

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data:
        - Remove duplicates
        - Sort by user and timestamp
        - Filter by min user interactions and min item frequency
        """
        logger.info("Filtering data...")

        # Remove duplicates
        df = df.drop_duplicates(subset=['user_idx', 'item_idx', 'timestamp'])
        logger.info(f"After removing duplicates: {len(df)} rows")

        # Sort by user and timestamp
        df = df.sort_values(['user_idx', 'timestamp']).reset_index(drop=True)

        # Filter by minimum user interactions
        min_user_interactions = self.config['dataset'].get('min_user_interactions', 5)
        user_counts = df['user_idx'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_idx'].isin(valid_users)].copy()
        logger.info(f"After filtering users (min {min_user_interactions} interactions): {len(df)} rows")

        # Filter by minimum item frequency
        min_item_freq = self.config['dataset'].get('min_item_frequency', 5)
        item_counts = df['item_idx'].value_counts()
        valid_items = item_counts[item_counts >= min_item_freq].index
        df = df[df['item_idx'].isin(valid_items)].copy()
        logger.info(f"After filtering items (min {min_item_freq} frequency): {len(df)} rows")

        # Rebuild vocabulary after filtering
        df = self.build_vocabulary(df)

        self.num_interactions = len(df)

        logger.info(f"Final dataset: {self.num_users} users, {self.num_items} items, {self.num_interactions} interactions")

        return df

    def save_metadata(self):
        """Save vocabulary and metadata"""
        metadata = {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_interactions': self.num_interactions,
            'user2idx': self.user2idx,
            'idx2user': self.idx2user,
            'item2idx': self.item2idx,
            'idx2item': self.idx2item
        }

        metadata_path = self.processed_dir / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved metadata to {metadata_path}")

    def run(self) -> pd.DataFrame:
        """Run full preprocessing pipeline"""
        logger.info("=" * 60)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 60)

        # Load
        df = self.load_raw_data()

        # Build vocabulary
        df = self.build_vocabulary(df)

        # Filter
        df = self.filter_data(df)

        # Save metadata
        self.save_metadata()

        logger.info("Preprocessing complete!")
        logger.info("=" * 60)

        return df


class TafengPreprocessor(Preprocessor):
    """Preprocessor for Tafeng dataset"""

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load Tafeng dataset
        Format: TRANSACTION_DT, CUSTOMER_ID, PRODUCT_ID, etc.
        """
        logger.info("Loading Tafeng dataset...")

        data_file = self.data_dir / 'ta_feng.csv'

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Load with encoding to handle BOM
        df = pd.read_csv(data_file, encoding='utf-8-sig')

        logger.info(f"Loaded {len(df)} rows from Tafeng")

        # Rename columns to standard format
        df = df.rename(columns={
            'CUSTOMER_ID': 'user_id',
            'PRODUCT_ID': 'item_id',
            'TRANSACTION_DT': 'timestamp'
        })

        # Convert date to timestamp
        # Format: "11/1/2000" -> Unix timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y')
        df['timestamp'] = df['timestamp'].astype(np.int64) // 10**9

        logger.info(f"Converted {len(df)} transactions")

        return df[['user_id', 'item_id', 'timestamp']]


def get_preprocessor(config: Dict) -> Preprocessor:
    """Factory function to get dataset preprocessor"""
    dataset_name = config['dataset']['name']

    preprocessors = {
        'tafeng': TafengPreprocessor,
        'baseline': Preprocessor,
    }

    if dataset_name not in preprocessors:
        logger.warning(f"No specific preprocessor for {dataset_name}, using base Preprocessor")
        return Preprocessor(config)

    return preprocessors[dataset_name](config)


if __name__ == '__main__':
    import yaml
    import sys

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/baseline.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Run preprocessing
    preprocessor = get_preprocessor(config)
    df = preprocessor.run()

    # Save processed dataframe
    output_path = Path(config['dataset']['processed_dir']) / 'interactions.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed interactions to {output_path}")