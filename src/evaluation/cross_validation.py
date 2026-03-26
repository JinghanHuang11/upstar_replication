"""
10-Fold Cross-Validation for UPSTAR

Strategy: User-level 10-fold split
1. Shuffle all users randomly
2. Split into 10 equal folds
3. For each fold: use as test set, remaining 9 folds as train
4. Train 10 models, evaluate on corresponding fold
5. Report mean ± std across 10 runs

Key points:
- Each user appears in only 1 fold
- Roughly equal users per fold
- Maintain sequence integrity (don't split user sequences)
- Fixed random seed for reproducibility
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging
import json
import shutil

from src.models.upstar import UPSTARMoodel
from src.data.upstar_dataset import get_upstar_dataloader
from src.training.losses import UPSTARLoss, create_loss_from_config
from src.evaluation.metrics import (
    compute_all_metrics,
    print_main_results,
    compute_mean_std,
    print_mean_std
)
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    10-Fold Cross-Validation for UPSTAR

    Handles:
    - Data splitting into 10 folds
    - Training and evaluation for each fold
    - Aggregating results (mean ± std)
    - Saving intermediate results
    """

    def __init__(
        self,
        config: Dict,
        num_folds: int = 10,
        random_seed: int = 42
    ):
        """
        Initialize cross-validator

        Args:
            config: configuration dict
            num_folds: number of folds (default 10)
            random_seed: random seed for reproducibility
        """
        self.config = config
        self.num_folds = num_folds
        self.random_seed = random_seed

        # Load data
        self._load_data()

        # Create folds
        self._create_folds()

    def _load_data(self):
        """Load all sequences"""
        processed_dir = Path(self.config['dataset']['processed_dir'])

        # Load metadata
        with open(processed_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        self.num_items = metadata['num_items']

        # Load train sequences (for CV, we use full dataset)
        with open(processed_dir / 'train_sequences.pkl', 'rb') as f:
            all_sequences = pickle.load(f)

        self.all_sequences = all_sequences

        # Load STB results
        stb_scores_path = Path(self.config['stb']['motivation_labels_path'])
        if stb_scores_path.exists():
            self.motivation_labels = np.load(stb_scores_path)
        else:
            self.motivation_labels = None

        logger.info(f"Loaded {len(all_sequences)} users for cross-validation")

    def _create_folds(self):
        """
        Create 10 folds of users

        Strategy:
        1. Get all user indices
        2. Shuffle randomly
        3. Split into 10 equal folds
        """
        set_seed(self.random_seed)

        # Get all user indices
        user_indices = list(self.all_sequences.keys())

        # Shuffle
        np.random.shuffle(user_indices)

        # Split into folds
        num_users = len(user_indices)
        fold_size = num_users // self.num_folds

        self.folds = []
        for i in range(self.num_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.num_folds - 1 else num_users

            fold_users = user_indices[start_idx:end_idx]
            self.folds.append(fold_users)

        # Log fold sizes
        logger.info(f"Created {self.num_folds} folds:")
        for i, fold in enumerate(self.folds):
            logger.info(f"  Fold {i + 1}: {len(fold)} users")

    def get_fold_data(
        self,
        fold_idx: int
    ) -> Tuple[Dict, Dict]:
        """
        Get train and test data for a fold

        Args:
            fold_idx: fold index (0-9)

        Returns:
            train_sequences: dict of training sequences
            test_sequences: dict of test sequences
        """
        # Get test users for this fold
        test_users = self.folds[fold_idx]

        # Get train users (all other folds)
        train_users = []
        for i, fold in enumerate(self.folds):
            if i != fold_idx:
                train_users.extend(fold)

        # Create sequence dicts
        train_sequences = {uid: self.all_sequences[uid] for uid in train_users}
        test_sequences = {uid: self.all_sequences[uid] for uid in test_users}

        logger.info(f"Fold {fold_idx + 1}: "
                   f"Train={len(train_sequences)}, Test={len(test_sequences)}")

        return train_sequences, test_sequences

    def train_fold(
        self,
        fold_idx: int,
        device: torch.device,
        output_dir: Path
    ) -> Dict[str, float]:
        """
        Train and evaluate a single fold

        Args:
            fold_idx: fold index (0-9)
            device: device to use
            output_dir: directory to save results

        Returns:
            metrics: dict of metric names and values
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Training Fold {fold_idx + 1}/{self.num_folds}")
        logger.info(f"{'=' * 80}")

        # Get fold data
        train_sequences, test_sequences = self.get_fold_data(fold_idx)

        # Create dataloaders
        max_seq_length = self.config['model'].get('max_seq_length', 50)
        batch_size = self.config['training']['batch_size']

        train_loader = get_upstar_dataloader(
            sequences=train_sequences,
            motivation_labels=self.motivation_labels,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            split='train',
            num_workers=0
        )

        test_loader = get_upstar_dataloader(
            sequences=test_sequences,
            motivation_labels=self.motivation_labels,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            split='test',
            num_workers=0
        )

        # Create model
        model = UPSTARMoodel(
            num_items=self.num_items,
            embed_dim=self.config['model']['embed_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            use_gate=self.config['model']['use_gate']
        ).to(device)

        # Create loss function
        loss_fn = create_loss_from_config(self.config)

        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Training parameters
        max_epochs = self.config['training']['max_epochs_per_stage']
        patience = self.config['training']['early_stop_patience']
        k_values = self.config['evaluation']['k_values']

        # Training loop
        best_metric = 0.0
        no_improve_count = 0

        fold_output_dir = output_dir / f'fold_{fold_idx + 1}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(max_epochs):
            # Train
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                # Move to device
                seq_stable = batch['seq_stable'].to(device)
                len_stable = batch['len_stable'].to(device)
                seq_exploratory = batch['seq_exploratory'].to(device)
                len_exploratory = batch['len_exploratory'].to(device)
                seq_entire = batch['seq_entire'].to(device)
                len_entire = batch['len_entire'].to(device)
                targets = batch['target'].to(device)
                motivation_labels_batch = batch['motivation_labels'].to(device)

                # Forward
                optimizer.zero_grad()
                output = model(
                    seq_stable, len_stable,
                    seq_exploratory, len_exploratory,
                    seq_entire, len_entire
                )

                # Loss
                predictions = {
                    'y_hat_global': output['y_hat_global'],
                    'y_hat_stab': output['y_hat_stab'],
                    'y_hat_expl': output['y_hat_expl'],
                    'y_hat_other': output['y_hat_other'],
                    'z_stab': output['z_stab'],
                    'z_expl': output['z_expl'],
                    'z_other': output['z_other']
                }

                loss_output = loss_fn(predictions, targets, motivation_labels_batch)
                loss = loss_output['total']

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Evaluate
            model.eval()
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch in test_loader:
                    seq_stable = batch['seq_stable'].to(device)
                    len_stable = batch['len_stable'].to(device)
                    seq_exploratory = batch['seq_exploratory'].to(device)
                    len_exploratory = batch['len_exploratory'].to(device)
                    seq_entire = batch['seq_entire'].to(device)
                    len_entire = batch['len_entire'].to(device)
                    targets = batch['target'].to(device)

                    output = model(
                        seq_stable, len_stable,
                        seq_exploratory, len_exploratory,
                        seq_entire, len_entire
                    )

                    all_predictions.append(output['y_hat_global'].cpu())
                    all_targets.append(targets.cpu())

            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Compute metrics
            metrics = compute_all_metrics(all_predictions, all_targets, k_values=k_values)
            current_metric = metrics[self.config['training']['metric_for_best']]

            # Log
            logger.info(f"Epoch {epoch + 1}/{max_epochs}: "
                       f"Loss={train_loss:.4f}, "
                       f"NDCG@10={metrics['NDCG@10']:.4f}")

            # Check if best
            if current_metric > best_metric:
                best_metric = current_metric
                no_improve_count = 0

                # Save best model
                best_path = fold_output_dir / 'best_model.pt'
                torch.save({
                    'fold': fold_idx,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }, best_path)
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model and evaluate
        best_checkpoint = torch.load(fold_output_dir / 'best_model.pt', weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        final_metrics = best_checkpoint['metrics']

        # Save final metrics
        metrics_path = fold_output_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)

        logger.info(f"Fold {fold_idx + 1} complete: NDCG@10={final_metrics['NDCG@10']:.4f}")

        return final_metrics

    def run_cross_validation(
        self,
        device: torch.device,
        output_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run full 10-fold cross-validation

        Args:
            device: device to use
            output_dir: directory to save results

        Returns:
            cv_results: dict with mean and std for each metric
        """
        if output_dir is None:
            output_dir = Path(self.config['logging']['checkpoint_dir']) / 'cross_validation'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Starting {self.num_folds}-Fold Cross-Validation")
        logger.info(f"{'=' * 80}")
        logger.info(f"Output directory: {output_dir}")

        # Train and evaluate each fold
        all_fold_metrics = []

        for fold_idx in range(self.num_folds):
            fold_metrics = self.train_fold(fold_idx, device, output_dir)
            all_fold_metrics.append(fold_metrics)

        # Aggregate results
        cv_results = compute_mean_std(all_fold_metrics)

        # Print results
        print_mean_std(cv_results, f"{self.num_folds}-Fold Cross-Validation Results")

        # Save aggregated results
        results_path = output_dir / 'cv_results.json'
        with open(results_path, 'w') as f:
            # Convert to percentage for saving
            cv_results_pct = {}
            for metric_name, stats in cv_results.items():
                cv_results_pct[metric_name] = {
                    'mean': stats['mean'] * 100,
                    'std': stats['std'] * 100
                }
            json.dump(cv_results_pct, f, indent=2)

        logger.info(f"\nSaved cross-validation results to {results_path}")

        # Save per-fold results
        fold_results_path = output_dir / 'per_fold_results.json'
        with open(fold_results_path, 'w') as f:
            fold_results_pct = []
            for metrics in all_fold_metrics:
                metrics_pct = {k: v * 100 for k, v in metrics.items()}
                fold_results_pct.append(metrics_pct)
            json.dump(fold_results_pct, f, indent=2)

        logger.info(f"Saved per-fold results to {fold_results_path}")

        return cv_results


def run_cross_validation(
    config_path: str,
    num_folds: int = 10,
    output_dir: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to run cross-validation

    Args:
        config_path: path to config file
        num_folds: number of folds
        output_dir: directory to save results

    Returns:
        cv_results: dict with mean and std for each metric
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config.get('seed', 42))

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create cross-validator
    cv = CrossValidator(
        config=config,
        num_folds=num_folds,
        random_seed=config.get('seed', 42)
    )

    # Run cross-validation
    cv_results = cv.run_cross_validation(device, output_dir)

    return cv_results


if __name__ == '__main__':
    print("Testing Cross-Validator...")

    # This would require actual config
    # For testing, just verify imports work
    from src.evaluation.metrics import compute_mean_std

    # Test mean/std computation
    dummy_metrics = [
        {'NDCG@10': 0.05, 'Precision@5': 0.03},
        {'NDCG@10': 0.06, 'Precision@5': 0.04},
        {'NDCG@10': 0.055, 'Precision@5': 0.035},
    ]

    stats = compute_mean_std(dummy_metrics)
    print("\nTest statistics:")
    for metric_name, metric_stats in stats.items():
        print(f"  {metric_name}: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}")

    print("\nCross-validator module loaded successfully!")
