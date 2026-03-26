"""
Unified Evaluator for UPSTAR

Decouples evaluation from training loop.
Supports single evaluation and cross-validation.
"""

import torch
import torch.nn as nn
import yaml
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
import logging
import json
import csv

from src.models.upstar import UPSTARMoodel
from src.data.upstar_dataset import get_upstar_dataloader
from src.evaluation.metrics import (
    compute_all_metrics,
    print_metrics,
    print_main_results,
    format_metrics_percentage
)

logger = logging.getLogger(__name__)


class UPSTAREvaluator:
    """
    Unified evaluator for UPSTAR model

    Handles:
    - Model loading
    - Data loading
    - Metric computation
    - Result saving
    """

    def __init__(
        self,
        config: Dict,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize evaluator

        Args:
            config: configuration dict
            checkpoint_path: path to model checkpoint
            device: device to use ('cuda' or 'cpu')
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Load data
        self._load_data()

        # Load model if checkpoint provided
        self.model = None
        if checkpoint_path:
            self._load_model()

    def _load_data(self):
        """Load test data"""
        processed_dir = Path(self.config['dataset']['processed_dir'])

        # Load metadata
        with open(processed_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        self.num_items = metadata['num_items']

        # Load test sequences
        with open(processed_dir / 'test_sequences.pkl', 'rb') as f:
            self.test_sequences = pickle.load(f)

        # Load STB results if needed
        stb_scores_path = Path(self.config['stb']['motivation_labels_path'])
        if stb_scores_path.exists():
            self.motivation_labels = np.load(stb_scores_path)
        else:
            logger.warning(f"Motivation labels not found: {stb_scores_path}")
            self.motivation_labels = None

        logger.info(f"Loaded test data: {len(self.test_sequences)} users")

    def _load_model(self):
        """Load model from checkpoint"""
        checkpoint = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)

        # Create model
        self.model = UPSTARMoodel(
            num_items=self.num_items,
            embed_dim=self.config['model']['embed_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            use_gate=self.config['model']['use_gate']
        ).to(self.device)

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Loaded model from {self.checkpoint_path}")

        # Print checkpoint info if available
        if 'epoch' in checkpoint:
            logger.info(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            logger.info(f"  Checkpoint metrics: {checkpoint['metrics']}")

    def evaluate(
        self,
        split: str = 'test',
        k_values: Optional[List[int]] = None,
        save_predictions: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on specified split

        Args:
            split: 'train', 'val', or 'test'
            k_values: list of k values for metrics
            save_predictions: if True, save predictions
            output_dir: directory to save results

        Returns:
            metrics: dict of metric names and values
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please provide checkpoint_path.")

        # Use default k values from config if not specified
        if k_values is None:
            k_values = self.config['evaluation'].get('k_values', [5, 10, 20])

        # Load sequences for split
        if split == 'test':
            sequences = self.test_sequences
        elif split == 'val':
            with open(Path(self.config['dataset']['processed_dir']) / 'val_sequences.pkl', 'rb') as f:
                sequences = pickle.load(f)
        else:  # train
            with open(Path(self.config['dataset']['processed_dir']) / 'train_sequences.pkl', 'rb') as f:
                sequences = pickle.load(f)

        # Create dataloader
        max_seq_length = self.config['model'].get('max_seq_length', 50)
        batch_size = self.config['evaluation'].get('batch_size', 256)

        dataloader = get_upstar_dataloader(
            sequences=sequences,
            motivation_labels=self.motivation_labels,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            split=split,
            num_workers=0
        )

        # Evaluate
        all_predictions = []
        all_targets = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
                # Move to device
                seq_stable = batch['seq_stable'].to(self.device)
                len_stable = batch['len_stable'].to(self.device)
                seq_exploratory = batch['seq_exploratory'].to(self.device)
                len_exploratory = batch['len_exploratory'].to(self.device)
                seq_entire = batch['seq_entire'].to(self.device)
                len_entire = batch['len_entire'].to(self.device)
                targets = batch['target'].to(self.device)

                # Forward pass
                output = self.model(
                    seq_stable, len_stable,
                    seq_exploratory, len_exploratory,
                    seq_entire, len_entire
                )

                # Use global predictions
                predictions = output['y_hat_global']

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        # Concatenate
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = compute_all_metrics(all_predictions, all_targets, k_values=k_values)

        # Print results
        print_main_results(metrics, f"{split.capitalize()} Results")

        # Save results if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics as JSON
            metrics_path = output_dir / f'{split}_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_path}")

            # Save metrics as CSV
            csv_path = output_dir / f'{split}_metrics.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for name, value in metrics.items():
                    writer.writerow([name, f'{value * 100:.2f}%'])
            logger.info(f"Saved CSV to {csv_path}")

            # Save predictions if requested
            if save_predictions:
                pred_path = output_dir / f'{split}_predictions.pkl'
                with open(pred_path, 'wb') as f:
                    pickle.dump({
                        'predictions': all_predictions.numpy(),
                        'targets': all_targets.numpy()
                    }, f)
                logger.info(f"Saved predictions to {pred_path}")

        return metrics

    def evaluate_all_splits(
        self,
        k_values: Optional[List[int]] = None,
        save_predictions: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on all splits (train, val, test)

        Args:
            k_values: list of k values
            save_predictions: if True, save predictions
            output_dir: directory to save results

        Returns:
            all_metrics: dict with metrics for each split
        """
        all_metrics = {}

        for split in ['train', 'val', 'test']:
            logger.info(f"\nEvaluating on {split} split...")
            metrics = self.evaluate(
                split=split,
                k_values=k_values,
                save_predictions=save_predictions,
                output_dir=output_dir
            )
            all_metrics[split] = metrics

        return all_metrics


def evaluate_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    split: str = 'test',
    k_values: Optional[List[int]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Convenience function to evaluate from checkpoint

    Args:
        checkpoint_path: path to model checkpoint
        config_path: path to config file
        split: 'train', 'val', or 'test'
        k_values: list of k values
        output_dir: directory to save results

    Returns:
        metrics: dict of metric names and values
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create evaluator
    evaluator = UPSTAREvaluator(
        config=config,
        checkpoint_path=checkpoint_path
    )

    # Evaluate
    metrics = evaluator.evaluate(
        split=split,
        k_values=k_values,
        save_predictions=True,
        output_dir=output_dir or Path(checkpoint_path).parent
    )

    return metrics


if __name__ == '__main__':
    print("Testing UPSTAR Evaluator...")

    # This would require actual checkpoint and config
    # For testing, just verify imports work
    from src.evaluation.metrics import compute_all_metrics

    # Create dummy data
    batch_size = 100
    num_items = 1000
    predictions = torch.randn(batch_size, num_items)
    targets = torch.randint(0, num_items, (batch_size,))

    # Test metrics
    metrics = compute_all_metrics(predictions, targets, k_values=[1, 5, 10, 15, 20, 50])
    print_main_results(metrics, "Test Results")

    print("\nEvaluator module loaded successfully!")
