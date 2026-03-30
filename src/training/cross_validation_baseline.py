"""
10-Fold Cross-Validation Training for LSTM Baseline

This is the ONLY training entry point for baseline model.
Implements clean, unified cv10 training with internal validation split.

Paper-Aligned:
- User-level 10-fold cross-validation
- Each user's FULL sequence used in their assigned fold
- Internal validation split from training data for early stopping
- Final evaluation on held-out test fold
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.baseline_lstm import LSTMRec
from src.data.dataloader import get_dataloader
from src.evaluation.metrics import compute_all_metrics
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline 10-Fold CV Training')
    parser.add_argument('--config', type=str, default='configs/tafeng_baseline.yaml')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--num-folds', type=int, default=None,
                       help='Number of folds for cross-validation (default: 10 from config, use 1 for quick test)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode: 1 fold, 3 epochs, small dataset')
    return parser.parse_args()


def split_train_val(
    train_sequences: dict,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> tuple:
    """
    Split training sequences into train and internal validation sets.

    Strategy: User-level split (randomly sample users for validation)
    - Ensures no data leakage between train/val
    - Simple and stable for early stopping

    Args:
        train_sequences: training sequences for this fold
        val_ratio: fraction of users to use for validation
        random_seed: random seed for reproducibility

    Returns:
        train_sequences_split, val_sequences_split
    """
    np.random.seed(random_seed)

    user_ids = list(train_sequences.keys())
    np.random.shuffle(user_ids)

    num_val = int(len(user_ids) * val_ratio)
    val_user_ids = user_ids[:num_val]
    train_user_ids = user_ids[num_val:]

    train_split = {uid: train_sequences[uid] for uid in train_user_ids}
    val_split = {uid: train_sequences[uid] for uid in val_user_ids}

    logger.info(f"Internal validation split: {len(train_split)} train, {len(val_split)} val users")

    return train_split, val_split


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # **UNIFIED BATCH FORMAT**: items, seq_length, target
        items = batch['items'].to(device)
        seq_lengths = batch['seq_length'].to(device)
        targets = batch['target'].to(device)

        # Forward pass
        logits = model(items, seq_lengths)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    k_values: list
) -> dict:
    """Evaluate model"""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # **UNIFIED BATCH FORMAT**: items, seq_length, target
            items = batch['items'].to(device)
            seq_lengths = batch['seq_length'].to(device)
            targets = batch['target'].to(device)

            # Forward pass
            logits = model(items, seq_lengths)

            # Collect predictions and targets
            all_predictions.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(all_predictions, all_targets, k_values)

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_path: Path,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    # Save checkpoint
    checkpoint_path = save_path / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = save_path / 'best_model.pt'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")


def train_fold(
    fold_idx: int,
    train_sequences: dict,
    test_sequences: dict,
    config: dict,
    device: torch.device,
    output_dir: Path,
    quick_test: bool = False
) -> dict:
    """
    Train a single fold with internal validation split.

    Workflow:
    1. Split train_sequences into train + internal validation
    2. Train on train split, early stop on val split
    3. Load best model and evaluate on test split

    Args:
        fold_idx: fold index (0-9)
        train_sequences: training sequences for this fold
        test_sequences: test sequences for this fold
        config: configuration dict
        device: device to use
        output_dir: directory to save results
        quick_test: if True, use minimal epochs

    Returns:
        test_metrics: final test metrics
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Training Fold {fold_idx + 1}")
    logger.info(f"{'=' * 80}")

    # Load metadata
    processed_dir = Path(config['dataset']['processed_dir'])
    with open(processed_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    num_items = metadata['num_items']

    # **INTERNAL VALIDATION SPLIT**: Split train into train + val
    val_ratio = config['training'].get('val_ratio', 0.1)
    val_random_seed = config['training'].get('val_random_seed', 42)
    train_split, val_split = split_train_val(
        train_sequences,
        val_ratio=val_ratio,
        random_seed=val_random_seed + fold_idx  # Different seed per fold
    )

    # Create dataloaders
    max_seq_length = config['model']['max_seq_length']
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 0)

    train_loader = get_dataloader(
        sequences=train_split,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        split='train',
        num_workers=num_workers
    )

    val_loader = get_dataloader(
        sequences=val_split,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        split='val',
        num_workers=num_workers
    )

    test_loader = get_dataloader(
        sequences=test_sequences,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        split='test',
        num_workers=num_workers
    )

    # Create model
    model = LSTMRec(
        num_items=num_items,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    logger.info(f"Fold {fold_idx + 1} Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Training loop
    max_epochs = config['training']['max_epochs']
    patience = config['training']['early_stop_patience']
    metric_for_best = config['training']['metric_for_best']

    best_metric = 0.0
    no_improve_count = 0

    fold_output_dir = output_dir / f'fold_{fold_idx + 1}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        logger.info(f"Fold {fold_idx + 1} - Epoch {epoch + 1}/{max_epochs}: Train loss = {train_loss:.4f}")

        # Validate on internal validation set
        val_metrics = evaluate(model, val_loader, device, config['evaluation']['k_values'])
        current_metric = val_metrics[metric_for_best]

        logger.info(f"Fold {fold_idx + 1} - Epoch {epoch + 1}: {metric_for_best} = {current_metric:.4f}")

        # Check if best
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            no_improve_count = 0
            logger.info(f"Fold {fold_idx + 1} - New best {metric_for_best}: {best_metric:.4f}")
        else:
            no_improve_count += 1
            logger.info(f"Fold {fold_idx + 1} - No improvement for {no_improve_count} epochs")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch + 1, val_metrics,
            fold_output_dir, is_best
        )

        # Early stopping
        if no_improve_count >= patience:
            logger.info(f"Fold {fold_idx + 1} - Early stopping at epoch {epoch + 1}")
            break

    # Load best model and evaluate on test set
    logger.info(f"\nFold {fold_idx + 1} - Evaluating best model on test set...")
    best_checkpoint = torch.load(fold_output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, config['evaluation']['k_values'])

    # Log test results
    logger.info(f"\nFold {fold_idx + 1} Final Test Results:")
    for metric in ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']:
        if metric in test_metrics:
            logger.info(f"  {metric}: {test_metrics[metric]:.4f}")

    # Save test metrics
    metrics_path = fold_output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert to percentage for saving
        test_metrics_pct = {k: v * 100 for k, v in test_metrics.items()}
        json.dump(test_metrics_pct, f, indent=2)

    return test_metrics


def run_cross_validation(
    config_path: str,
    output_dir: str = None,
    num_folds: int = None,
    quick_test: bool = False
) -> dict:
    """
    Run 10-fold cross-validation for baseline model.

    This is the MAIN entry point for baseline training.

    Args:
        config_path: path to config file
        output_dir: directory to save results
        num_folds: number of folds (default: from config, use 1 for quick test)
        quick_test: if True, use minimal epochs (3 instead of 20)

    Returns:
        cv_results: dict with mean and std for each metric
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config.get('seed', 42))

    # Quick test overrides
    if quick_test:
        config['training']['max_epochs'] = 3
        config['training']['early_stop_patience'] = 2
        logger.info("Quick test mode: using 3 epochs")

    # Use provided num_folds or config default
    if num_folds is None:
        num_folds = config['dataset'].get('num_folds', 10)
    logger.info(f"Running {num_folds}-fold cross-validation")

    # Setup logging
    if output_dir is None:
        output_dir = Path(config['logging']['checkpoint_dir']).parent.parent / 'cross_validation'
    else:
        output_dir = Path(output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'cv.log'

    global logger
    logger = get_logger(__name__, str(log_file))

    logger.info("=" * 80)
    logger.info("Baseline 10-Fold Cross-Validation Training")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Output directory: {output_dir}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load CV splits (REQUIRED)
    processed_dir = Path(config['dataset']['processed_dir'])
    cv_dir = processed_dir / 'cv_splits'

    if not cv_dir.exists():
        raise FileNotFoundError(
            f"CV splits not found: {cv_dir}\n"
            f"Please run preprocessing first: bash scripts/run_preprocess_cv.sh"
        )

    if not (cv_dir / 'cv_metadata.pkl').exists():
        raise FileNotFoundError(f"CV metadata not found: {cv_dir / 'cv_metadata.pkl'}")

    # Load pre-computed CV splits
    logger.info(f"Loading pre-computed CV splits from {cv_dir}")
    with open(cv_dir / 'cv_metadata.pkl', 'rb') as f:
        cv_metadata = pickle.load(f)

    num_folds = cv_metadata['num_folds']
    logger.info(f"Found {num_folds}-fold CV splits")

    # Load all fold splits
    cv_splits = []
    for fold_idx in range(1, num_folds + 1):
        fold_dir = cv_dir / f'fold_{fold_idx}'
        with open(fold_dir / 'train_sequences.pkl', 'rb') as f:
            train_sequences = pickle.load(f)
        with open(fold_dir / 'test_sequences.pkl', 'rb') as f:
            test_sequences = pickle.load(f)
        cv_splits.append({
            'train': train_sequences,
            'test': test_sequences
        })
        logger.info(f"  Fold {fold_idx}: Train={len(train_sequences)}, Test={len(test_sequences)}")

    # Train and evaluate each fold
    all_fold_metrics = []

    for fold_idx in range(num_folds):
        train_sequences = cv_splits[fold_idx]['train']
        test_sequences = cv_splits[fold_idx]['test']

        fold_metrics = train_fold(
            fold_idx,
            train_sequences,
            test_sequences,
            config,
            device,
            output_dir,
            quick_test=quick_test
        )

        all_fold_metrics.append(fold_metrics)

    # Aggregate results
    logger.info(f"\n{'=' * 80}")
    logger.info("10-Fold CV Aggregate Results")
    logger.info(f"{'=' * 80}")

    metric_names = all_fold_metrics[0].keys()
    cv_results = {}

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_fold_metrics]
        cv_results[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1)
        }

    # Print main metrics
    main_metrics = ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']
    for metric in main_metrics:
        if metric in cv_results:
            mean_val = cv_results[metric]['mean']
            std_val = cv_results[metric]['std']
            logger.info(f"{metric}: {mean_val * 100:.2f} ± {std_val * 100:.2f}%")

    logger.info("=" * 80)

    # Save aggregated results
    results_path = output_dir / 'cv_results.json'
    with open(results_path, 'w') as f:
        # Convert to percentage
        cv_results_pct = {}
        for metric_name, stats in cv_results.items():
            cv_results_pct[metric_name] = {
                'mean': stats['mean'] * 100,
                'std': stats['std'] * 100
            }
        json.dump(cv_results_pct, f, indent=2)

    logger.info(f"\nSaved CV results to {results_path}")

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


if __name__ == '__main__':
    args = parse_args()
    run_cross_validation(
        args.config,
        args.output_dir,
        num_folds=args.num_folds,
        quick_test=args.quick_test
    )
