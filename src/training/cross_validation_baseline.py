"""
Cross-Validation for Baseline Model

User-level 10-fold cross-validation for baseline LSTM.
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
import sys
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.baseline_lstm import BaselineLSTM
from src.data.dataset import get_dataloader
from src.evaluation.metrics import compute_all_metrics
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/tafeng_baseline.yaml')
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args()


def create_folds(all_sequences, num_folds=10, random_seed=42):
    """Create 10 folds of users"""
    set_seed(random_seed)

    # Get all user indices
    user_indices = list(all_sequences.keys())

    # Shuffle
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

    logger.info(f"Created {num_folds} folds:")
    for i, fold in enumerate(folds):
        logger.info(f"  Fold {i + 1}: {len(fold)} users")

    return folds


def get_fold_data(fold_idx, folds, all_sequences):
    """Get train and test data for a fold"""
    # Test users
    test_users = folds[fold_idx]

    # Train users (all other folds)
    train_users = []
    for i, fold in enumerate(folds):
        if i != fold_idx:
            train_users.extend(fold)

    # Create sequence dicts
    train_sequences = {uid: all_sequences[uid] for uid in train_users}
    test_sequences = {uid: all_sequences[uid] for uid in test_users}

    logger.info(f"Fold {fold_idx + 1}: Train={len(train_sequences)}, Test={len(test_sequences)}")

    return train_sequences, test_sequences


def train_fold(
    fold_idx,
    train_sequences,
    val_sequences,
    test_sequences,
    config,
    device,
    output_dir
):
    """Train a single fold"""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Training Fold {fold_idx + 1}")
    logger.info(f"{'=' * 80}")

    # Load metadata
    processed_dir = Path(config['dataset']['processed_dir'])
    with open(processed_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    num_items = metadata['num_items']

    # Create dataloaders
    max_seq_length = config['model'].get('max_seq_length', 50)
    batch_size = config['training']['batch_size']

    train_loader = get_dataloader(
        config, 'train',
        batch_size=batch_size,
        num_workers=0
    )

    val_loader = get_dataloader(
        config, 'val',
        batch_size=batch_size,
        num_workers=0
    )

    test_loader = get_dataloader(
        config, 'test',
        batch_size=batch_size,
        num_workers=0
    )

    # Create model
    model = BaselineLSTM(
        vocab_size=num_items,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
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
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_items = batch['input_items'].to(device)
            seq_length = batch['seq_length'].to(device)
            target_item = batch['target_item'].to(device)

            optimizer.zero_grad()

            # Forward
            logits = model(input_items, seq_length)

            # Loss
            loss = criterion(logits, target_item)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches

        # Validate
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_items = batch['input_items'].to(device)
                seq_length = batch['seq_length'].to(device)
                target_item = batch['target_item'].to(device)

                logits = model(input_items, seq_length)
                all_predictions.append(logits.cpu())
                all_targets.append(target_item.cpu())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        val_metrics = compute_all_metrics(
            all_predictions, all_targets,
            k_values=[5, 10, 20]
        )

        current_metric = val_metrics[metric_for_best]

        # Log
        logger.info(f"Epoch {epoch + 1}/{max_epochs}: "
                   f"Loss={train_loss:.4f}, "
                   f"NDCG@10={val_metrics['NDCG@10']:.4f}")

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
                'metrics': val_metrics
            }, best_path)
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model and test
    best_checkpoint = torch.load(fold_output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_items = batch['input_items'].to(device)
            seq_length = batch['seq_length'].to(device)
            target_item = batch['target_item'].to(device)

            logits = model(input_items, seq_length)
            all_predictions.append(logits.cpu())
            all_targets.append(target_item.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute test metrics
    test_metrics = compute_all_metrics(
        all_predictions, all_targets,
        k_values=[1, 5, 10, 15, 20, 50]
    )

    # Save test metrics
    metrics_path = fold_output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({k: v * 100 for k, v in test_metrics.items()}, f, indent=2)

    logger.info(f"Fold {fold_idx + 1} complete: NDCG@10={test_metrics['NDCG@10']:.4f}")

    return test_metrics


def run_cross_validation(
    config_path: str,
    num_folds: int = 10,
    output_dir: str = None
):
    """Run 10-fold cross-validation"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config.get('seed', 42))

    # Setup logging
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'cv.log'

    global logger
    logger = get_logger(__name__, str(log_file))

    logger.info("=" * 80)
    logger.info("Baseline 10-Fold Cross-Validation")
    logger.info("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    processed_dir = Path(config['dataset']['processed_dir'])
    with open(processed_dir / 'train_sequences.pkl', 'rb') as f:
        all_sequences = pickle.load(f)

    logger.info(f"Loaded {len(all_sequences)} users for cross-validation")

    # Create folds
    folds = create_folds(all_sequences, num_folds=num_folds, random_seed=42)

    # Output directory
    if output_dir is None:
        output_dir = Path(config['logging']['checkpoint_dir']).parent.parent / 'cross_validation'
    else:
        output_dir = Path(output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train and evaluate each fold
    all_fold_metrics = []

    for fold_idx in range(num_folds):
        # For cross-validation, we use the full dataset
        # Split into train/val/test within each fold
        fold_users = folds[fold_idx]
        other_users = []
        for i, fold in enumerate(folds):
            if i != fold_idx:
                other_users.extend(fold)

        # Create sequences
        train_sequences = {uid: all_sequences[uid] for uid in other_users}
        test_sequences = {uid: all_sequences[uid] for uid in fold_users}

        # Train and test
        fold_metrics = train_fold(
            fold_idx,
            train_sequences,
            {},  # No val set, use early stopping on train
            test_sequences,
            config,
            device,
            output_dir
        )

        all_fold_metrics.append(fold_metrics)

    # Aggregate results
    metric_names = all_fold_metrics[0].keys()
    cv_results = {}

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_fold_metrics]
        cv_results[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1)
        }

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Cross-Validation Results")
    logger.info("=" * 80)

    main_metrics = ['Precision@5', 'Precision@20', 'NDCG@5', 'NDCG@20', 'MRR@5', 'MRR@20']

    for metric in main_metrics:
        if metric in cv_results:
            mean_val = cv_results[metric]['mean']
            std_val = cv_results[metric]['std']
            logger.info(f"{metric}: {mean_val * 100:.2f} ± {std_val * 100:.2f}%")

    logger.info("=" * 80)

    # Save results
    import json
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

    logger.info(f"\nSaved results to {results_path}")

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
    run_cross_validation(args.config, args.num_folds, args.output_dir)
