"""
10-Fold Cross-Validation Training for UPSTAR Model

This is the ONLY training entry point for UPSTAR model.
Implements clean, unified cv10 training with internal validation split.

Paper-Aligned:
- User-level 10-fold cross-validation
- Each user's FULL sequence used in their assigned fold
- Internal validation split from training data for early stopping
- Final evaluation on held-out test fold
- Staged curriculum (engineering, not paper modification)
- Joint loss: L_total = L_global + λ*L_S&E&O + L_orth + L_distill

Key Improvements:
- Early stopping on INTERNAL validation set (NOT test set)
- Explicit best model reload before final test evaluation
- Clear separation: train_split / val_split / test_split
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
from typing import Dict, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.upstar import UPSTARModel
from src.data.upstar_dataset import get_upstar_dataloader
from src.training.losses import UPSTARLoss, create_loss_from_config
from src.evaluation.metrics import compute_all_metrics, compute_mean_std, print_mean_std
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='UPSTAR 10-Fold CV Training')
    parser.add_argument('--config', type=str, default='configs/tafeng_upstar.yaml')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--stage', type=int, default=None,
                       help='Train specific stage only (1-4, default: all stages)')
    parser.add_argument('--num-folds', type=int, default=None,
                       help='Number of folds for cross-validation (default: 10 from config, use 1 for quick test)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode: 1 fold, 2 epochs per stage, small dataset')
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


def train_stage(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: UPSTARLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    stage_idx: int,
    max_epochs: int,
    patience: int,
    metric_for_best: str,
    fold_output_dir: Path
) -> tuple:
    """
    Train a single stage with early stopping on validation set.

    Args:
        model: UPSTAR model
        train_loader: training dataloader
        val_loader: validation dataloader (INTERNAL, not test)
        loss_fn: UPSTAR loss function
        optimizer: optimizer
        device: device
        stage_idx: stage index (1-4)
        max_epochs: max epochs per stage
        patience: early stopping patience
        metric_for_best: metric to monitor for early stopping
        fold_output_dir: directory to save checkpoints

    Returns:
        best_metric: best validation metric achieved
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training Stage {stage_idx}")
    logger.info(f"{'=' * 60}")

    best_metric = 0.0
    no_improve_count = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Stage {stage_idx} - Epoch {epoch + 1}"):
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

        # Validate on INTERNAL validation set (NOT test set)
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
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
        val_metrics = compute_all_metrics(
            all_predictions, all_targets,
            k_values=[5, 10, 20]
        )
        current_metric = val_metrics[metric_for_best]

        # Log
        logger.info(f"Stage {stage_idx} - Epoch {epoch + 1}/{max_epochs}: "
                   f"Loss={train_loss:.4f}, "
                   f"Val {metric_for_best}={current_metric:.4f}")

        # Check if best
        if current_metric > best_metric:
            best_metric = current_metric
            no_improve_count = 0

            # Save best model for this stage
            best_path = fold_output_dir / f'best_model_stage{stage_idx}.pt'
            torch.save({
                'stage': stage_idx,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, best_path)
            logger.info(f"Stage {stage_idx} - New best {metric_for_best}: {best_metric:.4f}")
        else:
            no_improve_count += 1
            logger.info(f"Stage {stage_idx} - No improvement for {no_improve_count} epochs")

        # Early stopping
        if no_improve_count >= patience:
            logger.info(f"Stage {stage_idx} - Early stopping at epoch {epoch + 1}")
            break

    # Load best model for this stage
    best_path = fold_output_dir / f'best_model_stage{stage_idx}.pt'
    best_checkpoint = torch.load(best_path, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    logger.info(f"Stage {stage_idx} - Loaded best model from epoch {best_checkpoint['epoch']}")

    return best_metric


def train_fold(
    fold_idx: int,
    train_sequences: dict,
    test_sequences: dict,
    motivation_labels: np.ndarray,
    config: dict,
    device: torch.device,
    output_dir: Path,
    train_single_stage: int = None
) -> dict:
    """
    Train a single fold with internal validation split.

    Workflow:
    1. Split train_sequences into train + internal validation
    2. Train 4 stages (or single stage if specified), early stopping on val
    3. Load best model from stage 4 and evaluate on test set

    Args:
        fold_idx: fold index (0-9)
        train_sequences: training sequences for this fold
        test_sequences: test sequences for this fold
        motivation_labels: STB motivation labels
        config: configuration dict
        device: device to use
        output_dir: directory to save results
        train_single_stage: if specified, only train this stage (1-4)

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

    train_loader = get_upstar_dataloader(
        sequences=train_split,
        motivation_labels=motivation_labels,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        split='train',
        num_workers=num_workers
    )

    val_loader = get_upstar_dataloader(
        sequences=val_split,
        motivation_labels=motivation_labels,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        split='val',
        num_workers=num_workers
    )

    test_loader = get_upstar_dataloader(
        sequences=test_sequences,
        motivation_labels=motivation_labels,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        split='test',
        num_workers=num_workers
    )

    # Create model
    model = UPSTARModel(
        num_items=num_items,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        use_gate=config['model']['use_gate']
    ).to(device)

    logger.info(f"Fold {fold_idx + 1} Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create loss function
    loss_fn = create_loss_from_config(config)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Training parameters
    max_epochs = config['training']['max_epochs_per_stage']
    patience = config['training']['early_stop_patience']
    metric_for_best = config['training']['metric_for_best']

    fold_output_dir = output_dir / f'fold_{fold_idx + 1}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    # Train stages
    if train_single_stage is not None:
        # Train single stage
        logger.info(f"Training single stage: {train_single_stage}")
        stages_to_train = [train_single_stage]
    else:
        # Train all 4 stages
        stages_to_train = [1, 2, 3, 4]

    for stage_idx in stages_to_train:
        # Update loss function for this stage
        loss_fn.set_stage(stage_idx)

        # Train this stage
        train_stage(
            model, train_loader, val_loader, loss_fn, optimizer,
            device, stage_idx, max_epochs, patience, metric_for_best,
            fold_output_dir
        )

    # Load best model from final stage and evaluate on TEST set
    logger.info(f"\nFold {fold_idx + 1} - Evaluating best model on TEST set...")

    final_stage = stages_to_train[-1]
    best_path = fold_output_dir / f'best_model_stage{final_stage}.pt'
    best_checkpoint = torch.load(best_path, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Evaluate on TEST set (held-out, not used for early stopping)
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

    # Compute test metrics
    test_metrics = compute_all_metrics(
        all_predictions, all_targets,
        k_values=config['evaluation']['k_values']
    )

    # Log test results
    logger.info(f"\nFold {fold_idx + 1} Final TEST Results:")
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
    train_single_stage: int = None,
    num_folds: int = None,
    quick_test: bool = False
) -> dict:
    """
    Run 10-fold cross-validation for UPSTAR model.

    This is the MAIN entry point for UPSTAR training.

    Args:
        config_path: path to config file
        output_dir: directory to save results
        train_single_stage: if specified, only train this stage (1-4)
        num_folds: number of folds (default: from config, use 1 for quick test)
        quick_test: if True, use minimal epochs (2 per stage instead of 10)

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
        config['training']['max_epochs_per_stage'] = 2
        config['training']['early_stop_patience'] = 2
        logger.info("Quick test mode: using 2 epochs per stage")

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
    logger.info("UPSTAR 10-Fold Cross-Validation Training")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    if train_single_stage:
        logger.info(f"Training single stage: {train_single_stage}")
    else:
        logger.info("Training all 4 stages")

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

    # Load STB motivation labels
    stb_scores_path = Path(config['stb']['motivation_labels_path'])
    if not stb_scores_path.exists():
        raise FileNotFoundError(f"Motivation labels not found: {stb_scores_path}")

    motivation_labels = np.load(stb_scores_path)
    logger.info(f"Loaded motivation labels: {motivation_labels.shape}")

    # Train and evaluate each fold
    all_fold_metrics = []

    for fold_idx in range(num_folds):
        train_sequences = cv_splits[fold_idx]['train']
        test_sequences = cv_splits[fold_idx]['test']

        fold_metrics = train_fold(
            fold_idx,
            train_sequences,
            test_sequences,
            motivation_labels,
            config,
            device,
            output_dir,
            train_single_stage
        )

        all_fold_metrics.append(fold_metrics)

    # Aggregate results
    logger.info(f"\n{'=' * 80}")
    logger.info("10-Fold CV Aggregate Results")
    logger.info(f"{'=' * 80}")

    cv_results = compute_mean_std(all_fold_metrics)

    # Print results
    print_mean_std(cv_results, f"{num_folds}-Fold Cross-Validation Results")

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
        args.stage,
        args.num_folds,
        args.quick_test
    )
