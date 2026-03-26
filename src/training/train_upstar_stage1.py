"""
Training script for UPSTAR Stage 1

Three-path model with global fusion.
Simple training without complex losses.
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

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.upstar import UPSTARMoodel
from src.data.upstar_dataset import get_upstar_dataloader
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/upstar_stage1.yaml')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def load_data(config: dict):
    """Load data and motivation labels"""
    processed_dir = Path(config['dataset']['processed_dir'])

    # Load metadata
    with open(processed_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    num_items = metadata['num_items']

    # Load sequences
    with open(processed_dir / 'train_sequences.pkl', 'rb') as f:
        train_sequences = pickle.load(f)

    with open(processed_dir / 'val_sequences.pkl', 'rb') as f:
        val_sequences = pickle.load(f)

    with open(processed_dir / 'test_sequences.pkl', 'rb') as f:
        test_sequences = pickle.load(f)

    # Load STB results
    stb_scores_path = Path(config['stb']['stb_scores_path'])
    motivation_labels_path = Path(config['stb']['motivation_labels_path'])

    if not stb_scores_path.exists():
        raise FileNotFoundError(f"STB scores not found: {stb_scores_path}")

    if not motivation_labels_path.exists():
        raise FileNotFoundError(f"Motivation labels not found: {motivation_labels_path}")

    motivation_labels = np.load(motivation_labels_path)

    logger.info(f"Loaded data: {num_items} items")
    logger.info(f"  Train: {len(train_sequences)} users")
    logger.info(f"  Val: {len(val_sequences)} users")
    logger.info(f"  Test: {len(test_sequences)} users")
    logger.info(f"  Motivation labels: {motivation_labels.shape}")

    return train_sequences, val_sequences, test_sequences, num_items, motivation_labels


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10
) -> float:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    num_batches = 0

    gate_weights_list = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        seq_stable = batch['seq_stable'].to(device)
        len_stable = batch['len_stable'].to(device)
        seq_exploratory = batch['seq_exploratory'].to(device)
        len_exploratory = batch['len_exploratory'].to(device)
        seq_entire = batch['seq_entire'].to(device)
        len_entire = batch['len_entire'].to(device)
        targets = batch['target'].to(device)

        # Forward pass
        optimizer.zero_grad()

        output = model(
            seq_stable, len_stable,
            seq_exploratory, len_exploratory,
            seq_entire, len_entire
        )

        # Use global loss
        logits = output['y_hat_global']
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Track gate weights
        gate_weights = output['gate_weights']  # [B, 3]
        gate_weights_list.append(gate_weights.mean(dim=0).detach().cpu())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log gate weights
        if (batch_idx + 1) % log_interval == 0:
            avg_gate = gate_weights.mean(dim=0)
            logger.info(f"  Batch {batch_idx+1}: Gate weights (avg) - "
                       f"Stab={avg_gate[0]:.3f}, Expl={avg_gate[1]:.3f}, Other={avg_gate[2]:.3f}")

    avg_loss = total_loss / num_batches

    # Epoch gate statistics
    all_gate_weights = torch.stack(gate_weights_list, dim=0)  # [num_batches, 3]
    epoch_gate_avg = all_gate_weights.mean(dim=0)

    logger.info(f"Epoch {epoch} average gate weights: "
               f"Stab={epoch_gate_avg[0]:.3f}, "
               f"Expl={epoch_gate_avg[1]:.3f}, "
               f"Other={epoch_gate_avg[2]:.3f}")

    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_items: int,
    k_values: list = [5, 10, 20]
) -> dict:
    """Evaluate model"""
    from src.evaluation.metrics import compute_all_metrics

    model.eval()

    all_predictions = []
    all_targets = []

    gate_weights_list = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        seq_stable = batch['seq_stable'].to(device)
        len_stable = batch['len_stable'].to(device)
        seq_exploratory = batch['seq_exploratory'].to(device)
        len_exploratory = batch['len_exploratory'].to(device)
        seq_entire = batch['seq_entire'].to(device)
        len_entire = batch['len_entire'].to(device)
        targets = batch['target'].to(device)

        # Forward pass
        output = model(
            seq_stable, len_stable,
            seq_exploratory, len_exploratory,
            seq_entire, len_entire
        )

        # Collect predictions and targets
        all_predictions.append(output['y_hat_global'].cpu())
        all_targets.append(targets.cpu())

        # Track gate weights
        gate_weights = output['gate_weights']
        gate_weights_list.append(gate_weights.mean(dim=0).cpu())

    # Concatenate
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(all_predictions, all_targets, k_values)

    # Gate statistics
    all_gate_weights = torch.stack(gate_weights_list, dim=0)
    gate_avg = all_gate_weights.mean(dim=0)

    metrics['gate_stab'] = gate_avg[0].item()
    metrics['gate_expl'] = gate_avg[1].item()
    metrics['gate_other'] = gate_avg[2].item()

    return metrics


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config['seed'])

    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'train.log'

    global logger
    logger = get_logger(__name__, str(log_file))

    logger.info("=" * 60)
    logger.info("UPSTAR Stage 1 Training")
    logger.info("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    train_sequences, val_sequences, test_sequences, num_items, motivation_labels = load_data(config)

    # Update config with num_items
    config['model']['num_items'] = num_items

    # Create dataloaders
    max_seq_length = config['model'].get('max_seq_length', 50)
    batch_size = config['training']['batch_size']

    train_loader = get_upstar_dataloader(
        sequences=train_sequences,
        motivation_labels=motivation_labels,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        split='train',
        num_workers=0
    )

    val_loader = get_upstar_dataloader(
        sequences=val_sequences,
        motivation_labels=motivation_labels,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        split='val',
        num_workers=0
    )

    test_loader = get_upstar_dataloader(
        sequences=test_sequences,
        motivation_labels=motivation_labels,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        split='test',
        num_workers=0
    )

    # Create model
    model = UPSTARMoodel(
        num_items=num_items,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        use_gate=config['model']['use_gate']
    ).to(device)

    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

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
    best_metric = 0.0
    no_improve_count = 0

    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{max_epochs}")
        logger.info("=" * 60)

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1,
            log_interval=config['training']['log_gate_interval']
        )
        logger.info(f"Train loss: {train_loss:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device, num_items, config['evaluation']['k_values'])

        logger.info(f"Validation NDCG@10: {val_metrics['NDCG@10']:.4f}")
        logger.info(f"Validation Gate Weights: "
                   f"Stab={val_metrics['gate_stab']:.3f}, "
                   f"Expl={val_metrics['gate_expl']:.3f}, "
                   f"Other={val_metrics['gate_other']:.3f}")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Check if best
        current_metric = val_metrics[config['training'].get('metric_for_best', 'NDCG@10')]
        if current_metric > best_metric:
            best_metric = current_metric
            no_improve_count = 0

            # Save best model
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, best_path)
            logger.info(f"New best NDCG@10: {best_metric:.4f}")
            logger.info(f"Saved best model to {best_path}")
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epochs")

        if no_improve_count >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Test
    logger.info("\n" + "=" * 60)
    logger.info("Testing on best model")
    logger.info("=" * 60)

    best_checkpoint = torch.load(checkpoint_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, num_items, config['evaluation']['k_values'])

    logger.info("\nTest Results:")
    from src.evaluation.metrics import print_metrics
    print_metrics(test_metrics, "Test Metrics")

    logger.info(f"\nTest Gate Weights: "
               f"Stab={test_metrics['gate_stab']:.3f}, "
               f"Expl={test_metrics['gate_expl']:.3f}, "
               f"Other={test_metrics['gate_other']:.3f}")

    # Save test results
    results_path = Path(config['logging']['prediction_dir']) / 'test_results.pkl'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(test_metrics, f)

    logger.info(f"\nSaved test results to {results_path}")
    logger.info("=" * 60)
    logger.info("UPSTAR Stage 1 training complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
