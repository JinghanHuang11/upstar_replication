"""
Training script for LSTM baseline
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.baseline_lstm import LSTMRec
from src.data.dataloader import get_dataloader
from src.evaluation.metrics import compute_all_metrics, print_metrics
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()


def load_data(config: dict):
    """Load processed data and create dataloaders"""
    processed_dir = Path(config['dataset']['processed_dir'])

    # Load metadata
    with open(processed_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    num_users = metadata['num_users']
    num_items = metadata['num_items']

    logger.info(f"Dataset: {num_users} users, {num_items} items")

    # Load sequences
    with open(processed_dir / 'train_sequences.pkl', 'rb') as f:
        train_sequences = pickle.load(f)

    with open(processed_dir / 'val_sequences.pkl', 'rb') as f:
        val_sequences = pickle.load(f)

    with open(processed_dir / 'test_sequences.pkl', 'rb') as f:
        test_sequences = pickle.load(f)

    logger.info(f"Loaded {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test sequences")

    # Create dataloaders
    max_seq_length = config['model']['max_seq_length']
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 0)

    train_loader = get_dataloader(
        sequences=train_sequences,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        split='train',
        num_workers=num_workers
    )

    val_loader = get_dataloader(
        sequences=val_sequences,
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

    return train_loader, val_loader, test_loader, num_items


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
        # Move to device
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
    k_values: list = [5, 10, 20]
) -> dict:
    """Evaluate model"""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
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
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save best model
    if is_best:
        best_path = save_path / 'best_model.pt'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")


def main(args=None):
    if args is None:
        args = parse_args()
    if not hasattr(args, 'resume'):
        args.resume = None

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
    logger.info("Training LSTM Baseline")
    logger.info("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, num_items = load_data(config)

    # Create model
    model = LSTMRec(
        num_items=num_items,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Resume from checkpoint
    start_epoch = 0
    best_metric = 0.0

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['metrics'].get(config['training']['metric_for_best'], 0.0)

    # Training loop
    max_epochs = config['training']['max_epochs']
    patience = config['training']['early_stop_patience']
    no_improve_count = 0

    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, max_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{max_epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        logger.info(f"Train loss: {train_loss:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device, config['evaluation']['k_values'])
        print_metrics(val_metrics, f"Validation Metrics (Epoch {epoch + 1})")

        # Check if best
        current_metric = val_metrics[config['training']['metric_for_best']]
        is_best = current_metric > best_metric

        if is_best:
            best_metric = current_metric
            no_improve_count = 0
            logger.info(f"New best {config['training']['metric_for_best']}: {best_metric:.4f}")
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epochs")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch + 1, val_metrics,
            checkpoint_dir, is_best
        )

        # Early stopping
        if no_improve_count >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Test
    logger.info(f"\n{'='*60}")
    logger.info("Testing on best model")
    logger.info(f"{'='*60}")

    # Load best model
    best_checkpoint = torch.load(checkpoint_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, config['evaluation']['k_values'])
    print_metrics(test_metrics, "Test Metrics (Best Model)")

    # Save test results
    results_path = Path(config['logging']['prediction_dir']) / 'test_results.pkl'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(test_metrics, f)

    logger.info(f"Saved test results to {results_path}")
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)

    return test_metrics


if __name__ == '__main__':
    main()