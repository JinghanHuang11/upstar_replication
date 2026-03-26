"""
Training script for UPSTAR with Staged Loss Training (Phase 5)

Implements four-stage training:
1. Stage 1: L_global only
2. Stage 2: L_global + L_branch
3. Stage 3: L_global + L_branch + L_orth
4. Stage 4: All losses (including L_distill)

Each stage:
- Trains independently with early stopping
- Saves separate checkpoint
- Reports validation metrics
- Can resume from previous stage
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

from src.models.upstar import UPSTARModel
from src.data.upstar_dataset import get_upstar_dataloader
from src.training.losses import UPSTARLoss, create_loss_from_config
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/tafeng_upstar.yaml')
    parser.add_argument('--stage', type=int, default=None,
                       help='Train specific stage only (1-4)')
    parser.add_argument('--resume_stage', type=int, default=None,
                       help='Resume from stage checkpoint')
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
    loss_fn: UPSTARLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    stage: int,
    log_interval: int = 10
) -> dict:
    """Train for one epoch"""
    model.train()

    total_losses = {
        'total': 0.0,
        'global': 0.0,
        'branch': 0.0,
        'ortho': 0.0,
        'distill': 0.0
    }
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

        # Get motivation labels for distillation
        motivation_labels_batch = batch['motivation_labels'].to(device)

        # Forward pass
        optimizer.zero_grad()

        output = model(
            seq_stable, len_stable,
            seq_exploratory, len_exploratory,
            seq_entire, len_entire
        )

        # Prepare predictions for loss
        predictions = {
            'y_hat_global': output['y_hat_global'],
            'y_hat_stab': output['y_hat_stab'],
            'y_hat_expl': output['y_hat_expl'],
            'y_hat_other': output['y_hat_other'],
            'z_stab': output['z_stab'],
            'z_expl': output['z_expl'],
            'z_other': output['z_other']
        }

        # Compute loss
        loss_output = loss_fn(predictions, targets, motivation_labels_batch)
        loss = loss_output['total']

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Track losses
        total_losses['total'] += loss.item()
        total_losses['global'] += loss_output['global'].item()
        total_losses['branch'] += loss_output['branch']['total_branch'].item()
        total_losses['ortho'] += loss_output['ortho'].item()
        total_losses['distill'] += loss_output['distill']['loss_distill'].item()
        num_batches += 1

        # Track gate weights
        gate_weights = output['gate_weights']  # [B, 3]
        gate_weights_list.append(gate_weights.mean(dim=0).detach().cpu())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log detailed losses
        if (batch_idx + 1) % log_interval == 0:
            avg_gate = gate_weights.mean(dim=0)
            logger.info(f"  Batch {batch_idx+1}:")
            logger.info(f"    Losses - Total={loss.item():.4f}, "
                       f"Global={loss_output['global'].item():.4f}, "
                       f"Branch={loss_output['branch']['total_branch'].item():.4f}, "
                       f"Ortho={loss_output['ortho'].item():.4f}, "
                       f"Distill={loss_output['distill']['loss_distill'].item():.4f}")
            logger.info(f"    Gate (avg) - Stab={avg_gate[0]:.3f}, "
                       f"Expl={avg_gate[1]:.3f}, Other={avg_gate[2]:.3f}")
            if loss_output['distill']['s_teaches_e'] > 0 or loss_output['distill']['e_teaches_s'] > 0:
                logger.info(f"    Distill - S→E: {loss_output['distill']['s_teaches_e']:.0f}, "
                           f"E→S: {loss_output['distill']['e_teaches_s']:.0f}")

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    # Epoch gate statistics
    all_gate_weights = torch.stack(gate_weights_list, dim=0)  # [num_batches, 3]
    epoch_gate_avg = all_gate_weights.mean(dim=0)

    logger.info(f"Epoch {epoch} average losses:")
    logger.info(f"  Total:   {avg_losses['total']:.4f}")
    logger.info(f"  Global:  {avg_losses['global']:.4f}")
    logger.info(f"  Branch:  {avg_losses['branch']:.4f}")
    logger.info(f"  Ortho:   {avg_losses['ortho']:.4f}")
    logger.info(f"  Distill: {avg_losses['distill']:.4f}")
    logger.info(f"Epoch {epoch} average gate weights: "
               f"Stab={epoch_gate_avg[0]:.3f}, "
               f"Expl={epoch_gate_avg[1]:.3f}, "
               f"Other={epoch_gate_avg[2]:.3f}")

    return avg_losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    loss_fn: UPSTARLoss,
    device: torch.device,
    num_items: int,
    k_values: list = [5, 10, 20]
) -> dict:
    """Evaluate model"""
    from src.evaluation.metrics import compute_all_metrics

    model.eval()

    all_predictions = []
    all_targets = []
    total_losses = {
        'total': 0.0,
        'global': 0.0,
        'branch': 0.0,
        'ortho': 0.0,
        'distill': 0.0
    }
    num_batches = 0

    gate_weights_list = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        seq_stable = batch['seq_stable'].to(device)
        len_stable = batch['len_stable'].to(device)
        seq_exploratory = batch['seq_exploratory'].to(device)
        len_exploratory = batch['len_exploratory'].to(device)
        seq_entire = batch['seq_entire'].to(device)
        len_entire = batch['len_entire'].to(device)
        targets = batch['target'].to(device)
        motivation_labels_batch = batch['motivation_labels'].to(device)

        # Forward pass
        output = model(
            seq_stable, len_stable,
            seq_exploratory, len_exploratory,
            seq_entire, len_entire
        )

        # Compute loss
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

        # Collect predictions and targets
        all_predictions.append(output['y_hat_global'].cpu())
        all_targets.append(targets.cpu())

        # Track losses
        total_losses['total'] += loss_output['total'].item()
        total_losses['global'] += loss_output['global'].item()
        total_losses['branch'] += loss_output['branch']['total_branch'].item()
        total_losses['ortho'] += loss_output['ortho'].item()
        total_losses['distill'] += loss_output['distill']['loss_distill'].item()
        num_batches += 1

        # Track gate weights
        gate_weights = output['gate_weights']
        gate_weights_list.append(gate_weights.mean(dim=0).cpu())

    # Concatenate
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(all_predictions, all_targets, k_values)

    # Average losses
    avg_losses = {f'loss_{k}': v / num_batches for k, v in total_losses.items()}
    metrics.update(avg_losses)

    # Gate statistics
    all_gate_weights = torch.stack(gate_weights_list, dim=0)
    gate_avg = all_gate_weights.mean(dim=0)

    metrics['gate_stab'] = gate_avg[0].item()
    metrics['gate_expl'] = gate_avg[1].item()
    metrics['gate_other'] = gate_avg[2].item()

    return metrics


def train_stage(
    stage_num: int,
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: UPSTARLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict,
    checkpoint_dir: Path
) -> dict:
    """Train a single stage"""

    stage_config = config['training'][f'stage{stage_num}']
    max_epochs = config['training']['max_epochs_per_stage']
    patience = config['training']['early_stop_patience']
    metric_for_best = config['training']['metric_for_best']

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"STAGE {stage_num}: {get_stage_description(stage_num)}")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  use_global_loss:           {stage_config['use_global_loss']}")
    logger.info(f"  use_branch_loss:           {stage_config['use_branch_loss']}")
    logger.info(f"  use_orthogonality_loss:    {stage_config['use_orthogonality_loss']}")
    logger.info(f"  use_distillation_loss:     {stage_config['use_distillation_loss']}")
    logger.info("")

    # Update loss function for this stage
    loss_fn.use_global_loss = stage_config['use_global_loss']
    loss_fn.use_branch_loss = stage_config['use_branch_loss']
    loss_fn.use_orthogonality_loss = stage_config['use_orthogonality_loss']
    loss_fn.use_distillation_loss = stage_config['use_distillation_loss']

    best_metric = 0.0
    no_improve_count = 0
    stage_checkpoint_dir = checkpoint_dir / f'stage{stage_num}'
    stage_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{max_epochs}")
        logger.info("-" * 60)

        # Train
        train_losses = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch + 1, stage_num,
            log_interval=config['training']['log_gate_interval']
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, loss_fn, device,
            config['model']['num_items'],
            config['evaluation']['k_values']
        )

        logger.info(f"\nValidation Results:")
        logger.info(f"  NDCG@10:  {val_metrics['NDCG@10']:.4f}")
        logger.info(f"  Recall@10: {val_metrics['Recall@10']:.4f}")
        logger.info(f"  Loss - Total: {val_metrics['loss_total']:.4f}, "
                   f"Global: {val_metrics['loss_global']:.4f}, "
                   f"Branch: {val_metrics['loss_branch']:.4f}, "
                   f"Ortho: {val_metrics['loss_ortho']:.4f}, "
                   f"Distill: {val_metrics['loss_distill']:.4f}")
        logger.info(f"  Gate - Stab: {val_metrics['gate_stab']:.3f}, "
                   f"Expl: {val_metrics['gate_expl']:.3f}, "
                   f"Other: {val_metrics['gate_other']:.3f}")

        # Save checkpoint
        checkpoint_path = stage_checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'stage': stage_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
            'stage_config': stage_config
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Check if best
        current_metric = val_metrics[metric_for_best]
        if current_metric > best_metric:
            best_metric = current_metric
            no_improve_count = 0

            # Save best model for this stage
            best_path = stage_checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch + 1,
                'stage': stage_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'stage_config': stage_config
            }, best_path)
            logger.info(f"New best {metric_for_best}: {best_metric:.4f}")
            logger.info(f"Saved best model to {best_path}")
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epochs")

        if no_improve_count >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model for this stage
    best_checkpoint = torch.load(stage_checkpoint_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    logger.info(f"\nStage {stage_num} complete!")
    logger.info(f"Best {metric_for_best}: {best_metric:.4f}")

    return best_checkpoint['metrics']


def get_stage_description(stage_num: int) -> str:
    """Get stage description"""
    descriptions = {
        1: "Global Loss Only (L_global)",
        2: "Global + Branch Losses (L_global + L_S&E&O)",
        3: "Global + Branch + Orthogonality (L_global + L_S&E&O + L_orth)",
        4: "All Losses (L_global + L_S&E&O + L_orth + L_distill)"
    }
    return descriptions.get(stage_num, f"Stage {stage_num}")


def main(args=None):
    if args is None:
        args = parse_args()
    if not hasattr(args, 'resume_stage'):
        args.resume_stage = None

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

    logger.info("=" * 80)
    logger.info("UPSTAR Staged Loss Training (Phase 5)")
    logger.info("=" * 80)

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
    model = UPSTARModel(
        num_items=num_items,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        use_gate=config['model']['use_gate']
    ).to(device)

    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create loss function
    loss_fn = create_loss_from_config(config)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determine which stages to train
    if args.stage is not None:
        stages_to_train = [args.stage]
        start_stage = args.stage
    else:
        stages_to_train = [1, 2, 3, 4]
        start_stage = 1

    # Resume from stage if specified
    if args.resume_stage is not None:
        resume_path = checkpoint_dir / f'stage{args.resume_stage}' / 'best_model.pt'
        if resume_path.exists():
            logger.info(f"Resuming from stage {args.resume_stage}")
            checkpoint = torch.load(resume_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_stage = args.resume_stage + 1
            stages_to_train = [s for s in stages_to_train if s >= start_stage]
        else:
            logger.warning(f"Resume checkpoint not found: {resume_path}")
            logger.warning("Starting from stage 1")

    # Train each stage
    stage_results = {}
    for stage_num in stages_to_train:
        stage_metrics = train_stage(
            stage_num, model, train_loader, val_loader,
            loss_fn, optimizer, device, config, checkpoint_dir
        )
        stage_results[f'stage{stage_num}'] = stage_metrics

        # Save final model after each stage
        final_path = checkpoint_dir / f'model_after_stage{stage_num}.pt'
        torch.save({
            'stage': stage_num,
            'model_state_dict': model.state_dict(),
            'metrics': stage_metrics,
            'config': config
        }, final_path)
        logger.info(f"Saved model after stage {stage_num} to {final_path}")

    # Test on final model
    logger.info("\n" + "=" * 80)
    logger.info("Testing on Final Model")
    logger.info("=" * 80)

    test_metrics = evaluate(
        model, test_loader, loss_fn, device,
        num_items, config['evaluation']['k_values']
    )

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
        pickle.dump({
            'test_metrics': test_metrics,
            'stage_results': stage_results,
            'config': config
        }, f)

    logger.info(f"\nSaved test results to {results_path}")

    # Print summary of all stages
    logger.info("\n" + "=" * 80)
    logger.info("STAGE SUMMARY")
    logger.info("=" * 80)
    for stage_key, stage_metrics in stage_results.items():
        logger.info(f"{stage_key.upper()}: NDCG@10 = {stage_metrics['NDCG@10']:.4f}, "
                   f"Recall@10 = {stage_metrics['Recall@10']:.4f}")

    logger.info(f"\nFINAL TEST: NDCG@10 = {test_metrics['NDCG@10']:.4f}, "
               f"Recall@10 = {test_metrics['Recall@10']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("UPSTAR Staged Loss Training Complete!")
    logger.info("=" * 80)

    return {
        'test_metrics': test_metrics,
        'stage_results': stage_results,
    }


if __name__ == '__main__':
    main()
