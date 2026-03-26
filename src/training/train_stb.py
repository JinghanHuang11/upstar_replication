"""
STB (Stable Transaction Bias) Computation - Offline Module

Computes stability scores for items and classifies into motivations.
Version 1: Engineering approximation
"""

import argparse
import torch
import numpy as np
import yaml
import pickle
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.graphs.item_time_graph import ItemTimeGraphBuilder
from src.models.stb_encoder import STBScorer
from src.graphs.perturbation import CombinedPerturbation
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stb.yaml')
    return parser.parse_args()


def load_data(config: dict) -> tuple:
    """Load sequences and item embeddings"""
    # Load sequences
    builder = ItemTimeGraphBuilder(config)
    train_sequences = builder.load_sequences('train')

    # Load item embeddings
    item_embeddings = builder.load_item_embeddings()

    return train_sequences, item_embeddings


def build_or_load_item_time_graph(
    config: dict,
    train_sequences: dict,
    item_embeddings: torch.Tensor
) -> dict:
    """Build or load item-time graph"""
    builder = ItemTimeGraphBuilder(config)

    # Try to load from cache
    graph = builder.load_graph('item_time_graph.pt')

    if graph is None:
        logger.info("Building item-time graph...")
        graph = builder.build_item_time_graph(train_sequences, item_embeddings)
        builder.save_graph(graph, 'item_time_graph.pt')

        # Log statistics
        stats = builder.compute_graph_statistics(graph)
        logger.info("Item-Time Graph Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    return graph


def compute_stb_scores(
    graph: dict,
    config: dict,
    device: torch.device
) -> torch.Tensor:
    """
    Compute STB scores using perturbation-based stability measure

    Args:
        graph: item-time graph
        config: configuration
        device: torch device

    Returns:
        stb_scores: [num_items] - stability scores (0-1)
    """
    logger.info("Computing STB scores...")

    # Create scorer
    input_dim = graph['node_features'].shape[1]
    scorer = STBScorer(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout']
    ).to(device)

    # Move graph to device
    graph_device = {
        'node_features': graph['node_features'].to(device),
        'edge_index': graph['edge_index'].to(device),
        'item_node_indices': graph['item_node_indices'].to(device),
        'time_node_indices': graph['time_node_indices'].to(device)
    }

    # Compute original representations
    logger.info("Computing original item representations...")
    with torch.no_grad():
        original_repr = scorer.encoder(
            graph_device['node_features'],
            graph_device['edge_index'],
            graph_device['item_node_indices']
        )

    # Perturbation rounds
    num_rounds = config['stb']['perturbation_rounds']
    all_perturbed_repr = []

    perturbation = CombinedPerturbation(
        noise_std=config['stb']['feature_noise_std'],
        removal_rate=config['stb']['edge_removal_rate'],
        addition_rate=config['stb']['edge_addition_rate']
    )

    logger.info(f"Running {num_rounds} perturbation rounds...")

    for round_idx in tqdm(range(num_rounds), desc="Perturbation"):
        # Perturb graph
        perturbed_graph = perturbation.perturb(graph)

        # Move to device
        perturbed_graph_device = {
            'node_features': perturbed_graph['node_features'].to(device),
            'edge_index': perturbed_graph['edge_index'].to(device),
            'item_node_indices': perturbed_graph['item_node_indices'].to(device)
        }

        # Compute perturbed representations
        with torch.no_grad():
            perturbed_repr = scorer.encoder(
                perturbed_graph_device['node_features'],
                perturbed_graph_device['edge_index'],
                perturbed_graph_device['item_node_indices']
            )

        all_perturbed_repr.append(perturbed_repr.cpu())

    # Stack all perturbed representations: [num_rounds, num_items, dim]
    all_perturbed_repr = torch.stack(all_perturbed_repr, dim=0)

    # Compute STB scores with worst-case aggregation (internal min over rounds)
    if scorer.use_mi and scorer.mi_estimator is not None:
        # MI mode: need graph summary with item/time separation
        # Explicitly pass item_node_indices and time_node_indices for accurate summary
        graph_summary = scorer.compute_graph_summary(
            graph_device['node_features'],
            graph_device['edge_index'],
            item_node_indices=graph_device['item_node_indices'],
            time_node_indices=graph_device['time_node_indices']
        )
        stb_scores = scorer.compute_stability_score(
            original_repr.cpu(),
            all_perturbed_repr,
            graph_summary.cpu()
        )
    else:
        # Baseline mode: cosine similarity
        stb_scores = scorer.compute_stability_score(
            original_repr.cpu(),
            all_perturbed_repr
        )

    logger.info(f"STB scores computed:")
    logger.info(f"  Min: {stb_scores.min():.4f}")
    logger.info(f"  Max: {stb_scores.max():.4f}")
    logger.info(f"  Mean: {stb_scores.mean():.4f}")
    logger.info(f"  Std: {stb_scores.std():.4f}")

    return stb_scores


def classify_motivations(
    stb_scores: torch.Tensor,
    config: dict
) -> torch.Tensor:
    """
    Classify items into motivation types based on STB scores

    ===================================================================
    PAPER ALIGNMENT (Section 3.1.4: Motivation Classification)
    ===================================================================
    Classification rule:
      - Top 50% STB scores → stable preference (label=1)
      - Bottom 40% STB scores → exploratory intent (label=0)
      - Middle 10% → uncategorized (label=2)

    ===================================================================
    ENGINEERING APPROXIMATION
    ===================================================================
    Current implementation uses GLOBAL ranking across all items:
      1. Sort all items by STB score (descending)
      2. Apply threshold ratios globally

    The paper does not specify whether ranking should be:
      - Global (all items together) ← CURRENT IMPLEMENTATION
      - Per-user (within each user's purchased items)
      - Per-time-period (within each time window)

    This global ranking is an engineering approximation that works well
    for datasets where item popularity follows a power law distribution.

    Future work: Experiment with per-user or per-time ranking strategies.

    ===================================================================

    Args:
        stb_scores: [num_items] - STB stability scores (higher = more stable)
        config: configuration dict with 'stb.stable_ratio' and 'stb.exploratory_ratio'

    Returns:
        labels: [num_items] - motivation labels
            - 0 = exploratory intent
            - 1 = stable preference
            - 2 = uncategorized
    """
    logger.info("Classifying motivations...")

    num_items = stb_scores.shape[0]

    # Read thresholds from config (paper-aligned: 0.5 and 0.4)
    stable_ratio = config['stb']['stable_ratio']        # Paper: 0.5
    exploratory_ratio = config['stb']['exploratory_ratio']  # Paper: 0.4

    # Validate ratios sum to ≤ 1.0
    if stable_ratio + exploratory_ratio > 1.0:
        logger.warning(
            f"Classification thresholds sum to {stable_ratio + exploratory_ratio:.2f} > 1.0, "
            f"clamping exploratory_ratio to {1.0 - stable_ratio:.2f}"
        )
        exploratory_ratio = 1.0 - stable_ratio

    # ===================================================================
    # GLOBAL RANKING (engineering approximation)
    # ===================================================================
    # Sort all items by STB score (descending: highest = most stable)
    # ===================================================================
    sorted_scores, sorted_indices = torch.sort(stb_scores, descending=False)

    # Initialize all labels as exploratory (0)
    labels = torch.zeros(num_items, dtype=torch.long)

    # ===================================================================
    # Top X% → stable preference (label=1)
    # ===================================================================
    # Paper: "top 50% STB scores → stable preference"
    # High STB = stable under worst-case perturbation = stable preference
    # ===================================================================
    num_stable = int(num_items * stable_ratio)
    if num_stable > 0:
        stable_indices = sorted_indices[-num_stable:]  # Last N items (highest scores)
        labels[stable_indices] = 1

    # ===================================================================
    # Bottom Y% → exploratory intent (label=0, already set)
    # ===================================================================
    # Paper: "bottom 40% STB scores → exploratory intent"
    # Low STB = unstable under perturbation = exploratory intent
    # ===================================================================
    num_exploratory = int(num_items * exploratory_ratio)
    # Labels are already 0 (exploratory), no need to explicitly set

    # ===================================================================
    # Middle → uncategorized (label=2)
    # ===================================================================
    # Paper: "remaining 10% → uncategorized"
    # These items have moderate STB scores, not clearly stable or exploratory
    # ===================================================================
    num_uncategorized = num_items - num_stable - num_exploratory
    if num_uncategorized > 0:
        # Middle items: indices [num_exploratory : num_exploratory + num_uncategorized]
        uncategorized_indices = sorted_indices[num_exploratory:num_exploratory + num_uncategorized]
        labels[uncategorized_indices] = 2

    # ===================================================================
    # Statistics & Logging
    # ===================================================================
    num_stable_final = (labels == 1).sum().item()
    num_exploratory_final = (labels == 0).sum().item()
    num_uncategorized_final = (labels == 2).sum().item()

    logger.info("Motivation classification (GLOBAL RANKING):")
    logger.info(f"  Stable (top {stable_ratio:.0%}):       "
               f"{num_stable_final:6d} ({num_stable_final/num_items*100:.1f}%)")
    logger.info(f"  Exploratory (bottom {exploratory_ratio:.0%}):  "
               f"{num_exploratory_final:6d} ({num_exploratory_final/num_items*100:.1f}%)")
    logger.info(f"  Uncategorized (middle):       "
               f"{num_uncategorized_final:6d} ({num_uncategorized_final/num_items*100:.1f}%)")

    return labels


def save_results(
    stb_scores: torch.Tensor,
    labels: torch.Tensor,
    config: dict
):
    """Save STB scores and labels"""
    output_dir = Path(config['logging']['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scores
    scores_path = output_dir / 'stb_scores.npy'
    np.save(scores_path, stb_scores.numpy())
    logger.info(f"Saved STB scores to {scores_path}")

    # Save labels
    labels_path = output_dir / 'motivation_labels.npy'
    np.save(labels_path, labels.numpy())
    logger.info(f"Saved motivation labels to {labels_path}")

    # Save statistics
    stats = {
        'num_items': len(stb_scores),
        'mean_score': stb_scores.mean().item(),
        'std_score': stb_scores.std().item(),
        'min_score': stb_scores.min().item(),
        'max_score': stb_scores.max().item(),
        'num_stable': (labels == 1).sum().item(),
        'num_exploratory': (labels == 0).sum().item(),
        'num_uncategorized': (labels == 2).sum().item()
    }

    stats_path = output_dir / 'stb_stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    logger.info(f"Saved statistics to {stats_path}")


def main(args=None):
    if args is None:
        args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config['seed'])

    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'stb.log'

    global logger
    logger = get_logger(__name__, str(log_file))

    logger.info("=" * 60)
    logger.info("STB Computation (Version 1: Engineering Approximation)")
    logger.info("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    train_sequences, item_embeddings = load_data(config)

    # Build item-time graph
    graph = build_or_load_item_time_graph(config, train_sequences, item_embeddings)

    # Compute STB scores
    stb_scores = compute_stb_scores(graph, config, device)

    # Classify motivations
    labels = classify_motivations(stb_scores, config)

    # Save results
    save_results(stb_scores, labels, config)

    logger.info("=" * 60)
    logger.info("STB computation complete!")
    logger.info("=" * 60)

    return {
        'stb_scores': stb_scores,
        'labels': labels,
    }


if __name__ == '__main__':
    main()
