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
        'item_node_indices': graph['item_node_indices'].to(device)
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
    all_similarities = []

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

        # Compute stability (cosine similarity)
        similarity = scorer.compute_stability_score(original_repr, perturbed_repr)
        all_similarities.append(similarity.cpu())

    # Average across rounds
    all_similarities = torch.stack(all_similarities, dim=0)  # [num_rounds, num_items]
    stb_scores = all_similarities.mean(dim=0)  # [num_items]

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

    Rules:
    - Top 50% stb_score -> stable (label=1)
    - Bottom 40% stb_score -> exploratory (label=0)
    - Middle 10% -> uncategorized (label=2)

    Args:
        stb_scores: [num_items]
        config: configuration

    Returns:
        labels: [num_items] - 0=exploratory, 1=stable, 2=uncategorized
    """
    logger.info("Classifying motivations...")

    num_items = stb_scores.shape[0]

    # Sort scores
    sorted_scores, sorted_indices = torch.sort(stb_scores)

    # Initialize labels as exploratory (0)
    labels = torch.zeros(num_items, dtype=torch.long)

    # Top 50% -> stable
    num_stable = int(num_items * 0.5)
    stable_indices = sorted_indices[-num_stable:]
    labels[stable_indices] = 1

    # Bottom 40% -> exploratory (already 0)
    num_exploratory = int(num_items * 0.4)

    # Middle 10% -> uncategorized
    num_uncategorized = num_items - num_stable - num_exploratory
    uncategorized_indices = sorted_indices[num_exploratory:num_exploratory + num_uncategorized]
    labels[uncategorized_indices] = 2

    # Statistics
    num_stable = (labels == 1).sum().item()
    num_exploratory = (labels == 0).sum().item()
    num_uncategorized = (labels == 2).sum().item()

    logger.info("Motivation classification:")
    logger.info(f"  Stable:       {num_stable:6d} ({num_stable/num_items*100:.1f}%)")
    logger.info(f"  Exploratory:  {num_exploratory:6d} ({num_exploratory/num_items*100:.1f}%)")
    logger.info(f"  Uncategorized: {num_uncategorized:6d} ({num_uncategorized/num_items*100:.1f}%)")

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
