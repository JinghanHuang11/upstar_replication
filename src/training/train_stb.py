"""
STB (Stable Transaction Bias) Computation - Paper-Aligned Default

This is the MAIN training entry point for STB computation.
Defaults to paper-aligned implementation with engineering fallback available.

Paper: UPSTAR Section 3.1.3 - STB Approximation via Mutual Information
Key Formula:
    STB_{n,t} = inf_{S'∈B_t} [1 - Pr[h([e(S')]_n) ≠ Stable Preference]]
             ≤ (I(S'; e(S')) + log 2) / log|Y|

Implementation:
    - I(S; e(S)) estimated using MINE (Mutual Information Neural Estimation)
    - Worst-case aggregation: min over β=40 perturbation rounds
    - PGD-based perturbation with ε_x=0.1, ε_a=0.1, α=0.4

Fallback:
    - Cosine similarity baseline (engineering approximation, NOT paper-aligned)
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
    parser = argparse.ArgumentParser(description='STB Computation (Paper-Aligned Default)')
    parser.add_argument('--config', type=str, default='configs/stb.yaml')
    parser.add_argument('--mode', type=str, default='paper',
                       choices=['paper', 'baseline', 'auto'],
                       help='Mode: paper (MI-based, default), baseline (cosine similarity), auto (auto-detect)')
    return parser.parse_args()


def load_data(config: dict) -> tuple:
    """Load sequences and item embeddings"""
    # Load sequences
    builder = ItemTimeGraphBuilder(config)
    train_sequences = builder.load_sequences('train')

    # Load item embeddings (GRAPH-ENHANCED from Phase 2)
    item_embeddings = builder.load_item_embeddings()

    return train_sequences, item_embeddings


def build_or_load_item_time_graph(
    config: dict,
    train_sequences: dict,
    item_embeddings: torch.Tensor
) -> dict:
    """Build or load item-time bipartite graph"""
    builder = ItemTimeGraphBuilder(config)

    # Try to load from cache
    graph = builder.load_graph('item_time_graph.pt')

    if graph is None:
        logger.info("Building item-time bipartite graph...")
        graph = builder.build_item_time_graph(train_sequences, item_embeddings)
        builder.save_graph(graph, 'item_time_graph.pt')

        # Log statistics
        stats = builder.compute_graph_statistics(graph)
        logger.info("Item-Time Bipartite Graph Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    return graph


def compute_stb_scores_paper_aligned(
    graph: dict,
    config: dict,
    device: torch.device
) -> torch.Tensor:
    """
    Compute STB scores using PAPER-ALIGNED implementation.

    Paper Formula (Section 3.1.3):
        STB = min_{rounds} [1 - MI_lower_bound]
        where MI_lower_bound = (I(S'; e(S')) + log 2) / log|Y|

    Key Components:
    1. MI-based scoring (not cosine similarity)
    2. Worst-case aggregation (min over β=40 rounds)
    3. PGD-based perturbation (ε_x=0.1, ε_a=0.1, α=0.4)
    4. Item-time bipartite graph (enforced bipartite edges)

    Args:
        graph: item-time bipartite graph
        config: configuration
        device: torch device

    Returns:
        stb_scores: [num_items] - stability scores (0-1, higher = more stable)
    """
    logger.info("=" * 80)
    logger.info("Computing STB Scores (PAPER-ALIGNED MODE)")
    logger.info("=" * 80)
    logger.info("Method: MI-based scoring with worst-case aggregation")
    logger.info("Reference: UPSTAR Section 3.1.3")
    logger.info("")

    # Import advanced perturbation if available
    try:
        from src.graphs.perturbation_advanced import PGDPerturbation
        use_pgd_perturbation = True
        logger.info("Using PGD-based perturbation (paper-aligned)")
    except ImportError:
        from src.graphs.perturbation import CombinedPerturbation
        use_pgd_perturbation = False
        logger.warning("PGD perturbation not available, using baseline perturbation")

    # Create scorer with MI estimator
    input_dim = graph['node_features'].shape[1]
    mi_hidden_dim = config['model'].get('mi_hidden_dim', 256)

    scorer = STBScorer(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        use_mi=True,  # PAPER-ALIGNED: MI-based scoring
        mi_hidden_dim=mi_hidden_dim
    ).to(device)

    logger.info(f"STB Scorer: MI-based (paper-aligned)")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Hidden dim: {config['model']['hidden_dim']}")
    logger.info(f"  MI hidden dim: {mi_hidden_dim}")
    logger.info("")

    # Move graph to device
    graph_device = {
        'node_features': graph['node_features'].to(device),
        'edge_index': graph['edge_index'].to(device),
        'item_node_indices': graph['item_node_indices'].to(device),
        'time_node_indices': graph['time_node_indices'].to(device)
    }

    # Compute original representations
    logger.info("Step 1: Computing original item representations...")
    with torch.no_grad():
        original_repr = scorer.encoder(
            graph_device['node_features'],
            graph_device['edge_index'],
            graph_device['item_node_indices']
        )

    # Perturbation rounds
    num_rounds = config['stb']['perturbation_rounds']  # Paper: β = 40
    all_perturbed_repr = []

    logger.info(f"Step 2: Running worst-case perturbation (β={num_rounds} rounds)...")
    logger.info("  Goal: inf_{S'∈B_t} [min over perturbation rounds]")

    if use_pgd_perturbation:
        # Paper-aligned PGD perturbation
        perturbation = PGDPerturbation(
            epsilon_x=config['stb']['epsilon_x'],      # Paper: ε_x = 0.1
            epsilon_a=config['stb']['epsilon_a'],      # Paper: ε_a = 0.1
            alpha=config['stb']['feature_step_size'],  # Paper: α = 0.4
            num_iterations=config['stb'].get('pgd_iterations', 10),  # PGD iterations
            enforce_bipartite=True  # Paper: bipartite constraint
        )
    else:
        # Baseline perturbation
        perturbation = CombinedPerturbation(
            noise_std=config['stb'].get('feature_noise_std', 0.1),
            removal_rate=config['stb'].get('edge_removal_rate', 0.1),
            addition_rate=config['stb'].get('edge_addition_rate', 0.1)
        )

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

        all_perturbed_repr.append(perturbed_repr)

    # Stack all perturbed representations: [num_rounds, num_items, dim]
    all_perturbed_repr = torch.stack(all_perturbed_repr, dim=0)

    logger.info(f"  Perturbation rounds completed: {num_rounds}")
    logger.info(f"  Shape: {all_perturbed_repr.shape}")
    logger.info("")

    # Compute graph summary for MI estimation
    logger.info("Step 3: Computing graph summary for MI estimation...")
    graph_summary = scorer.compute_graph_summary(
        graph_device['node_features'],
        graph_device['edge_index'],
        item_node_indices=graph_device['item_node_indices'],
        time_node_indices=graph_device['time_node_indices']
    )
    logger.info(f"  Graph summary shape: {graph_summary.shape}")
    logger.info("")

    # Compute STB scores with MI-based worst-case aggregation
    logger.info("Step 4: Computing STB scores (MI-based, worst-case aggregation)...")
    stb_scores = scorer.compute_stability_score(
        original_repr,
        all_perturbed_repr,
        graph_summary
    )

    logger.info(f"STB scores computed:")
    logger.info(f"  Min: {stb_scores.min():.4f}")
    logger.info(f"  Max: {stb_scores.max():.4f}")
    logger.info(f"  Mean: {stb_scores.mean():.4f}")
    logger.info(f"  Std: {stb_scores.std():.4f}")
    logger.info("")

    return stb_scores


def compute_stb_scores_baseline(
    graph: dict,
    config: dict,
    device: torch.device
) -> torch.Tensor:
    """
    Compute STB scores using BASELINE (engineering approximation).

    Uses cosine similarity instead of MI estimation.
    This is NOT paper-aligned but can be useful for debugging.

    Args:
        graph: item-time graph
        config: configuration
        device: torch device

    Returns:
        stb_scores: [num_items] - stability scores (0-1)
    """
    logger.info("=" * 80)
    logger.info("Computing STB Scores (BASELINE MODE - NOT PAPER-ALIGNED)")
    logger.info("=" * 80)
    logger.info("Method: Cosine similarity (engineering approximation)")
    logger.warning("This is NOT paper-aligned! Use --mode=paper for accurate results.")
    logger.info("")

    # Create scorer WITHOUT MI estimator
    input_dim = graph['node_features'].shape[1]
    scorer = STBScorer(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        use_mi=False  # BASELINE: cosine similarity
    ).to(device)

    logger.info(f"STB Scorer: Cosine similarity (baseline)")
    logger.info("")

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

    logger.info(f"Running {num_rounds} perturbation rounds...")

    perturbation = CombinedPerturbation(
        noise_std=config['stb']['feature_noise_std'],
        removal_rate=config['stb']['edge_removal_rate'],
        addition_rate=config['stb']['edge_addition_rate']
    )

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

        all_perturbed_repr.append(perturbed_repr)

    # Stack all perturbed representations: [num_rounds, num_items, dim]
    all_perturbed_repr = torch.stack(all_perturbed_repr, dim=0)

    # Compute STB scores with cosine similarity
    stb_scores = scorer.compute_stability_score(
        original_repr.cpu(),
        all_perturbed_repr
    )

    logger.info(f"STB scores computed:")
    logger.info(f"  Min: {stb_scores.min():.4f}")
    logger.info(f"  Max: {stb_scores.max():.4f}")
    logger.info(f"  Mean: {stb_scores.mean():.4f}")
    logger.info(f"  Std: {stb_scores.std():.4f}")
    logger.info("")

    return stb_scores


def classify_motivations(
    stb_scores: torch.Tensor,
    config: dict
) -> torch.Tensor:
    """
    Classify items into motivation types based on STB scores

    PAPER-ALIGNED RULE (Section 3.1.4):
      - Top 50% STB scores → stable preference (label=1)
      - Bottom 40% STB scores → exploratory intent (label=0)
      - Middle 10% → uncategorized (label=2)

    Args:
        stb_scores: [num_items] - STB stability scores (higher = more stable)
        config: configuration dict

    Returns:
        labels: [num_items] - motivation labels
            - 0 = exploratory intent
            - 1 = stable preference
            - 2 = uncategorized
    """
    logger.info("Classifying motivations (PAPER-ALIGNED)...")

    num_items = stb_scores.shape[0]

    # Paper-aligned thresholds
    stable_ratio = config['stb']['stable_ratio']        # Paper: 0.5
    exploratory_ratio = config['stb']['exploratory_ratio']  # Paper: 0.4

    # Validate ratios sum to ≤ 1.0
    if stable_ratio + exploratory_ratio > 1.0:
        logger.warning(
            f"Classification thresholds sum to {stable_ratio + exploratory_ratio:.2f} > 1.0, "
            f"clamping exploratory_ratio to {1.0 - stable_ratio:.2f}"
        )
        exploratory_ratio = 1.0 - stable_ratio

    # Sort all items by STB score (descending: highest = most stable)
    sorted_scores, sorted_indices = torch.sort(stb_scores, descending=True)

    # Determine cutoff indices
    num_stable = int(num_items * stable_ratio)
    num_exploratory = int(num_items * exploratory_ratio)

    # Initialize labels as uncategorized (label=2)
    labels = torch.full((num_items,), 2, dtype=torch.long)

    # Assign stable labels (top 50%)
    stable_indices = sorted_indices[:num_stable]
    labels[stable_indices] = 1

    # Assign exploratory labels (bottom 40%)
    exploratory_indices = sorted_indices[num_items - num_exploratory:]
    labels[exploratory_indices] = 0

    # Middle 10% remains uncategorized (label=2)

    # Log statistics
    num_stable = (labels == 1).sum().item()
    num_exploratory = (labels == 0).sum().item()
    num_uncategorized = (labels == 2).sum().item()

    logger.info(f"Motivation classification complete:")
    logger.info(f"  Stable preference (label=1): {num_stable} items ({num_stable/num_items*100:.1f}%)")
    logger.info(f"  Exploratory intent (label=0):   {num_exploratory} items ({num_exploratory/num_items*100:.1f}%)")
    logger.info(f"  Uncategorized (label=2):        {num_uncategorized} items ({num_uncategorized/num_items*100:.1f}%)")
    logger.info(f"")
    logger.info("Classification rule (paper-aligned):")
    logger.info(f"  Top {stable_ratio*100:.0f}% STB → stable")
    logger.info(f"  Bottom {exploratory_ratio*100:.0f}% STB → exploratory")
    logger.info(f"  Middle {10:.0f}% → uncategorized")

    return labels


def save_results(
    stb_scores: torch.Tensor,
    motivation_labels: torch.Tensor,
    config: dict
):
    """Save STB scores and motivation labels"""
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save STB scores
    stb_scores_path = Path(config['stb']['stb_scores_path'])
    stb_scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(stb_scores_path, stb_scores.cpu().numpy())
    logger.info(f"Saved STB scores to {stb_scores_path}")

    # Save motivation labels
    labels_path = Path(config['stb']['motivation_labels_path'])
    np.save(labels_path, motivation_labels.cpu().numpy())
    logger.info(f"Saved motivation labels to {labels_path}")

    # Save metadata
    metadata = {
        'num_items': stb_scores.shape[0],
        'stb_mean': float(stb_scores.mean().cpu()),
        'stb_std': float(stb_scores.std().cpu()),
        'num_stable': int((motivation_labels == 1).sum().cpu()),
        'num_exploratory': int((motivation_labels == 0).sum().cpu()),
        'num_uncategorized': int((motivation_labels == 2).sum().cpu()),
        'stable_ratio': config['stb']['stable_ratio'],
        'exploratory_ratio': config['stb']['exploratory_ratio']
    }

    metadata_path = checkpoint_dir / 'stb_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved metadata to {metadata_path}")


def main(args=None):
    if args is None:
        args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config.get('seed', 42))

    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'train_stb.log'

    global logger
    logger = get_logger(__name__, str(log_file))

    logger.info("=" * 80)
    logger.info("STB (Stable Transaction Bias) Computation")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Mode: {args.mode}")
    logger.info("")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info("")

    # Determine mode
    if args.mode == 'auto':
        # Auto-detect: try paper-aligned first, fallback to baseline
        logger.info("Auto mode: trying paper-aligned implementation...")
        try:
            from src.graphs.perturbation_advanced import PGDPerturbation
            mode = 'paper'
            logger.info("  → PGD perturbation available, using paper-aligned mode")
        except ImportError:
            mode = 'baseline'
            logger.warning("  → PGD perturbation not available, falling back to baseline mode")
            logger.warning("  → Install torch-geometric or use --mode=baseline explicitly")
    else:
        mode = args.mode

    # Load data
    train_sequences, item_embeddings = load_data(config)

    # Build item-time graph
    graph = build_or_load_item_time_graph(config, train_sequences, item_embeddings)

    # Compute STB scores
    if mode == 'paper':
        stb_scores = compute_stb_scores_paper_aligned(graph, config, device)
    else:  # baseline
        stb_scores = compute_stb_scores_baseline(graph, config, device)

    # Classify motivations
    motivation_labels = classify_motivations(stb_scores, config)

    # Save results
    save_results(stb_scores, motivation_labels, config)

    logger.info("=" * 80)
    logger.info("STB Computation Complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  STB scores: {config['stb']['stb_scores_path']}")
    logger.info(f"  Motivation labels: {config['stb']['motivation_labels_path']}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  Phase 4: Train UPSTAR model using these motivation labels")
    logger.info("    python -m src.training.train_upstar --config configs/tafeng_upstar.yaml")
    logger.info("")

    return {}


if __name__ == '__main__':
    main()
