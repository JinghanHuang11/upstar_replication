"""
Advanced STB Training Script (Phase 6: Upgrades)

Supports multiple perturbation strategies:
- V1: Simple baseline (original)
- Upgrade 1: PGD-based feature perturbation
- Upgrade 2: Continuous topology perturbation
- Upgrade 3: MI-based objective
- Upgrade 4: Full iterative worst-case

Provides comparison between all versions.
"""

import argparse
import torch
import torch.nn as nn
import yaml
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import json
import time

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.graphs.item_time_graph import create_item_time_graph
from src.models.load_item_embeddings import get_item_embeddings
from src.models.stb_encoder_advanced import (
    create_stb_encoder,
    STBComparator
)
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stb_advanced.yaml')
    parser.add_argument('--version', type=str, default=None,
                       help='Run specific version only (v1, upgrade1, upgrade2, upgrade3, full)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all versions')
    return parser.parse_args()


def load_item_embeddings(config):
    """Load item embeddings from Phase 2"""
    logger.info("Loading item embeddings...")

    embeddings_path = Path(config['stb']['item_embeddings_path'])
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Item embeddings not found: {embeddings_path}\n"
            f"Please run Phase 2 first: bash scripts/run_item_repr.sh"
        )

    item_embeddings = torch.load(embeddings_path)
    num_items, embed_dim = item_embeddings.shape

    logger.info(f"Loaded item embeddings: {item_embeddings.shape}")

    return item_embeddings, num_items, embed_dim


def create_item_time_graph(config, item_embeddings):
    """Create or load item-time graph"""
    cache_dir = Path(config['dataset']['cache_dir'])
    graph_path = cache_dir / 'item_time_graph.pt'

    if graph_path.exists():
        logger.info(f"Loading cached item-time graph from {graph_path}")
        graph_data = torch.load(graph_path)
        logger.info(f"  Nodes: {graph_data['node_features'].shape[0]}")
        logger.info(f"  Edges: {graph_data['edge_index'].shape[1]}")
        return graph_data
    else:
        logger.info("Creating item-time graph...")
        graph_data = create_item_time_graph(config, item_embeddings)

        # Cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(graph_data, graph_path)
        logger.info(f"Cached graph to {graph_path}")

        return graph_data


def run_single_version(
    version: str,
    config: dict,
    item_embeddings: torch.Tensor,
    graph_data: dict,
    output_dir: Path
) -> dict:
    """
    Run STB computation for a single version

    Returns:
        results: Dict with STB scores, labels, statistics
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running STB Version: {version}")
    logger.info(f"{'=' * 80}")

    start_time = time.time()

    # Get configuration for this version
    if version == 'v1':
        perturbation_config = {
            'noise_std': 0.1,
            'removal_rate': 0.1,
            'addition_rate': 0.1
        }
    elif version == 'upgrade1':
        perturbation_config = config['advanced'].get('upgrade1', {})
    elif version == 'upgrade2':
        perturbation_config = config['advanced'].get('upgrade2', {})
    elif version == 'upgrade3':
        perturbation_config = config['advanced'].get('upgrade3', {})
    elif version == 'full':
        perturbation_config = config['advanced'].get('full', {})
    else:
        raise ValueError(f"Unknown version: {version}")

    # Create encoder
    num_items = item_embeddings.shape[0]
    embed_dim = item_embeddings.shape[1]
    hidden_dim = config['model']['hidden_dim']

    encoder = create_stb_encoder(
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        version=version,
        **perturbation_config
    )

    # Prepare graph
    graph = {
        'node_features': graph_data['node_features'],
        'edge_index': graph_data['edge_index'],
        'item_node_indices': torch.arange(num_items),
        'num_items': num_items
    }

    # Compute STB scores
    num_rounds = config['stb']['perturbation_rounds']

    logger.info(f"Computing STB scores with {num_rounds} perturbation rounds...")
    stb_scores, details = encoder.compute_stability_scores(
        graph,
        num_perturbation_rounds=num_rounds,
        return_details=True
    )

    elapsed_time = time.time() - start_time

    # Statistics
    stats = {
        'mean': stb_scores.mean().item(),
        'std': stb_scores.std().item(),
        'min': stb_scores.min().item(),
        'max': stb_scores.max().item(),
        'median': stb_scores.median().item()
    }

    logger.info(f"\nSTB Score Statistics:")
    logger.info(f"  Mean:   {stats['mean']:.4f}")
    logger.info(f"  Std:    {stats['std']:.4f}")
    logger.info(f"  Min:    {stats['min']:.4f}")
    logger.info(f"  Max:    {stats['max']:.4f}")
    logger.info(f"  Median: {stats['median']:.4f}")
    logger.info(f"  Time:   {elapsed_time:.2f}s")

    # Classify motivations
    top_percent = config['stb']['top_percent']
    middle_percent = config['stb']['middle_percent']
    bottom_percent = config['stb']['bottom_percent']

    sorted_indices = torch.argsort(stb_scores, descending=True)

    num_stable = int(num_items * top_percent)
    num_middle = int(num_items * middle_percent)
    num_expl = num_items - num_stable - num_middle

    stable_indices = sorted_indices[:num_stable]
    middle_indices = sorted_indices[num_stable:num_stable + num_middle]
    expl_indices = sorted_indices[num_stable + num_middle:]

    motivation_labels = torch.zeros(num_items)
    motivation_labels[stable_indices] = 1  # Stable
    motivation_labels[middle_indices] = 2  # Uncategorized
    # Exploratory stays 0

    logger.info(f"\nMotivation Label Distribution:")
    logger.info(f"  Stable (1):         {num_stable} ({top_percent*100:.1f}%)")
    logger.info(f"  Exploratory (0):    {num_expl} ({bottom_percent*100:.1f}%)")
    logger.info(f"  Uncategorized (2):  {num_middle} ({middle_percent*100:.1f}%)")

    # Save results
    version_output_dir = output_dir / version
    version_output_dir.mkdir(parents=True, exist_ok=True)

    # Save STB scores
    np.save(version_output_dir / 'stb_scores.npy', stb_scores.cpu().numpy())

    # Save motivation labels
    np.save(version_output_dir / 'motivation_labels.npy', motivation_labels.cpu().numpy())

    # Save statistics
    stats_path = version_output_dir / 'stb_stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump({
            'stb_scores': stb_scores.cpu().numpy(),
            'motivation_labels': motivation_labels.cpu().numpy(),
            'statistics': stats,
            'distribution': {
                'num_stable': num_stable,
                'num_exploratory': num_expl,
                'num_uncategorized': num_middle,
                'pct_stable': top_percent,
                'pct_exploratory': bottom_percent,
                'pct_uncategorized': middle_percent
            },
            'time_elapsed': elapsed_time,
            'version': version,
            'config': perturbation_config
        }, f)

    logger.info(f"\nSaved results to {version_output_dir}")

    return {
        'stb_scores': stb_scores.cpu().numpy(),
        'motivation_labels': motivation_labels.cpu().numpy(),
        'statistics': stats,
        'distribution': {
            'num_stable': num_stable,
            'num_exploratory': num_expl,
            'num_uncategorized': num_middle
        },
        'time_elapsed': elapsed_time
    }


def compare_versions(
    config: dict,
    item_embeddings: torch.Tensor,
    graph_data: dict,
    output_dir: Path
) -> dict:
    """
    Compare multiple STB versions
    """
    versions = config['advanced'].get('versions', ['v1', 'upgrade1', 'full'])

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Comparing STB Versions: {versions}")
    logger.info(f"{'=' * 80}")

    # Run all versions
    all_results = {}
    for version in versions:
        results = run_single_version(
            version,
            config,
            item_embeddings,
            graph_data,
            output_dir
        )
        all_results[version] = results

    # Print comparison
    print("\n" + "=" * 80)
    print("STB VERSION COMPARISON")
    print("=" * 80)

    print(f"{'Version':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Time(s)':<10}")
    print("-" * 80)

    for version, results in all_results.items():
        stats = results['statistics']
        print(f"{version:<15} "
              f"{stats['mean']:<10.4f} "
              f"{stats['std']:<10.4f} "
              f"{stats['min']:<10.4f} "
              f"{stats['max']:<10.4f} "
              f"{results['time_elapsed']:<10.2f}")

    print("-" * 80)

    print("\nMotivation Label Distribution:")
    print(f"{'Version':<15} {'Stable':<12} {'Exploratory':<15} {'Uncategorized':<15}")
    print("-" * 80)

    for version, results in all_results.items():
        dist = results['distribution']
        print(f"{version:<15} "
              f"{dist['num_stable']:<12} "
              f"{dist['num_exploratory']:<15} "
              f"{dist['num_uncategorized']:<15}")

    print("=" * 80)

    # Save comparison
    comparison_path = output_dir / 'comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nSaved comparison to {comparison_path}")

    return all_results


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config.get('seed', 42))

    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'train.log'

    global logger
    logger = get_logger(__name__, str(log_file))

    logger.info("=" * 80)
    logger.info("Advanced STB Computation (Phase 6)")
    logger.info("=" * 80)

    # Load item embeddings
    item_embeddings, num_items, embed_dim = load_item_embeddings(config)

    # Create item-time graph
    graph_data = create_item_time_graph(config, item_embeddings)

    # Output directory
    output_dir = Path(config['stb']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run
    if args.compare:
        # Compare all versions
        results = compare_versions(config, item_embeddings, graph_data, output_dir)

        logger.info("\n" + "=" * 80)
        logger.info("Advanced STB Computation Complete!")
        logger.info("=" * 80)

        logger.info(f"\nResults saved to {output_dir}")
        logger.info("\nTo use a specific version for UPSTAR training:")
        logger.info("  Update config['stb']['motivation_labels_path'] to point to:")
        for version in results.keys():
            logger.info(f"    {output_dir / version / 'motivation_labels.npy'}")

    elif args.version:
        # Run single version
        results = run_single_version(
            args.version,
            config,
            item_embeddings,
            graph_data,
            output_dir
        )

        logger.info("\n" + "=" * 80)
        logger.info(f"STB Computation Complete for {args.version}!")
        logger.info("=" * 80)

        logger.info(f"\nResults saved to {output_dir / args.version}")
        logger.info(f"\nTo use with UPSTAR training:")
        logger.info(f"  motivation_labels_path: {output_dir / args.version / 'motivation_labels.npy'}")

    else:
        # Use default from config
        versions = config['advanced'].get('versions', ['v1'])

        if len(versions) == 1:
            results = run_single_version(
                versions[0],
                config,
                item_embeddings,
                graph_data,
                output_dir
            )
        else:
            results = compare_versions(config, item_embeddings, graph_data, output_dir)


if __name__ == '__main__':
    main()
