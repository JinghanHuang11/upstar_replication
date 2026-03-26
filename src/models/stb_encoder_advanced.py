"""
Advanced STB Encoder with Multiple Perturbation Strategies (Phase 6)

Supports:
- Version 1: Simple perturbation (baseline)
- Upgrade 1: PGD-based feature perturbation
- Upgrade 2: Continuous topology perturbation
- Upgrade 3: MI-based objective
- Upgrade 4: Full iterative worst-case

Provides unified interface for all versions with comparison capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


# Import base encoder
from src.models.stb_encoder import STBEncoder as STBEncoderV1, STBScorer as STBScorerV1


class AdvancedSTBEncoder(nn.Module):
    """
    Advanced STB encoder with support for multiple perturbation strategies

    Wrapper around base STBEncoder that adds advanced perturbation support.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        perturbation_version: str = 'v1',
        perturbation_config: Optional[Dict] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.perturbation_version = perturbation_version

        # Base encoder
        self.base_encoder = STBEncoderV1(input_dim, hidden_dim, dropout)

        # Create perturbation strategy
        from src.graphs.perturbation_advanced import create_perturbation

        perturbation_config = perturbation_config or {}
        self.perturbation = create_perturbation(
            version=perturbation_version,
            **perturbation_config
        )

        logger.info(f"Initialized AdvancedSTBEncoder: "
                   f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"perturbation={perturbation_version}")

    def compute_stability_scores(
        self,
        graph: Dict[str, torch.Tensor],
        num_perturbation_rounds: int = 10,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Compute STB scores using configured perturbation strategy

        Paper-aligned: STB = inf_{S'∈B_t} stability(S')
                      → approximated by min (worst-case) over perturbation rounds

        Args:
            graph: Dict with 'node_features', 'edge_index', 'item_node_indices'
            num_perturbation_rounds: Number of perturbation rounds
            return_details: If True, return detailed statistics

        Returns:
            stb_scores: [num_items] - WORST-CASE (min) stability scores (paper-aligned)
            details: Optional dict with:
                - 'per_round_scores': [rounds, items] - item-level scores for each round
                - 'worst_case_scores': [items] - min over rounds (PRIMARY STB metric)
                - 'mean_scores': [items] - mean over rounds (for comparison/debugging only)
                - 'attack_info': list of per-round attack info
                - 'original_repr': [items, dim]
                - 'perturbed_repr': [rounds, items, dim]
        """
        device = graph['node_features'].device
        item_indices = graph['item_node_indices']
        num_items = len(item_indices)

        # Get original representation
        with torch.no_grad():
            original_repr = self.base_encoder(
                graph['node_features'],
                graph['edge_index'],
                item_indices
            )

        # Perturbation rounds
        all_perturbed_repr = []
        all_attack_info = []

        for round_idx in range(num_perturbation_rounds):
            # Apply perturbation
            if self.perturbation_version == 'v1':
                # V1: Simple combined perturbation
                perturbed_graph = self.perturbation.perturb(graph)
                attack_info = {}
            elif self.perturbation_version == 'upgrade1':
                # PGD feature perturbation
                perturbed_features, attack_info = self.perturbation.perturb(
                    graph['node_features'],
                    graph['item_node_indices'],
                    encoder=self.base_encoder,
                    edge_index=graph['edge_index']
                )
                perturbed_graph = graph.copy()
                perturbed_graph['node_features'] = perturbed_features
            elif self.perturbation_version == 'upgrade2':
                # Continuous topology perturbation
                perturbed_edges, attack_info = self.perturbation.perturb(
                    graph['edge_index'],
                    graph['node_features'].shape[0],
                    graph['num_items'],
                    encoder=self.base_encoder,
                    node_features=graph['node_features'],
                    item_node_indices=graph['item_node_indices']
                )
                perturbed_graph = graph.copy()
                perturbed_graph['edge_index'] = perturbed_edges
            elif self.perturbation_version == 'upgrade3':
                # MI-based perturbation
                perturbed_features, attack_info = self.perturbation.perturb(
                    graph['node_features'],
                    graph['item_node_indices'],
                    encoder=self.base_encoder,
                    edge_index=graph['edge_index']
                )
                perturbed_graph = graph.copy()
                perturbed_graph['node_features'] = perturbed_features
            elif self.perturbation_version == 'full':
                # Full iterative worst-case
                perturbed_graph, attack_info = self.perturbation.perturb(
                    graph,
                    encoder=self.base_encoder,
                    target_items=None
                )
            else:
                raise ValueError(f"Unknown perturbation version: {self.perturbation_version}")

            # Get perturbed representation
            with torch.no_grad():
                perturbed_repr = self.base_encoder(
                    perturbed_graph['node_features'],
                    perturbed_graph['edge_index'],
                    item_indices
                )

            all_perturbed_repr.append(perturbed_repr)
            all_attack_info.append(attack_info)

        # Stack perturbed representations: [rounds, num_items, dim]
        stacked_perturbed = torch.stack(all_perturbed_repr, dim=0)

        # ===================================================================
        # Item-level score computation (per round)
        # ===================================================================
        # Current: cosine similarity between original and perturbed repr
        # NOTE: This is a BASELINE metric, not the full MI-based STB from paper.
        #       For true paper alignment, should use MI estimator like in stb_encoder.py
        # ===================================================================

        original_norm = F.normalize(original_repr, dim=1)  # [num_items, dim]
        perturbed_norm = F.normalize(stacked_perturbed, dim=2)  # [rounds, num_items, dim]

        # Per-round item-level scores: [rounds, num_items]
        per_round_scores = (original_norm.unsqueeze(0) * perturbed_norm).sum(dim=2)

        # ===================================================================
        # Aggregation over perturbation rounds
        # ===================================================================
        # Paper: STB = inf_{S'∈B_t} stability(S')
        #        → approximate infimum via MIN (worst-case) over rounds
        #
        # Mean is provided for comparison/debugging but NOT the primary metric
        # ===================================================================

        # PRIMARY: Worst-case (min) over rounds — paper-aligned STB
        worst_case_scores = per_round_scores.min(dim=0).values  # [num_items]

        # SECONDARY: Mean over rounds — for comparison with old implementation
        mean_scores = per_round_scores.mean(dim=0)  # [num_items]

        if return_details:
            details = {
                'per_round_scores': per_round_scores,  # [rounds, num_items]
                'worst_case_scores': worst_case_scores,  # [num_items] — PRIMARY
                'mean_scores': mean_scores,  # [num_items] — for comparison only
                'attack_info': all_attack_info,
                'original_repr': original_repr,
                'perturbed_repr': stacked_perturbed,
                # Metadata for interpretation
                'aggregation_info': {
                    'primary_metric': 'worst_case_scores (min over rounds)',
                    'primary_reason': 'Paper: STB = inf_{S\'∈B_t} stability(S\')',
                    'secondary_metric': 'mean_scores (for comparison with old impl)',
                    'score_type': 'cosine_similarity (baseline, not full MI-based)'
                }
            }
            return worst_case_scores, details
        else:
            return worst_case_scores, None


class STBComparator:
    """
    Compare different STB versions

    Runs multiple perturbation strategies on the same data and compares:
    - STB score distributions
    - Motivation label ratios
    - Computation time
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        versions: List[str] = ['v1', 'upgrade1', 'upgrade2', 'upgrade3', 'full']
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.versions = versions

        # Create encoders for each version
        self.encoders = {}
        for version in versions:
            self.encoders[version] = AdvancedSTBEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                perturbation_version=version
            )

        logger.info(f"Initialized STBComparator with versions: {versions}")

    def compare(
        self,
        graph: Dict[str, torch.Tensor],
        num_perturbation_rounds: int = 10,
        top_percent: float = 0.5,
        middle_percent: float = 0.1,
        bottom_percent: float = 0.4
    ) -> Dict:
        """
        Compare all versions

        Returns:
            comparison_results: Dict with results for each version
        """
        results = {}
        item_indices = graph['item_node_indices']
        num_items = len(item_indices)

        for version in self.versions:
            logger.info(f"\nEvaluating version: {version}")
            encoder = self.encoders[version]

            # Compute STB scores
            import time
            start_time = time.time()

            stb_scores, details = encoder.compute_stability_scores(
                graph,
                num_perturbation_rounds=num_perturbation_rounds,
                return_details=True
            )

            elapsed_time = time.time() - start_time

            # Classify motivations
            sorted_indices = torch.argsort(stb_scores, descending=True)

            # Top (stable)
            num_stable = int(num_items * top_percent)
            stable_indices = sorted_indices[:num_stable]

            # Bottom (exploratory)
            num_expl = int(num_items * bottom_percent)
            expl_indices = sorted_indices[-num_expl:]

            # Middle (uncategorized)
            num_middle = num_items - num_stable - num_expl
            middle_indices = sorted_indices[num_stable:num_stable + num_middle]

            # Create motivation labels
            motivation_labels = torch.zeros(num_items, device=stb_scores.device)
            motivation_labels[stable_indices] = 1  # Stable
            motivation_labels[middle_indices] = 2  # Uncategorized
            # Exploratory stays 0

            version_results = {
                'stb_scores': stb_scores.cpu(),
                'motivation_labels': motivation_labels.cpu(),
                'time_elapsed': elapsed_time,
                'statistics': {
                    'mean': stb_scores.mean().item(),
                    'std': stb_scores.std().item(),
                    'min': stb_scores.min().item(),
                    'max': stb_scores.max().item(),
                    'median': stb_scores.median().item()
                },
                'distribution': {
                    'num_stable': num_stable,
                    'num_exploratory': num_expl,
                    'num_uncategorized': num_middle,
                    'pct_stable': top_percent * 100,
                    'pct_exploratory': bottom_percent * 100,
                    'pct_uncategorized': middle_percent * 100
                },
                'details': details
            }

            results[version] = version_results

            # Log summary
            logger.info(f"  STB scores (worst-case/min): mean={version_results['statistics']['mean']:.4f}, "
                       f"std={version_results['statistics']['std']:.4f}")
            if details and 'mean_scores' in details:
                # Also log mean-over-rounds for comparison
                mean_over_rounds = details['mean_scores'].mean().item()
                logger.info(f"    (for comparison: mean-over-rounds={mean_over_rounds:.4f})")
            logger.info(f"  Distribution: Stable={num_stable}, "
                       f"Exploratory={num_expl}, Uncategorized={num_middle}")
            logger.info(f"  Time: {elapsed_time:.2f}s")

        return results

    def print_comparison(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Print comparison table

        Args:
            results: Results from compare()
            save_path: Optional path to save comparison
        """
        print("\n" + "=" * 80)
        print("STB VERSION COMPARISON")
        print("=" * 80)

        # Header
        print(f"{'Version':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Time(s)':<10}")
        print("-" * 80)

        # Rows
        for version, version_results in results.items():
            stats = version_results['statistics']
            print(f"{version:<15} "
                  f"{stats['mean']:<10.4f} "
                  f"{stats['std']:<10.4f} "
                  f"{stats['min']:<10.4f} "
                  f"{stats['max']:<10.4f} "
                  f"{version_results['time_elapsed']:<10.2f}")

        print("-" * 80)

        # Distribution comparison
        print("\nMotivation Label Distribution:")
        print(f"{'Version':<15} {'Stable':<12} {'Exploratory':<15} {'Uncategorized':<15}")
        print("-" * 80)

        for version, version_results in results.items():
            dist = version_results['distribution']
            print(f"{version:<15} "
                  f"{dist['num_stable']:<12} "
                  f"{dist['num_exploratory']:<15} "
                  f"{dist['num_uncategorized']:<15}")

        print("=" * 80)

        # Save if requested
        if save_path:
            import json
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert tensors to lists for JSON serialization
            save_results = {}
            for version, version_results in results.items():
                save_results[version] = {
                    'stb_scores': version_results['stb_scores'].tolist(),
                    'motivation_labels': version_results['motivation_labels'].tolist(),
                    'time_elapsed': version_results['time_elapsed'],
                    'statistics': version_results['statistics'],
                    'distribution': {
                        k: v if not isinstance(v, torch.Tensor) else v.item()
                        for k, v in version_results['distribution'].items()
                    }
                }

            with open(save_path, 'w') as f:
                json.dump(save_results, f, indent=2)

            logger.info(f"Saved comparison to {save_path}")


def create_stb_encoder(
    input_dim: int,
    hidden_dim: int = 512,
    version: str = 'v1',
    **kwargs
) -> AdvancedSTBEncoder:
    """
    Factory function to create STB encoder

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        version: 'v1', 'upgrade1', 'upgrade2', 'upgrade3', 'full'
        **kwargs: Additional arguments

    Returns:
        STB encoder
    """
    return AdvancedSTBEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        perturbation_version=version,
        perturbation_config=kwargs
    )


if __name__ == '__main__':
    # Test advanced encoder
    print("Testing Advanced STB Encoder...")

    num_items = 1000
    num_times = 50
    input_dim = 128
    hidden_dim = 512

    # Create dummy graph
    graph = {
        'node_features': torch.randn(num_items + num_times, input_dim),
        'edge_index': torch.randint(0, num_items + num_times, (2, 5000)),
        'item_node_indices': torch.arange(num_items),
        'num_items': num_items
    }

    # Test different versions
    print("\n1. Testing different versions:")
    for version in ['v1', 'upgrade1', 'full']:
        print(f"\n  Version: {version}")
        encoder = create_stb_encoder(input_dim, hidden_dim, version=version)

        stb_scores, _ = encoder.compute_stability_scores(
            graph,
            num_perturbation_rounds=3,
            return_details=True
        )

        print(f"    STB scores shape: {stb_scores.shape}")
        print(f"    Mean: {stb_scores.mean():.4f}, Std: {stb_scores.std():.4f}")

    # Test comparator
    print("\n2. Testing comparator:")
    comparator = STBComparator(input_dim, hidden_dim, versions=['v1', 'upgrade1'])
    results = comparator.compare(graph, num_perturbation_rounds=3)
    comparator.print_comparison(results)

    print("\nAll advanced encoder tests passed!")
