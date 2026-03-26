"""
Perturbation strategies for STB computation (Version 1: Engineering approximation)

Lightweight perturbations for measuring item stability:
1. Feature perturbation: add noise to item embeddings
2. Edge perturbation: remove/add edges
"""

import torch
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FeaturePerturbation:
    """
    Perturb node features (item embeddings)
    """

    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std

    def perturb(
        self,
        node_features: torch.Tensor,
        item_node_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Add Gaussian noise to item node features

        Args:
            node_features: [num_nodes, dim]
            item_node_indices: [num_items]

        Returns:
            perturbed_features: [num_nodes, dim]
        """
        perturbed = node_features.clone()

        # Add noise to item nodes only
        noise = torch.randn_like(perturbed[item_node_indices]) * self.noise_std
        perturbed[item_node_indices] += noise

        return perturbed


class EdgePerturbation:
    """
    Perturb graph structure (edges)

    Version 1: Simple edge removal/addition
    """

    def __init__(
        self,
        removal_rate: float = 0.1,
        addition_rate: float = 0.1
    ):
        self.removal_rate = removal_rate
        self.addition_rate = addition_rate

    def perturb(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_items: int
    ) -> torch.Tensor:
        """
        Perturb edges: remove some, add some

        Args:
            edge_index: [2, num_edges]
            num_nodes: total nodes
            num_items: number of item nodes

        Returns:
            perturbed_edge_index: [2, num_edges']
        """
        num_edges = edge_index.shape[1]

        # Edge removal
        keep_mask = torch.rand(num_edges) > self.removal_rate
        kept_edges = edge_index[:, keep_mask]

        # Edge addition
        num_add = int(num_edges * self.addition_rate)
        if num_add > 0:
            # Add random edges (prefer item-item or item-time)
            new_src = torch.randint(0, num_items, (num_add,))
            new_dst = torch.randint(0, num_nodes, (num_add,))
            new_edges = torch.stack([new_src, new_dst], dim=0)

            perturbed_edges = torch.cat([kept_edges, new_edges], dim=1)
        else:
            perturbed_edges = kept_edges

        return perturbed_edges


class CombinedPerturbation:
    """
    Combine feature and edge perturbations
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        removal_rate: float = 0.1,
        addition_rate: float = 0.1
    ):
        self.feature_perturb = FeaturePerturbation(noise_std)
        self.edge_perturb = EdgePerturbation(removal_rate, addition_rate)

    def perturb(
        self,
        graph: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply both feature and edge perturbations

        Args:
            graph: {
                'node_features': [num_nodes, dim],
                'edge_index': [2, num_edges],
                'item_node_indices': [num_items],
                ...
            }

        Returns:
            perturbed_graph: same structure
        """
        perturbed = graph.copy()

        # Perturb features
        perturbed['node_features'] = self.feature_perturb.perturb(
            graph['node_features'],
            graph['item_node_indices']
        )

        # Perturb edges
        perturbed['edge_index'] = self.edge_perturb.perturb(
            graph['edge_index'],
            graph['node_features'].shape[0],
            graph['num_items']
        )

        return perturbed


if __name__ == '__main__':
    # Test perturbations
    print("Testing Perturbations...")

    num_nodes = 1000
    num_items = 800
    num_edges = 5000
    dim = 128

    # Create dummy graph
    graph = {
        'node_features': torch.randn(num_nodes, dim),
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
        'item_node_indices': torch.arange(num_items),
        'num_items': num_items
    }

    # Test feature perturbation
    print("\n1. Feature Perturbation:")
    feature_perturb = FeaturePerturbation(noise_std=0.1)
    perturbed_features = feature_perturb.perturb(
        graph['node_features'],
        graph['item_node_indices']
    )
    print(f"Original mean: {graph['node_features'].mean():.4f}")
    print(f"Perturbed mean: {perturbed_features.mean():.4f}")
    print(f"Difference: {(perturbed_features - graph['node_features']).abs().mean():.4f}")

    # Test edge perturbation
    print("\n2. Edge Perturbation:")
    edge_perturb = EdgePerturbation(removal_rate=0.1, addition_rate=0.1)
    perturbed_edges = edge_perturb.perturb(
        graph['edge_index'],
        num_nodes,
        num_items
    )
    print(f"Original edges: {graph['edge_index'].shape[1]}")
    print(f"Perturbed edges: {perturbed_edges.shape[1]}")

    # Test combined
    print("\n3. Combined Perturbation:")
    combined = CombinedPerturbation(noise_std=0.1, removal_rate=0.1, addition_rate=0.1)
    perturbed_graph = combined.perturb(graph)
    print(f"Features changed: {(perturbed_graph['node_features'] != graph['node_features']).sum().item()}")
    print(f"Edges changed: {perturbed_graph['edge_index'].shape[1]} -> {graph['edge_index'].shape[1]}")

    print("\nAll perturbation tests passed!")
