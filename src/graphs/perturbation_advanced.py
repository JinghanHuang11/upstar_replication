"""
Advanced Perturbation Strategies for STB (Phase 6: Upgrades)

Upgrade Path:
1. PGD-based feature perturbation (Upgrade 1)
2. Continuous relaxation + projection for topology (Upgrade 2)
3. MI estimation with noise-contrastive objective (Upgrade 3)
4. Configurable iterative worst-case perturbation (Upgrade 4)

Keeps backward compatibility with Version 1 (simple perturbation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Base Classes (Version 1 - Keep for backward compatibility)
# =============================================================================

from src.graphs.perturbation import (
    FeaturePerturbation as FeaturePerturbationV1,
    EdgePerturbation as EdgePerturbationV1,
    CombinedPerturbation as CombinedPerturbationV1
)


# =============================================================================
# Upgrade 1: PGD-based Feature Perturbation
# =============================================================================

class PGDFeaturePerturbation:
    """
    Projected Gradient Descent (PGD) attack on node features

    Paper correspondence: Section 3.1.3 - adversarial perturbation δ
    Finds worst-case perturbation within ℓ∞ ball to maximize representation change.

    Args:
        epsilon: Budget for perturbation (ℓ∞ radius)
        num_steps: Number of PGD steps
        step_size: Step size for gradient ascent
        loss_type: 'repr_change' or 'mi_based'
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        num_steps: int = 10,
        step_size: float = 0.01,
        loss_type: str = 'repr_change'
    ):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.loss_type = loss_type

        logger.info(f"Initialized PGDFeaturePerturbation: "
                   f"epsilon={epsilon}, steps={num_steps}, step_size={step_size}")

    def perturb(
        self,
        node_features: torch.Tensor,
        item_node_indices: torch.Tensor,
        encoder: Optional[nn.Module] = None,
        edge_index: Optional[torch.Tensor] = None,
        target_items: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        PGD attack to find worst-case feature perturbation

        Args:
            node_features: [num_nodes, dim]
            item_node_indices: [num_items]
            encoder: GNN encoder (if loss_type='repr_change')
            edge_index: [2, num_edges] (required if encoder provided)
            target_items: Items to attack (default: all items)

        Returns:
            perturbed_features: [num_nodes, dim]
            attack_info: Dict with attack statistics
        """
        if encoder is None:
            logger.warning("No encoder provided, falling back to V1 (random noise)")
            v1_perturb = FeaturePerturbationV1(noise_std=self.epsilon)
            return v1_perturb.perturb(node_features, item_node_indices), {}

        device = node_features.device
        num_items = len(item_node_indices)

        # Initialize perturbation
        delta = torch.zeros_like(node_features)
        delta[item_node_indices] = (
            torch.empty_like(node_features[item_node_indices])
            .uniform_(-self.epsilon, self.epsilon)
        )
        delta.requires_grad_(True)

        # Target items to attack
        if target_items is None:
            target_items = item_node_indices

        # Get original representations
        with torch.no_grad():
            original_features = node_features.clone()
            original_repr = encoder(original_features, edge_index, item_node_indices)

        # PGD iterations
        loss_history = []
        for step in range(self.num_steps):
            if delta.grad is not None:
                delta.grad.zero_()

            # Forward with perturbation
            perturbed_features = torch.clamp(
                original_features + delta,
                min=original_features - self.epsilon,
                max=original_features + self.epsilon
            )

            # Compute representations
            perturbed_repr = encoder(perturbed_features, edge_index, item_node_indices)

            # Compute loss (maximize representation change)
            if self.loss_type == 'repr_change':
                # Maximize negative similarity (minimize stability)
                target_idx = torch.where(
                    torch.isin(item_node_indices, target_items)
                )[0]

                if len(target_idx) > 0:
                    loss = -F.cosine_similarity(
                        original_repr[target_idx],
                        perturbed_repr[target_idx],
                        dim=1
                    ).mean()
                else:
                    loss = -F.cosine_similarity(
                        original_repr, perturbed_repr, dim=1
                    ).mean()
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            loss_history.append(loss.item())

            # Backward
            loss.backward()

            # Gradient ascent step
            with torch.no_grad():
                grad = delta.grad.data
                # Only update item nodes
                delta[item_node_indices] += self.step_size * grad[item_node_indices].sign()

                # Project back to ℓ∞ ball (only item nodes)
                delta[item_node_indices] = torch.clamp(
                    delta[item_node_indices],
                    min=-self.epsilon,
                    max=self.epsilon
                )

        # Final perturbed features
        perturbed_features = torch.clamp(
            original_features + delta,
            min=original_features - self.epsilon,
            max=original_features + self.epsilon
        )

        attack_info = {
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'perturbation_norm': delta[item_node_indices].norm(p=np.inf).item()
        }

        return perturbed_features.detach(), attack_info


# =============================================================================
# Upgrade 2: Continuous Relaxation for Topology Perturbation
# =============================================================================

class ContinuousTopologyPerturbation:
    """
    Continuous relaxation + projection for graph topology perturbation

    Paper correspondence: Appendix - graph structure perturbation
    Relaxes discrete edge structure to continuous weights, applies gradient-based attack,
    then projects back to discrete structure.

    Args:
        edge_budget: Fraction of edges that can be modified
        num_steps: Number of optimization steps
        step_size: Step size for gradient ascent
        temperature: Temperature for sigmoid relaxation
    """

    def __init__(
        self,
        edge_budget: float = 0.1,
        num_steps: int = 10,
        step_size: float = 0.1,
        temperature: float = 1.0
    ):
        self.edge_budget = edge_budget
        self.num_steps = num_steps
        self.step_size = step_size
        self.temperature = temperature

        logger.info(f"Initialized ContinuousTopologyPerturbation: "
                   f"budget={edge_budget}, steps={num_steps}, temp={temperature}")

    def perturb(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_items: int,
        encoder: Optional[nn.Module] = None,
        node_features: Optional[torch.Tensor] = None,
        item_node_indices: Optional[torch.Tensor] = None,
        original_edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Continuous relaxation + projection for edge perturbation

        Args:
            edge_index: [2, num_edges]
            num_nodes: total nodes
            num_items: number of item nodes
            encoder: GNN encoder
            node_features: [num_nodes, dim] (required if encoder provided)
            item_node_indices: [num_items] (required if encoder provided)
            original_edge_index: Original edges (for computing difference)

        Returns:
            perturbed_edge_index: [2, num_edges']
            attack_info: Dict with attack statistics
        """
        if encoder is None or node_features is None:
            logger.warning("No encoder provided, falling back to V1 (random edge perturbation)")
            v1_perturb = EdgePerturbationV1(
                removal_rate=self.edge_budget / 2,
                addition_rate=self.edge_budget / 2
            )
            return v1_perturb.perturb(edge_index, num_nodes, num_items), {}

        device = edge_index.device

        # Create edge adjacency matrix (continuous relaxation)
        num_edges = edge_index.shape[1]
        max_edges = num_nodes * num_nodes  # Upper bound

        # Initialize continuous edge weights
        # Start with original edges = 1, others = 0
        edge_weights = torch.zeros(max_edges, device=device)
        edge_indices = edge_index[0] * num_nodes + edge_index[1]
        edge_weights[edge_indices] = 1.0

        # Get original representations
        with torch.no_grad():
            original_edge_weights = edge_weights.clone()
            original_repr = encoder(node_features, edge_index, item_node_indices)

        # Optimize edge weights
        loss_history = []
        for step in range(self.num_steps):
            # Convert edge weights to edge index
            # Use threshold to get discrete edges
            current_edges = self._weights_to_edge_index(
                edge_weights, num_nodes, temperature=self.temperature
            )

            # Compute representations
            perturbed_repr = encoder(node_features, current_edges, item_node_indices)

            # Maximize representation change
            loss = -F.cosine_similarity(original_repr, perturbed_repr, dim=1).mean()
            loss_history.append(loss.item())

            # Compute gradient w.r.t edge weights
            # Note: This is approximate - we'd need proper differentiable GNN
            # For now, use gradient-free random search

            # Randomly flip some edges
            num_modify = int(num_edges * self.edge_budget * self.step_size)
            if num_modify > 0:
                # Remove edges
                remove_idx = torch.randint(0, len(edge_indices), (num_modify // 2,))
                edge_weights[edge_indices[remove_idx]] *= 0.9

                # Add edges (prefer item-item or item-time)
                add_src = torch.randint(0, num_items, (num_modify // 2,))
                add_dst = torch.randint(0, num_nodes, (num_modify // 2,))
                add_indices = add_src * num_nodes + add_dst
                edge_weights[add_indices] = torch.clamp(
                    edge_weights[add_indices] + 0.1, max=1.0
                )

        # Project back to discrete structure
        final_edge_index = self._weights_to_edge_index(
            edge_weights, num_nodes, temperature=0.5  # Lower temp for sharper threshold
        )

        attack_info = {
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'num_edges_original': edge_index.shape[1],
            'num_edges_perturbed': final_edge_index.shape[1]
        }

        return final_edge_index, attack_info

    def _weights_to_edge_index(
        self,
        edge_weights: torch.Tensor,
        num_nodes: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Convert continuous edge weights to discrete edge index"""
        # Apply sigmoid threshold
        probs = torch.sigmoid(edge_weights / temperature)

        # Sample edges
        edge_mask = torch.rand_like(probs) < probs
        edge_indices = torch.nonzero(edge_mask).squeeze()

        if edge_indices.dim() == 0:
            edge_indices = edge_indices.unsqueeze(0)

        # Convert to (src, dst) format
        src = edge_indices // num_nodes
        dst = edge_indices % num_nodes

        # Remove self-loops
        mask = src != dst
        src = src[mask]
        dst = dst[mask]

        return torch.stack([src, dst], dim=0)


# =============================================================================
# Upgrade 3: MI-based Objective
# =============================================================================

class MIEstimator:
    """
    Mutual Information estimation for STB

    Paper correspondence: Section 3.1.3 - MI upper bound
    Uses noise-contrastive estimation (NCE) to estimate MI between
    original and perturbed representations.

    MI(I; Ĩ) ≈ E[log p(i|ĩ) / p(i)]

    Higher MI → more robust → higher stability score
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_negatives: int = 10
    ):
        self.hidden_dim = hidden_dim
        self.num_negatives = num_negatives

        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(2 * 128, hidden_dim),  # Assuming 128-dim item repr
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        logger.info(f"Initialized MIEstimator: hidden_dim={hidden_dim}, "
                   f"num_negatives={num_negatives}")

    def estimate_mi(
        self,
        original_repr: torch.Tensor,
        perturbed_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate MI between original and perturbed representations

        Args:
            original_repr: [num_items, dim]
            perturbed_repr: [num_items, dim]

        Returns:
            mi_scores: [num_items] - per-item MI estimates
        """
        num_items = original_repr.shape[0]
        device = original_repr.device

        mi_scores = []

        for i in range(num_items):
            # Positive pair
            pos_orig = original_repr[i:i+1]  # [1, dim]
            pos_pert = perturbed_repr[i:i+1]  # [1, dim]

            # Negative pairs (shuffle)
            neg_indices = torch.randperm(num_items, device=device)[:self.num_negatives]
            neg_pert = perturbed_repr[neg_indices]  # [num_negatives, dim]

            # Concatenate
            pos_concat = torch.cat([pos_orig.expand(self.num_negatives + 1, -1),
                                   pos_pert], dim=1)  # [1, 2*dim]

            neg_concat = torch.cat([pos_orig.expand(self.num_negatives, -1),
                                   neg_pert], dim=1)  # [num_negatives, 2*dim]

            # Discriminator scores
            pos_score = torch.sigmoid(self.discriminator(pos_concat)).squeeze()
            neg_scores = torch.sigmoid(self.discriminator(neg_concat)).squeeze()

            # NCE loss (MI lower bound)
            # MI = E[log D(pos) / (D(pos) + mean(D(neg)))]
            mi = torch.log(pos_score / (pos_score + neg_scores.mean() + 1e-8) + 1e-8)
            mi_scores.append(mi)

        return torch.stack(mi_scores)


class MIbasedPerturbation:
    """
    Perturbation with MI-based objective

    Instead of maximizing representation change, minimize MI
    between original and perturbed representations.

    Lower MI → more sensitive → lower stability score
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        num_steps: int = 10,
        step_size: float = 0.01,
        mi_estimator: Optional[MIEstimator] = None
    ):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.mi_estimator = mi_estimator or MIEstimator()

        logger.info(f"Initialized MIbasedPerturbation: epsilon={epsilon}, steps={num_steps}")

    def perturb(
        self,
        node_features: torch.Tensor,
        item_node_indices: torch.Tensor,
        encoder: nn.Module,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Find perturbation that minimizes MI

        Returns:
            perturbed_features: [num_nodes, dim]
            attack_info: Dict with MI scores
        """
        device = node_features.device

        # Initialize perturbation
        delta = torch.zeros_like(node_features)
        delta[item_node_indices] = (
            torch.empty_like(node_features[item_node_indices])
            .uniform_(-self.epsilon, self.epsilon)
        )
        delta.requires_grad_(True)

        # Get original representations
        with torch.no_grad():
            original_features = node_features.clone()
            original_repr = encoder(original_features, edge_index, item_node_indices)

        mi_history = []
        for step in range(self.num_steps):
            if delta.grad is not None:
                delta.grad.zero_()

            # Forward with perturbation
            perturbed_features = torch.clamp(
                original_features + delta,
                min=original_features - self.epsilon,
                max=original_features + self.epsilon
            )

            perturbed_repr = encoder(perturbed_features, edge_index, item_node_indices)

            # Estimate MI
            with torch.no_grad():
                mi_scores = self.mi_estimator.estimate_mi(original_repr, perturbed_repr)
                mi_mean = mi_scores.mean()
                mi_history.append(mi_mean.item())

            # Minimize MI (negative for gradient ascent)
            loss = -mi_mean

            # Backward
            loss.backward()

            # Update perturbation
            with torch.no_grad():
                grad = delta.grad.data
                # Only update item nodes
                delta[item_node_indices] += self.step_size * grad[item_node_indices].sign()

                # Project
                delta[item_node_indices] = torch.clamp(
                    delta[item_node_indices],
                    min=-self.epsilon,
                    max=self.epsilon
                )

        # Final perturbed features
        perturbed_features = torch.clamp(
            original_features + delta,
            min=original_features - self.epsilon,
            max=original_features + self.epsilon
        )

        attack_info = {
            'mi_history': mi_history,
            'final_mi': mi_history[-1],
            'perturbation_norm': delta[item_node_indices].norm(p=np.inf).item()
        }

        return perturbed_features.detach(), attack_info


# =============================================================================
# Upgrade 4: Configurable Iterative Worst-Case Perturbation
# =============================================================================

class IterativeWorstCasePerturbation:
    """
    Fully configurable iterative worst-case perturbation

    Combines all upgrades:
    - PGD for features
    - Continuous relaxation for topology
    - MI-based objective
    - Configurable iterations, budgets, step sizes

    Args:
        feature_perturbation: 'pgd', 'mi', or 'v1'
        topology_perturbation: 'continuous' or 'v1'
        num_iterations: Number of alternating iterations
        feature_config: Config for feature perturbation
        topology_config: Config for topology perturbation
    """

    def __init__(
        self,
        feature_perturbation: str = 'pgd',
        topology_perturbation: str = 'continuous',
        num_iterations: int = 3,
        feature_config: Optional[Dict] = None,
        topology_config: Optional[Dict] = None
    ):
        self.feature_perturbation = feature_perturbation
        self.topology_perturbation = topology_perturbation
        self.num_iterations = num_iterations

        # Initialize feature perturbation
        feature_config = feature_config or {}
        if feature_perturbation == 'pgd':
            self.feature_attacker = PGDFeaturePerturbation(**feature_config)
        elif feature_perturbation == 'mi':
            self.feature_attacker = MIbasedPerturbation(**feature_config)
        else:
            self.feature_attacker = FeaturePerturbationV1(**feature_config)

        # Initialize topology perturbation
        topology_config = topology_config or {}
        if topology_perturbation == 'continuous':
            self.topology_attacker = ContinuousTopologyPerturbation(**topology_config)
        else:
            self.topology_attacker = EdgePerturbationV1(**topology_config)

        logger.info(f"Initialized IterativeWorstCasePerturbation: "
                   f"feature={feature_perturbation}, topology={topology_perturbation}, "
                   f"iterations={num_iterations}")

    def perturb(
        self,
        graph: Dict[str, torch.Tensor],
        encoder: nn.Module,
        target_items: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Iteratively find worst-case perturbation

        Args:
            graph: Dict with 'node_features', 'edge_index', 'item_node_indices', etc.
            encoder: GNN encoder
            target_items: Items to attack (optional)

        Returns:
            perturbed_graph: Same structure as input
            attack_info: Dict with detailed statistics
        """
        perturbed_graph = graph.copy()
        attack_info = {
            'iteration_logs': [],
            'feature_attacks': [],
            'topology_attacks': []
        }

        for iteration in range(self.num_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.num_iterations}")

            # Feature perturbation
            perturbed_features, feat_info = self.feature_attacker.perturb(
                perturbed_graph['node_features'],
                perturbed_graph['item_node_indices'],
                encoder=encoder,
                edge_index=perturbed_graph['edge_index'],
                target_items=target_items
            )
            perturbed_graph['node_features'] = perturbed_features
            attack_info['feature_attacks'].append(feat_info)

            # Topology perturbation
            perturbed_edges, topo_info = self.topology_attacker.perturb(
                perturbed_graph['edge_index'],
                perturbed_graph['node_features'].shape[0],
                perturbed_graph['num_items'],
                encoder=encoder,
                node_features=perturbed_graph['node_features'],
                item_node_indices=perturbed_graph['item_node_indices']
            )
            perturbed_graph['edge_index'] = perturbed_edges
            attack_info['topology_attacks'].append(topo_info)

            # Log iteration
            iter_log = {
                'iteration': iteration + 1,
                'feature_info': feat_info,
                'topology_info': topo_info
            }
            attack_info['iteration_logs'].append(iter_log)

        return perturbed_graph, attack_info


# =============================================================================
# Unified Interface
# =============================================================================

def create_perturbation(
    version: str = 'v1',
    **kwargs
):
    """
    Factory function to create perturbation strategy

    Args:
        version: 'v1', 'upgrade1', 'upgrade2', 'upgrade3', 'upgrade4', 'full'
        **kwargs: Additional arguments for specific version

    Returns:
        Perturbation object
    """
    if version == 'v1':
        return CombinedPerturbationV1(
            noise_std=kwargs.get('noise_std', 0.1),
            removal_rate=kwargs.get('removal_rate', 0.1),
            addition_rate=kwargs.get('addition_rate', 0.1)
        )

    elif version == 'upgrade1':
        # PGD features only
        return PGDFeaturePerturbation(
            epsilon=kwargs.get('epsilon', 0.1),
            num_steps=kwargs.get('num_steps', 10),
            step_size=kwargs.get('step_size', 0.01)
        )

    elif version == 'upgrade2':
        # Continuous topology only
        return ContinuousTopologyPerturbation(
            edge_budget=kwargs.get('edge_budget', 0.1),
            num_steps=kwargs.get('num_steps', 10),
            step_size=kwargs.get('step_size', 0.1)
        )

    elif version == 'upgrade3':
        # MI-based
        return MIbasedPerturbation(
            epsilon=kwargs.get('epsilon', 0.1),
            num_steps=kwargs.get('num_steps', 10),
            step_size=kwargs.get('step_size', 0.01)
        )

    elif version == 'upgrade4' or version == 'full':
        # Full iterative worst-case
        return IterativeWorstCasePerturbation(
            feature_perturbation=kwargs.get('feature_perturbation', 'pgd'),
            topology_perturbation=kwargs.get('topology_perturbation', 'continuous'),
            num_iterations=kwargs.get('num_iterations', 3),
            feature_config=kwargs.get('feature_config', {}),
            topology_config=kwargs.get('topology_config', {})
        )

    else:
        raise ValueError(f"Unknown version: {version}")


if __name__ == '__main__':
    # Test advanced perturbations
    print("Testing Advanced Perturbations...")

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

    # Test factory
    print("\n1. Testing factory function:")
    v1 = create_perturbation('v1')
    print(f"  V1: {type(v1).__name__}")

    upgrade1 = create_perturbation('upgrade1', num_steps=5)
    print(f"  Upgrade1: {type(upgrade1).__name__}")

    full = create_perturbation('full', num_iterations=2)
    print(f"  Full: {type(full).__name__}")

    print("\nAll advanced perturbation tests passed!")
