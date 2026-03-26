"""
STB Encoder: GNN for encoding item-time graph

1-layer GNN encoder for learning item representations from item-time graph.
Used for STB (Stable Transaction Bias) computation.

Paper Reference: Section 3.1.3 - STB Approximation via Mutual Information

Key Idea:
    STB = min_S'∈B I(S'; e(S'))
    where I(S; e(S)) is approximated using a trainable discriminator (MINE).

Components:
    - STBEncoder: 1-layer GNN encoder
    - MIEstimator: Mutual Information estimator (discriminator)
    - STBScorer: Compute stability using MI (preferred) or cosine similarity (baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MIEstimator(nn.Module):
    """
    Mutual Information Estimator using MINE (Mutual Information Neural Estimation)

    Paper: Belghazi et al. (2018) - MINE: Mutual Information Neural Estimation
    UPSTAR Paper: Section 3.1.3 - Approximate Solution using MI lower bound

    Estimates I(S; e(S)) where:
        S: attributed item-time graph (features + structure)
        e(S): GNN encoder output (item representations)

    Architecture:
        - Joint network: T(x, y) for paired (graph, repr)
        - Marginal network: T(x, y_shuffle) for negative samples

    Loss:
        L = E[T(x, y)] - log(E[exp(T(x, y')))])

    MI estimate: I(S; e(S)) ≈ sample_mean(T(x, y)) - log(sample_mean(exp(T(x, y'))))
    """

    def __init__(
        self,
        repr_dim: int,
        graph_feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        """
        Args:
            repr_dim: Dimension of item representations e(S)
            graph_feat_dim: Dimension of graph summary features
            hidden_dim: Hidden dimension for discriminator
            num_layers: Number of hidden layers
        """
        super().__init__()

        self.repr_dim = repr_dim
        self.graph_feat_dim = graph_feat_dim
        self.hidden_dim = hidden_dim

        # Build statistic network T(x, y)
        # Takes concatenated [graph_summary, item_repr] -> scalar
        input_dim = repr_dim + graph_feat_dim

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        # Final layer outputs scalar statistic
        layers.append(nn.Linear(current_dim, 1))

        self.statistic_network = nn.Sequential(*layers)

        logger.info(f"Initialized MIEstimator: "
                   f"repr_dim={repr_dim}, graph_feat_dim={graph_feat_dim}, "
                   f"hidden_dim={hidden_dim}, num_layers={num_layers}")

    def compute_graph_summary(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        item_node_indices: Optional[torch.Tensor] = None,
        time_node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute graph-level summary features for attributed item-time bipartite graph

        ENHANCED (2026-03): Separates item-side and time-side summaries to better capture
        timing dependence and co-purchase patterns in the bipartite structure.

        Paper semantics (Section 3.1.3):
        - Attributed item-time graph encodes:
          * item intrinsic attributes (item node features)
          * timing dependence (time node connectivity patterns)
          * co-purchase dependence (item-time bipartite edges)
        - Exploratory intent often shows as:
          * one-hop neighbors skewed toward specific time nodes
          * two-hop relationships revealing co-purchase patterns
        - Stable preference should remain stable under perturbations

        Args:
            node_features: [num_nodes, feat_dim] - all node features
            edge_index: [2, num_edges] - bipartite edges (item→time)
            item_node_indices: [num_items] - optional, which nodes are items
            time_node_indices: [num_times] - optional, which nodes are times

        Returns:
            summary: [1, graph_feat_dim] - concatenated summary:
                * item_summary: mean of item node features
                * time_summary: mean of time node features (typically zeros)
                * edge_density: graph sparsity statistic
                * degree_stats: item/time degree distribution features
        """
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]

        # ===================================================================
        # Infer item/time node indices if not provided
        # ===================================================================
        # Heuristic: assume first N nodes are items, rest are times
        # (matches ItemTimeGraphBuilder convention)
        # ===================================================================
        if item_node_indices is None or time_node_indices is None:
            # Use edge structure to infer: sources are items, targets are times
            # (bipartite graph should have item→time edges only)
            src_nodes = edge_index[0].unique()
            tgt_nodes = edge_index[1].unique()

            # Items are sources, times are targets
            if item_node_indices is None:
                item_node_indices = src_nodes
            if time_node_indices is None:
                time_node_indices = tgt_nodes

        num_items = len(item_node_indices)
        num_times = len(time_node_indices)

        # ===================================================================
        # Separated node-type summaries (paper-aligned)
        # ===================================================================
        # Item summary: captures intrinsic item attributes
        item_features = node_features[item_node_indices]  # [num_items, feat_dim]
        item_summary = item_features.mean(dim=0, keepdim=True)  # [1, feat_dim]

        # Time summary: captures temporal patterns (typically zero vectors)
        time_features = node_features[time_node_indices]  # [num_times, feat_dim]
        time_summary = time_features.mean(dim=0, keepdim=True)  # [1, feat_dim]

        # ===================================================================
        # Graph structure statistics (capture timing & co-purchase dependence)
        # ===================================================================
        # Edge density: overall graph sparsity
        edge_density = num_edges / (num_nodes * num_nodes) if num_nodes > 0 else 0.0

        # Item degree distribution: how concentrated purchases are
        # (high variance = some items purchased much more than others)
        item_degrees = torch.bincount(
            edge_index[0], minlength=num_nodes
        )[item_node_indices].float()
        item_degree_mean = item_degrees.mean() if num_items > 0 else torch.tensor(0.0)
        item_degree_std = item_degrees.std() if num_items > 1 else torch.tensor(0.0)

        # Time degree distribution: how temporally concentrated purchases are
        # (high variance = purchases clustered in specific time periods)
        time_degrees = torch.bincount(
            edge_index[1], minlength=num_nodes
        )[time_node_indices].float()
        time_degree_mean = time_degrees.mean() if num_times > 0 else torch.tensor(0.0)
        time_degree_std = time_degrees.std() if num_times > 1 else torch.tensor(0.0)

        # ===================================================================
        # Concatenate all summary components
        # ===================================================================
        # [item_feat_dim] + [time_feat_dim] + [1 density] + [4 degree stats]
        # = 2 * feat_dim + 5
        summary_components = [
            item_summary,          # [1, feat_dim]
            time_summary,          # [1, feat_dim]
            torch.tensor([[edge_density]], device=node_features.device),  # [1, 1]
            torch.tensor([[item_degree_mean]], device=node_features.device),  # [1, 1]
            torch.tensor([[item_degree_std]], device=node_features.device),   # [1, 1]
            torch.tensor([[time_degree_mean]], device=node_features.device),  # [1, 1]
            torch.tensor([[time_degree_std]], device=node_features.device),   # [1, 1]
        ]

        summary = torch.cat(summary_components, dim=1)  # [1, 2*feat_dim + 5]

        # Project to target dimension if needed
        if summary.shape[1] != self.graph_feat_dim:
            if not hasattr(self, 'summary_proj'):
                self.summary_proj = nn.Linear(
                    summary.shape[1], self.graph_feat_dim
                ).to(node_features.device)
            summary = self.summary_proj(summary)

        return summary

    def forward(
        self,
        item_repr: torch.Tensor,
        graph_summary: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MI statistic T(x, y)

        Args:
            item_repr: [batch_size, repr_dim] - item representations e(S)
            graph_summary: [batch_size, graph_feat_dim] - graph summary S

        Returns:
            statistics: [batch_size, 1] - MI statistics
        """
        # Concatenate and pass through network
        combined = torch.cat([item_repr, graph_summary], dim=1)
        statistics = self.statistic_network(combined)

        return statistics

    def compute_mi_loss(
        self,
        item_repr: torch.Tensor,
        perturbed_item_repr: torch.Tensor,
        graph_summary: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute MI loss using MINE objective

        Args:
            item_repr: [num_items, repr_dim] - original e(S)
            perturbed_item_repr: [num_items, repr_dim] - perturbed e(S')
            graph_summary: [1, graph_feat_dim] - graph summary (broadcasted)

        Returns:
            loss: Scalar MI loss (to minimize)
            metrics: Dict with MI estimate details
        """
        num_items = item_repr.shape[0]

        # Expand graph summary to match items
        graph_summary_expanded = graph_summary.expand(num_items, -1)

        # Joint: T(x, y) - paired representations
        joint_stats = self(item_repr, graph_summary_expanded)  # [num_items, 1]

        # Marginal: T(x, y') - shuffled representations (negative samples)
        # Shuffle perturbed repr along batch dimension
        perm = torch.randperm(num_items, device=item_repr.device)
        marginal_repr = perturbed_item_repr[perm]

        marginal_stats = self(marginal_repr, graph_summary_expanded)  # [num_items, 1]

        # MINE loss: E[T(x, y)] - log(E[exp(T(x, y'))])
        joint_mean = joint_stats.mean()
        marginal_exp = torch.exp(marginal_stats).mean()

        # Avoid log(0)
        marginal_log = torch.log(marginal_exp + 1e-8)

        loss = joint_mean - marginal_log

        # MI estimate (lower bound)
        mi_estimate = joint_mean - marginal_log.detach()

        metrics = {
            'mi_estimate': mi_estimate.item(),
            'joint_mean': joint_mean.item(),
            'marginal_exp': marginal_exp.item(),
            'loss': loss.item()
        }

        return loss, metrics

    def estimate_mi(
        self,
        item_repr: torch.Tensor,
        graph_summary: torch.Tensor,
        num_neg_samples: int = 1
    ) -> torch.Tensor:
        """
        Estimate mutual information I(S; e(S))

        Args:
            item_repr: [num_items, repr_dim]
            graph_summary: [1, graph_feat_dim] or [num_items, graph_feat_dim]
            num_neg_samples: Number of negative samples for marginal

        Returns:
            mi_estimate: Scalar MI estimate
        """
        num_items = item_repr.shape[0]

        if graph_summary.shape[0] == 1:
            graph_summary = graph_summary.expand(num_items, -1)

        # Joint statistics — no no_grad so callers can differentiate through this
        # (needed for adversarial PGD loop that maximizes MI loss)
        joint_stats = self(item_repr, graph_summary)
        joint_mean = joint_stats.mean()

        # Marginal statistics (shuffle negatives, average over multiple shuffles)
        marginal_means = []
        for _ in range(num_neg_samples):
            perm = torch.randperm(num_items, device=item_repr.device)
            marginal_repr = item_repr[perm]
            marginal_stats = self(marginal_repr, graph_summary)
            marginal_means.append(torch.exp(marginal_stats).mean())

        marginal_mean = torch.stack(marginal_means).mean()

        # MINE lower bound: E[T(x,y)] - log(E[exp(T(x,y'))])
        mi_estimate = joint_mean - torch.log(marginal_mean + 1e-8)

        return mi_estimate


class STBEncoder(nn.Module):
    """
    1-layer GNN encoder for item-time graph

    Architecture:
    - Input: node features (item embeddings + time zeros)
    - 1-layer GNN aggregation
    - Output: item node representations

    Hidden size: 512 (as per requirements)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layer (simple aggregation)
        self.gnn_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

        logger.info(f"Initialized STBEncoder: input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        item_node_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode item-time graph

        Args:
            node_features: [num_nodes, input_dim] - all node features
            edge_index: [2, num_edges] - graph structure
            item_node_indices: [num_items] - which nodes are items

        Returns:
            item_repr: [num_items, input_dim] - item representations
        """
        num_nodes = node_features.shape[0]

        # Project to hidden space
        h = self.input_proj(node_features)  # [num_nodes, hidden_dim]
        h = self.dropout(h)

        # GNN aggregation
        # Aggregate neighbor information
        src, dst = edge_index[0], edge_index[1]

        # Initialize aggregated features
        h_agg = torch.zeros_like(h)

        # Aggregate (scatter add)
        h_agg.index_add_(0, dst, h[src])

        # Normalize by degree
        degree = torch.zeros(num_nodes, device=h.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        degree = torch.clamp(degree, min=1).unsqueeze(-1)
        h_agg = h_agg / degree

        # Apply GNN transformation
        h_out = self.gnn_weight(h_agg) + h  # Residual connection
        h_out = F.relu(h_out)

        # Extract item node representations
        item_repr = h_out[item_node_indices]  # [num_items, hidden_dim]

        # Project back to input dimension
        item_repr = self.output_proj(item_repr)  # [num_items, input_dim]

        return item_repr


class STBScorer(nn.Module):
    """
    STB Scorer: Compute stability scores for items

    Two modes:
    1. MI-based (preferred): Uses mutual information estimator
       - STB ≈ I(S; e(S)) under perturbations
       - More aligned with paper Section 3.1.3

    2. Cosine similarity (baseline): Simple representation stability
       - Faster, no training required
       - Engineering approximation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_mi: bool = False,
        mi_hidden_dim: int = 256
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: GNN hidden dimension
            dropout: Dropout rate
            use_mi: Whether to use MI estimator (True) or cosine similarity (False)
            mi_hidden_dim: Hidden dimension for MI estimator
        """
        super().__init__()

        self.input_dim = input_dim
        self.use_mi = use_mi

        self.encoder = STBEncoder(input_dim, hidden_dim, dropout)

        # Optional MI estimator
        if use_mi:
            # Compute graph feature dimension
            # (node_feat_dim + 1 for edge density)
            graph_feat_dim = input_dim + 1

            self.mi_estimator = MIEstimator(
                repr_dim=input_dim,
                graph_feat_dim=graph_feat_dim,
                hidden_dim=mi_hidden_dim,
                num_layers=2
            )

            logger.info(f"Initialized STBScorer with MI-based scoring")
        else:
            self.mi_estimator = None
            logger.info(f"Initialized STBScorer with cosine similarity (baseline)")

    def compute_stability_score(
        self,
        original_repr: torch.Tensor,
        perturbed_repr: torch.Tensor,
        graph_summary: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute stability score with worst-case perturbation (paper: inf_{S'∈B_t})

        NEW INTERFACE (2026-03): Supports batched perturbed representations to compute
        worst-case score internally instead of external averaging.

        Args:
            original_repr: [num_items, dim] - original (unperturbed) representations
            perturbed_repr:
                - Old (deprecated): [num_items, dim] - single perturbed repr
                - New: [num_rounds, num_items, dim] - multiple perturbed reprs
            graph_summary: [1, feat_dim] - graph summary (required for MI mode)

        Returns:
            stability_scores: [num_items] - higher = more stable preference
                - MI mode: worst-case (min) MI proxy over perturbation rounds
                - Baseline mode: worst-case (min) cosine similarity over rounds
        """
        # Handle backward compatibility: if 2D input, treat as single round
        if perturbed_repr.dim() == 2:
            perturbed_repr = perturbed_repr.unsqueeze(0)  # [1, num_items, dim]
            logger.warning("compute_stability_score received 2D perturbed_repr, "
                          "treating as single round. Pass 3D [rounds, items, dim] "
                          "for proper worst-case computation.")

        num_rounds, num_items, repr_dim = perturbed_repr.shape

        if self.use_mi:
            if graph_summary is None:
                raise ValueError("graph_summary required for MI-based scoring")

            # ===================================================================
            # MI-based STB approximation (paper Section 3.1.3)
            # ===================================================================
            # Theoretical bound:
            #   STB_{n,t} = inf_{S'∈B_t} [1 - Pr[h([e(S')]_n) != Stable]]
            #   ≤ (I(S'; e(S')) + log 2) / log |Y|
            #
            # Implementation: For each perturbation round, compute per-item MI proxy:
            #   score_n^{(r)} = T(e(S'_r)_n, S) - T(e(S'_shuffle)_n, S)
            # Then take WORST-CASE (min) over rounds to approximate infimum.
            #
            # Higher score → even worst perturbation preserves graph-repr dependency
            #              → stable preference.
            # ===================================================================

            graph_summary_expanded = graph_summary.expand(num_items, -1)  # [num_items, feat_dim]

            with torch.no_grad():
                # Compute per-round MI proxy scores
                round_scores = []  # List of [num_items] tensors

                for round_idx in range(num_rounds):
                    round_repr = perturbed_repr[round_idx]  # [num_items, dim]

                    # Joint: T(e(S'_r)_n, S) — paired representation with graph summary
                    joint_stats = self.mi_estimator(
                        round_repr, graph_summary_expanded
                    ).squeeze(-1)  # [num_items]

                    # Marginal: T(e(S'_shuffle)_n, S) — shuffle to break joint distribution
                    perm = torch.randperm(num_items, device=round_repr.device)
                    marginal_stats = self.mi_estimator(
                        round_repr[perm], graph_summary_expanded
                    ).squeeze(-1)  # [num_items]

                    # Per-item MI proxy for this round
                    round_score = joint_stats - marginal_stats  # [num_items]
                    round_scores.append(round_score)

                # Stack and take WORST-CASE (min) over rounds → approximates inf_{S'∈B_t}
                all_scores = torch.stack(round_scores, dim=0)  # [num_rounds, num_items]
                worst_case_scores = all_scores.min(dim=0).values  # [num_items]

            return worst_case_scores

        else:
            # ===================================================================
            # BASELINE MODE: Cosine similarity (NOT paper-faithful)
            # ===================================================================
            # This is an engineering approximation for comparison/debugging.
            # The paper does NOT define STB via embedding similarity.
            #
            # Still use worst-case (min) over rounds to match the infimum concept,
            # even though the similarity metric itself is not theory-aligned.
            # ===================================================================

            original_norm = F.normalize(original_repr, dim=1)  # [num_items, dim]
            perturbed_norm = F.normalize(perturbed_repr, dim=2)  # [num_rounds, num_items, dim]

            # Cosine similarity for each round: [num_rounds, num_items]
            similarities = (original_norm.unsqueeze(0) * perturbed_norm).sum(dim=2)

            # WORST-CASE (min) over rounds
            worst_case_similarities = similarities.min(dim=0).values  # [num_items]

            return worst_case_similarities

    def compute_graph_summary(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        item_node_indices: Optional[torch.Tensor] = None,
        time_node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute graph summary for MI estimation

        Enhanced (2026-03): Passes item/time node indices to compute separated
        item-side and time-side summaries, better capturing timing dependence
        and co-purchase patterns in the bipartite graph.

        Args:
            node_features: [num_nodes, dim]
            edge_index: [2, num_edges]
            item_node_indices: [num_items] - optional, for better summary computation
            time_node_indices: [num_times] - optional, for better summary computation

        Returns:
            summary: [1, graph_feat_dim] or None if MI estimator disabled
        """
        if self.mi_estimator is None:
            return None

        with torch.no_grad():
            summary = self.mi_estimator.compute_graph_summary(
                node_features,
                edge_index,
                item_node_indices=item_node_indices,
                time_node_indices=time_node_indices
            )

        return summary

    def train_mi_estimator(
        self,
        graph: Dict[str, torch.Tensor],
        perturbation_fn,
        num_steps: int = 100,
        lr: float = 1e-3
    ) -> Dict:
        """
        Train MI estimator on perturbed graphs

        Args:
            graph: Dict with 'node_features', 'edge_index', 'item_node_indices'
            perturbation_fn: Function that perturbs graph
            num_steps: Training steps
            lr: Learning rate

        Returns:
            training_metrics: Dict with loss history
        """
        if not self.use_mi:
            logger.warning("MI estimator not enabled, skipping training")
            return {}

        if self.mi_estimator is None:
            raise ValueError("MI estimator not initialized")

        optimizer = torch.optim.Adam(self.mi_estimator.parameters(), lr=lr)
        loss_history = []

        self.mi_estimator.train()
        self.encoder.eval()  # Freeze encoder during MI training

        logger.info(f"Training MI estimator for {num_steps} steps...")

        for step in range(num_steps):
            optimizer.zero_grad()

            # Get original representations
            with torch.no_grad():
                original_repr = self.encoder(
                    graph['node_features'],
                    graph['edge_index'],
                    graph['item_node_indices']
                )

            # Get graph summary
            with torch.no_grad():
                graph_summary = self.mi_estimator.compute_graph_summary(
                    graph['node_features'],
                    graph['edge_index']
                )

            # Apply perturbation
            perturbed_graph = perturbation_fn(graph)

            # Get perturbed representations
            with torch.no_grad():
                perturbed_repr = self.encoder(
                    perturbed_graph['node_features'],
                    perturbed_graph['edge_index'],
                    graph['item_node_indices']
                )

            # Compute MI loss
            loss, metrics = self.mi_estimator.compute_mi_loss(
                original_repr,
                perturbed_repr,
                graph_summary
            )

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if (step + 1) % 20 == 0:
                logger.info(f"  Step {step+1}/{num_steps}: "
                           f"loss={loss.item():.4f}, "
                           f"mi_est={metrics['mi_estimate']:.4f}")

        self.mi_estimator.eval()

        logger.info(f"MI estimator training complete. "
                   f"Final loss: {loss_history[-1]:.4f}")

        return {
            'loss_history': loss_history,
            'final_loss': loss_history[-1]
        }


if __name__ == '__main__':
    # Test STB encoder
    print("Testing STB Encoder...")

    num_items = 1000
    num_times = 50
    input_dim = 128
    hidden_dim = 512

    # Create dummy graph
    node_features = torch.randn(num_items + num_times, input_dim)
    edge_index = torch.randint(0, num_items + num_times, (2, 5000))
    item_node_indices = torch.arange(num_items)

    # Create encoder
    encoder = STBEncoder(input_dim, hidden_dim)

    # Forward pass
    item_repr = encoder(node_features, edge_index, item_node_indices)

    print(f"Output shape: {item_repr.shape}")
    print(f"Expected: [{num_items}, {input_dim}]")

    assert item_repr.shape == (num_items, input_dim)

    print("\nSTB Encoder test passed!")

    # Test MI Estimator
    print("\n" + "=" * 60)
    print("Testing MI Estimator...")
    print("=" * 60)

    mi_estimator = MIEstimator(
        repr_dim=input_dim,
        graph_feat_dim=input_dim + 1,
        hidden_dim=256,
        num_layers=2
    )

    # Compute graph summary
    graph_summary = mi_estimator.compute_graph_summary(node_features, edge_index)
    print(f"Graph summary shape: {graph_summary.shape}")

    # Forward pass
    stats = mi_estimator(item_repr, graph_summary.expand(num_items, -1))
    print(f"MI statistics shape: {stats.shape}")

    # Test MI loss
    perturbed_repr = torch.randn_like(item_repr)
    loss, metrics = mi_estimator.compute_mi_loss(item_repr, perturbed_repr, graph_summary)

    print(f"MI loss: {loss.item():.4f}")
    print(f"MI estimate: {metrics['mi_estimate']:.4f}")

    print("\nMI Estimator test passed!")

    # Test STB Scorer with MI
    print("\n" + "=" * 60)
    print("Testing STB Scorer (MI-based)...")
    print("=" * 60)

    scorer_mi = STBScorer(input_dim, hidden_dim, use_mi=True, mi_hidden_dim=256)

    # Compute graph summary
    graph_summary = scorer_mi.compute_graph_summary(node_features, edge_index)

    # Compute scores
    original = torch.randn(num_items, input_dim)
    perturbed = torch.randn_like(original)

    scores_mi = scorer_mi.compute_stability_score(original, perturbed, graph_summary)

    print(f"Scores shape: {scores_mi.shape}")
    print(f"Expected: [{num_items}]")
    print(f"Score range: [{scores_mi.min():.4f}, {scores_mi.max():.4f}]")

    assert scores_mi.shape == (num_items,)

    print("\nSTB Scorer (MI) test passed!")

    # Test STB Scorer baseline (cosine similarity)
    print("\n" + "=" * 60)
    print("Testing STB Scorer (Cosine Baseline)...")
    print("=" * 60)

    scorer_baseline = STBScorer(input_dim, hidden_dim, use_mi=False)

    scores_baseline = scorer_baseline.compute_stability_score(original, perturbed)

    print(f"Scores shape: {scores_baseline.shape}")
    print(f"Expected: [{num_items}]")
    print(f"Score range: [{scores_baseline.min():.4f}, {scores_baseline.max():.4f}]")

    assert scores_baseline.shape == (num_items,)

    print("\nSTB Scorer (Baseline) test passed!")

    # Test MI training
    print("\n" + "=" * 60)
    print("Testing MI Estimator Training...")
    print("=" * 60)

    graph = {
        'node_features': node_features,
        'edge_index': edge_index,
        'item_node_indices': item_node_indices
    }

    # Simple perturbation function
    def simple_perturb(g):
        import copy
        perturbed = copy.copy(g)
        perturbed['node_features'] = g['node_features'] + 0.01 * torch.randn_like(g['node_features'])
        return perturbed

    # Train MI estimator
    scorer_mi_train = STBScorer(input_dim, hidden_dim, use_mi=True, mi_hidden_dim=128)

    training_metrics = scorer_mi_train.train_mi_estimator(
        graph=graph,
        perturbation_fn=simple_perturb,
        num_steps=50,
        lr=1e-3
    )

    print(f"Training loss history (first 5): {training_metrics['loss_history'][:5]}")
    print(f"Final training loss: {training_metrics['final_loss']:.4f}")

    print("\nMI Estimator Training test passed!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)