"""
Item-GNN for learning item representations

Implements 1-layer GNN with remember gate for combining
node self-representation and neighborhood aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RememberGate(nn.Module):
    """
    Remember gate for controlling information flow

    r = sigmoid(W_r * h_self + U_r * h_neighbors + b_r)
    h_final = r * h_neighbors + (1 - r) * h_self

    This allows the model to balance between:
    - Keeping self information (1-r) * h_self
    - Accepting neighborhood information r * h_neighbors
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim

        # Linear transformations
        self.W_self = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_neighbors = nn.Linear(embed_dim, embed_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(embed_dim))

    def forward(
        self,
        h_self: torch.Tensor,
        h_neighbors: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_self: [num_nodes, embed_dim] - node self-representation
            h_neighbors: [num_nodes, embed_dim] - aggregated neighborhood

        Returns:
            h_final: [num_nodes, embed_dim] - final representation
        """
        # Compute gate
        gate = torch.sigmoid(self.W_self(h_self) + self.W_neighbors(h_neighbors) + self.b)

        # Combine
        h_final = gate * h_neighbors + (1 - gate) * h_self

        return h_final


class ItemGNNLayer(nn.Module):
    """
    Single GNN layer for item graph

    Distinguishes between in-neighbors and out-neighbors with separate
    transformation matrices, as per paper Section 3.2.

    Aggregates: h_n^{g(k)} = ReLU(W_in * sum_in + W_out * sum_out)
    Then combines with self using remember gate.
    """

    def __init__(
        self,
        embed_dim: int,
        use_remember_gate: bool = True,
        aggregation: str = 'sum'
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_remember_gate = use_remember_gate
        self.aggregation = aggregation

        # Remember gate
        if use_remember_gate:
            self.remember_gate = RememberGate(embed_dim)

        # Self transformation
        self.self_transform = nn.Linear(embed_dim, embed_dim)

        # Separate transformations for in/out neighbors (per paper)
        self.W_in = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)

        logger.info(f"Initialized ItemGNNLayer: "
                   f"dim={embed_dim}, remember_gate={use_remember_gate}, "
                   f"agg={aggregation}, separate_in_out=True")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, embed_dim] - node features
            edge_index: [2, num_edges] - edge indices (directed)
            edge_weight: [num_edges] - optional edge weights

        Returns:
            out: [num_nodes, embed_dim] - updated node features
        """
        num_nodes = x.shape[0]

        # Transform self
        h_self = self.self_transform(x)

        # Aggregate in-neighbors and out-neighbors separately
        h_in_agg = self._aggregate_in_neighbors(x, edge_index, edge_weight)
        h_out_agg = self._aggregate_out_neighbors(x, edge_index, edge_weight)

        # Apply separate transformations with ReLU (per paper Section 3.2 formula)
        h_agg = F.relu(self.W_in(h_in_agg) + self.W_out(h_out_agg))

        # Combine with remember gate
        if self.use_remember_gate:
            out = self.remember_gate(h_self, h_agg)
        else:
            out = h_agg + h_self

        return out

    def _aggregate_in_neighbors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate in-neighbors (edges pointing TO the node)

        For edge (src, dst), we aggregate src features into dst.
        In-neighbors of node n are all src where (src, n) is an edge.

        Args:
            x: [num_nodes, embed_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]

        Returns:
            aggregated: [num_nodes, embed_dim]
        """
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Weight messages by edge_weight if provided, else uniform weight=1
        if edge_weight is not None:
            # [num_edges, 1] for broadcasting with [num_edges, embed_dim]
            w = edge_weight.unsqueeze(-1)
            messages = x[src] * w  # weighted source features
        else:
            messages = x[src]

        # SUM aggregation: h_in(n) = SUM_{p in N_in(n)} h(p)  (paper formula)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)

        if self.aggregation == 'mean':
            in_degree = torch.zeros(num_nodes, device=x.device)
            in_degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            in_degree = torch.clamp(in_degree, min=1).unsqueeze(-1)
            aggregated = aggregated / in_degree

        return aggregated

    def _aggregate_out_neighbors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate out-neighbors (edges pointing FROM the node)

        For edge (src, dst), we aggregate dst features into src.
        Out-neighbors of node n are all dst where (n, dst) is an edge.

        Args:
            x: [num_nodes, embed_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]

        Returns:
            aggregated: [num_nodes, embed_dim]
        """
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Weight messages by edge_weight if provided, else uniform weight=1
        if edge_weight is not None:
            w = edge_weight.unsqueeze(-1)
            messages = x[dst] * w  # weighted destination features
        else:
            messages = x[dst]

        # SUM aggregation: h_out(n) = SUM_{q in N_out(n)} h(q)  (paper formula)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, src, messages)

        if self.aggregation == 'mean':
            out_degree = torch.zeros(num_nodes, device=x.device)
            out_degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
            out_degree = torch.clamp(out_degree, min=1).unsqueeze(-1)
            aggregated = aggregated / out_degree

        return aggregated


class ItemGNN(nn.Module):
    """
    Item-GNN for learning item representations

    Architecture:
    1. Item embedding layer
    2. 1-layer GNN with remember gate
    3. Output projection

    Paper: Section 3.2
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        num_layers: int = 1,
        use_remember_gate: bool = True,
        aggregation: str = 'sum',
        dropout: float = 0.1,
        item_feature_dim: Optional[int] = None
    ):
        super().__init__()

        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.item_feature_dim = item_feature_dim

        if item_feature_dim is not None:
            # Feature projector: intrinsic attributes -> embed_dim
            self.feat_proj = nn.Linear(item_feature_dim, embed_dim)
            self.item_embedding = None
        else:
            # Fallback: ID embedding
            self.feat_proj = None
            self.item_embedding = nn.Embedding(
                num_embeddings=num_items,
                embedding_dim=embed_dim,
                padding_idx=0
            )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                ItemGNNLayer(
                    embed_dim=embed_dim,
                    use_remember_gate=use_remember_gate,
                    aggregation=aggregation
                )
            )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection (optional, for link prediction)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        logger.info(f"Initialized ItemGNN: "
                   f"num_items={num_items}, embed_dim={embed_dim}, "
                   f"num_layers={num_layers}, remember_gate={use_remember_gate}, "
                   f"init={'attr_features' if item_feature_dim is not None else 'id_embedding'}")

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            edge_index: [2, num_edges] - graph structure
            edge_weight: [num_edges] - optional edge weights
            item_features: [num_items, item_feature_dim] - optional intrinsic attributes

        Returns:
            item_embeddings: [num_items, embed_dim]
        """
        # Initialize from intrinsic attributes or ID embedding
        if item_features is not None and self.feat_proj is not None:
            x = self.feat_proj(item_features)  # [num_items, embed_dim]
        else:
            x = self.item_embedding.weight  # [num_items, embed_dim]

        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index, edge_weight)

            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)

        # Optional output projection
        x = self.output_proj(x)

        return x

    def get_item_embeddings(self) -> torch.Tensor:
        """
        Get current item embeddings (ORIGINAL EMBEDDING TABLE - NOT GRAPH ENHANCED)

        ⚠️ WARNING: This returns the pre-GNN embedding table weights.
        For graph-enhanced embeddings, use get_final_item_embeddings() instead.

        Returns:
            embeddings: [num_items, embed_dim] - ORIGINAL embedding table (not graph-enhanced)
        """
        if self.item_embedding is not None:
            return self.item_embedding.weight.detach()
        # Feature mode: no stored embedding weight; caller must use forward() output
        raise RuntimeError(
            "get_item_embeddings() is not available in feature mode. "
            "Call forward(edge_index) and cache the result instead."
        )

    def get_final_item_embeddings(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get FINAL item embeddings AFTER GNN message passing (GRAPH-ENHANCED).

        This is the RECOMMENDED method for getting item representations that
        include graph structural information through GNN propagation.

        Args:
            edge_index: [2, num_edges] - graph structure
            edge_weight: [num_edges] - optional edge weights

        Returns:
            final_embeddings: [num_items, embed_dim] - GRAPH-ENHANCED embeddings

        Example:
            >>> model = ItemGNN(num_items=1000, embed_dim=128)
            >>> final_embeddings = model.get_final_item_embeddings(edge_index, edge_weight)
            >>> # Use these for STB / UPSTAR
        """
        with torch.no_grad():
            # Forward pass: includes GNN message passing
            final_embeddings = self.forward(edge_index, edge_weight)

        return final_embeddings.detach()

    def predict_links(
        self,
        node_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link existence for node pairs

        Args:
            node_pairs: [num_pairs, 2] - (source, target) pairs

        Returns:
            scores: [num_pairs] - link scores
        """
        with torch.no_grad():
            if self.item_embedding is not None:
                embeddings = self.item_embedding.weight
            else:
                raise RuntimeError(
                    "predict_links() requires a cached embedding in feature mode. "
                    "Pass embeddings explicitly."
                )
            src_emb = embeddings[node_pairs[:, 0]]
            dst_emb = embeddings[node_pairs[:, 1]]

            # Dot product as link score
            scores = (src_emb * dst_emb).sum(dim=1)

        return scores


if __name__ == '__main__':
    # Test Item-GNN
    print("Testing Item-GNN...")

    num_items = 1000
    embed_dim = 128

    model = ItemGNN(
        num_items=num_items,
        embed_dim=embed_dim,
        num_layers=1,
        use_remember_gate=True
    )

    # Create dummy graph
    edge_index = torch.randint(0, num_items, (2, 5000))
    edge_weight = torch.ones(5000)

    # Forward pass
    embeddings = model(edge_index, edge_weight)

    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: [{num_items}, {embed_dim}]")

    assert embeddings.shape == (num_items, embed_dim)

    print("\nItem-GNN test passed!")

    # Test remember gate
    print("\nTesting Remember Gate...")
    h_self = torch.randn(100, embed_dim)
    h_neighbors = torch.randn(100, embed_dim)

    gate = RememberGate(embed_dim)
    h_final = gate(h_self, h_neighbors)

    print(f"Input shape: {h_self.shape}")
    print(f"Output shape: {h_final.shape}")

    assert h_final.shape == (100, embed_dim)

    print("Remember Gate test passed!")