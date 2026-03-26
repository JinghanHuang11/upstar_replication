"""
Attributed Item-Time Graph Construction

Builds a bipartite graph with item nodes and time nodes.
Edges connect items to time nodes when purchases occur.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ItemTimeGraphBuilder:
    """
    Build attributed item-time graph

    Graph structure:
    - Item nodes: num_items
    - Time nodes: num_unique_days
    - Edges: item-time connections (when item was purchased)

    Node features:
    - Item nodes: from Phase 2 item embeddings (128-dim)
    - Time nodes: zero vectors (128-dim)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = Path(config['dataset']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Time granularity: day-level
        self.time_granularity = config['stb'].get('time_granularity', 'day')

    def load_sequences(self, split: str = 'train') -> Dict:
        """Load preprocessed sequences"""
        processed_dir = Path(self.config['dataset']['processed_dir'])
        file_path = processed_dir / f'{split}_sequences.pkl'

        with open(file_path, 'rb') as f:
            sequences = pickle.load(f)

        logger.info(f"Loaded {len(sequences)} {split} sequences")
        return sequences

    def load_item_embeddings(self) -> torch.Tensor:
        """Load item embeddings from Phase 2"""
        from src.models.load_item_embeddings import get_item_embeddings

        try:
            embeddings = get_item_embeddings(self.config)
            logger.info(f"Loaded item embeddings: {embeddings.shape}")
            return embeddings
        except FileNotFoundError:
            logger.error("Item embeddings not found. Please run Phase 2 first.")
            raise

    def build_time_nodes(
        self,
        user_sequences: Dict[int, Dict]
    ) -> Dict[str, int]:
        """
        Build time nodes (day-level)

        Args:
            user_sequences: {user_idx: {'items': [(item, timestamp), ...], 'target': int}}

        Returns:
            time2idx: mapping from date string to time node index
        """
        logger.info("Building time nodes (day-level)...")

        # Collect all unique day-level dates
        all_days = set()
        for user_idx, data in user_sequences.items():
            items_with_ts = data['items']
            for item, ts in items_with_ts:
                # Convert Unix timestamp to date string
                day_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                all_days.add(day_str)

        # Sort days chronologically and assign indices
        sorted_days = sorted(all_days)
        time2idx = {day: i for i, day in enumerate(sorted_days)}

        logger.info(f"Built {len(time2idx)} day-level time nodes")

        return time2idx

    def build_item_time_graph(
        self,
        user_sequences: Dict[int, Dict],
        item_embeddings: torch.Tensor,
        time2idx: Optional[Dict[str, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Build item-time bipartite graph

        Args:
            user_sequences: user sequences with format {user_idx: {'items': [(item, timestamp), ...]}}
            item_embeddings: [num_items, embed_dim]
            time2idx: time node mapping from date string to index (optional, will build if None)

        Returns:
            graph_data: {
                'edge_index': [2, num_edges],  # [item_idx, time_idx]
                'node_features': [num_nodes, embed_dim],
                'item_node_indices': [num_items],
                'time_node_indices': [num_times],
                'edge_type': [num_edges]  # 0 for item->time
            }
        """
        logger.info("Building item-time graph...")

        num_items = item_embeddings.shape[0]

        # Build time nodes if not provided
        if time2idx is None:
            time2idx = self.build_time_nodes(user_sequences)

        num_times = len(time2idx)

        # Build edges
        edges = []
        for user_idx, data in user_sequences.items():
            items_with_ts = data['items']

            # Create item-time edges
            # Each (item, timestamp) pair connects the item to its purchase day
            for item, ts in items_with_ts:
                day_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                time_node_idx = num_items + time2idx[day_str]  # time nodes are after item nodes
                edges.append((item, time_node_idx))

        # Convert to tensor
        if len(edges) == 0:
            logger.warning("No edges found!")
            return self._empty_graph(item_embeddings, num_times)

        edge_index = torch.tensor(edges, dtype=torch.long).T

        # Node features
        # Item nodes: use embeddings
        # Time nodes: zero vectors
        item_features = item_embeddings
        time_features = torch.zeros(num_times, item_embeddings.shape[1], device=item_embeddings.device)

        node_features = torch.cat([item_features, time_features], dim=0)

        # Node indices
        item_node_indices = torch.arange(num_items, dtype=torch.long)
        time_node_indices = torch.arange(num_items, num_items + num_times, dtype=torch.long)

        logger.info(f"Built item-time graph:")
        logger.info(f"  Item nodes: {num_items}")
        logger.info(f"  Time nodes: {num_times}")
        logger.info(f"  Total nodes: {num_items + num_times}")
        logger.info(f"  Edges: {len(edges)}")

        # VALIDATION: Ensure bipartite structure (paper Section 3.1.3)
        # Edge sources must be item nodes, targets must be time nodes
        src_nodes, tgt_nodes = edge_index[0], edge_index[1]
        assert (src_nodes >= 0).all() and (src_nodes < num_items).all(), \
            "Edge sources must be item nodes [0, num_items)"
        assert (tgt_nodes >= num_items).all() and (tgt_nodes < num_items + num_times).all(), \
            "Edge targets must be time nodes [num_items, num_items+num_times)"

        logger.info("✓ Bipartite structure validated: item→time edges only")

        return {
            'edge_index': edge_index,
            'node_features': node_features,
            'item_node_indices': item_node_indices,
            'time_node_indices': time_node_indices,
            # Note: edge_type removed (not needed for bipartite graph in paper)
            'num_items': num_items,
            'num_times': num_times
        }

    def compute_graph_statistics(
        self,
        graph: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute graph statistics"""
        edge_index = graph['edge_index']
        num_items = graph['num_items']
        num_times = graph['num_times']

        num_edges = edge_index.shape[1]

        # Item degree (how many times each item appears)
        item_degrees = torch.bincount(edge_index[0], minlength=num_items)

        # Time degree (how many items per time)
        time_degrees = torch.bincount(edge_index[1], minlength=num_times)

        stats = {
            'num_nodes': num_items + num_times,
            'num_edges': num_edges,
            'num_item_nodes': num_items,
            'num_time_nodes': num_times,
            'avg_item_degree': item_degrees.float().mean().item(),
            'avg_time_degree': time_degrees.float().mean().item(),
            'max_item_degree': item_degrees.max().item() if len(item_degrees) > 0 else 0,
            'max_time_degree': time_degrees.max().item() if len(time_degrees) > 0 else 0
        }

        return stats

    def save_graph(self, graph: Dict, filename: str):
        """Save graph to cache"""
        file_path = self.cache_dir / filename
        torch.save(graph, file_path)
        logger.info(f"Saved item-time graph to {file_path}")

    def load_graph(self, filename: str) -> Optional[Dict]:
        """Load graph from cache"""
        file_path = self.cache_dir / filename
        if file_path.exists():
            graph = torch.load(file_path)
            logger.info(f"Loaded item-time graph from {file_path}")
            return graph
        return None

    def _empty_graph(self, item_embeddings: torch.Tensor, num_times: int) -> Dict:
        """Return empty graph (bipartite, no edges)"""
        num_items = item_embeddings.shape[0]
        device = item_embeddings.device

        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long, device=device),
            'node_features': torch.cat([
                item_embeddings,
                torch.zeros(num_times, item_embeddings.shape[1], device=device)
            ], dim=0),
            'item_node_indices': torch.arange(num_items, dtype=torch.long, device=device),
            'time_node_indices': torch.arange(num_items, num_items + num_times, dtype=torch.long, device=device),
            # Note: edge_type removed (not needed for bipartite graph in paper)
            'num_items': num_items,
            'num_times': num_times
        }


if __name__ == '__main__':
    import yaml

    # Test item-time graph building
    with open('configs/stb.yaml', 'r') as f:
        config = yaml.safe_load(f)

    builder = ItemTimeGraphBuilder(config)

    # Load sequences
    train_sequences = builder.load_sequences('train')

    # Load item embeddings
    item_embeddings = builder.load_item_embeddings()

    # Build graph
    graph = builder.build_item_time_graph(train_sequences, item_embeddings)

    # Statistics
    stats = builder.compute_graph_statistics(graph)

    print("\n" + "=" * 60)
    print("Item-Time Graph Statistics")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key:20s}: {value}")

    # Save
    builder.save_graph(graph, 'item_time_graph.pt')