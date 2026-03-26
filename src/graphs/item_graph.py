"""
Item Graph Construction for UPSTAR

Builds in-session and cross-session item graphs for item representation learning.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ItemGraphBuilder:
    """
    Build item graphs from user sequences

    Two types of graphs:
    1. In-session graph: items co-occurring in the same session
    2. Cross-session graph: items from consecutive sessions

    For each graph, if item i appears before item j in any user sequence,
    we create a directed edge i -> j.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = Path(config['dataset']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Session time threshold (seconds)
        self.time_threshold = config['item_repr'].get('time_threshold', 86400)

    def load_sequences(self, split: str = 'train') -> Dict:
        """Load preprocessed sequences"""
        processed_dir = Path(self.config['dataset']['processed_dir'])
        file_path = processed_dir / f'{split}_sequences.pkl'

        with open(file_path, 'rb') as f:
            sequences = pickle.load(f)

        logger.info(f"Loaded {len(sequences)} {split} sequences")
        return sequences

    def build_session_sequences(
        self,
        user_sequences: Dict[int, Dict]
    ) -> Dict[int, List[List[int]]]:
        """
        Split user sequences into sessions based on time gap

        **PAPER-ALIGNED REQUIREMENT**: Requires timestamps in sequence data
        - Uses timestamps to detect session boundaries (time gap > threshold)
        - Essential for Phase 2 in-session/cross-session graph construction

        **BACKWARD COMPATIBILITY**: Supports old format without timestamps
        - If items don't have timestamps, treats entire user history as one session

        Args:
            user_sequences: {user_idx: {'items': [(item, timestamp), ...], 'target': int}}

        Returns:
            user_sessions: {user_idx: [[item1, item2], [item3], ...]}
            Items are extracted from (item, timestamp) tuples.
        """
        logger.info("Building session sequences...")

        user_sessions = {}

        for user_idx, data in user_sequences.items():
            raw_items = data['items']

            if not raw_items:
                user_sessions[user_idx] = []
                continue

            # **FORMAT DETECTION**: Check if items have timestamps
            has_timestamps = isinstance(raw_items[0], tuple)

            if not has_timestamps:
                # **BACKWARD COMPATIBILITY**: No timestamp info — one session per user
                user_sessions[user_idx] = [list(raw_items)]
                continue

            # **PAPER-ALIGNED**: Sort by timestamp ascending before splitting
            sorted_items = sorted(raw_items, key=lambda x: x[1])

            # Split by time gap: gap > threshold => new session
            sessions = []
            current_session = [sorted_items[0][0]]
            for k in range(1, len(sorted_items)):
                item, ts = sorted_items[k]
                prev_ts = sorted_items[k - 1][1]
                if (ts - prev_ts) > self.time_threshold:
                    sessions.append(current_session)
                    current_session = [item]
                else:
                    current_session.append(item)
            sessions.append(current_session)

            user_sessions[user_idx] = sessions

        logger.info(f"Built sessions for {len(user_sessions)} users")
        return user_sessions

    def build_in_session_graph(
        self,
        user_sessions: Dict[int, List[List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Build in-session item graph

        If item i and j appear in the same session (i before j),
        create directed edge i -> j.

        Args:
            user_sessions: {user_idx: [[session1_items], [session2_items], ...]}

        Returns:
            graph_data: {
                'edge_index': [2, num_edges],  # [source, target]
                'edge_type': [num_edges],       # 0 for in-session
                'edge_weight': [num_edges]      # frequency
            }
        """
        logger.info("Building in-session item graph...")

        # Count edges
        edge_counter = Counter()

        for user_idx, sessions in user_sessions.items():
            for session in sessions:
                # Only consecutive purchase transitions: i_(n-1) -> i_n
                for i in range(len(session) - 1):
                    src, dst = session[i], session[i + 1]
                    edge_counter[(src, dst)] += 1

        # Build edge index
        if len(edge_counter) == 0:
            logger.warning("No in-session edges found!")
            return self._empty_graph()

        edges = list(edge_counter.keys())
        weights = [edge_counter[e] for e in edges]

        edge_index = torch.tensor(edges, dtype=torch.long).T  # [2, num_edges]
        edge_type = torch.zeros(len(edges), dtype=torch.long)  # 0 for in-session
        edge_weight = torch.tensor(weights, dtype=torch.float)

        # Make bidirectional (for undirected message passing)
        # Or keep directed for directed message passing
        # Here we keep directed as per paper

        logger.info(f"In-session graph: {len(edges)} edges")
        logger.info(f"  Unique edges: {len(set(edges))}")

        return {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'edge_weight': edge_weight
        }

    def build_cross_session_graph(
        self,
        user_sessions: Dict[int, List[List[int]]],
        window: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Build cross-session item graph

        If item i from session t and item j from session t+window,
        create directed edge i -> j.

        Args:
            user_sessions: {user_idx: [[session1], [session2], ...]}
            window: session window (default: 1 for consecutive sessions)

        Returns:
            graph_data: same format as in-session graph
        """
        logger.info("Building cross-session item graph...")

        edge_counter = Counter()

        for user_idx, sessions in user_sessions.items():
            # Connect items from consecutive sessions
            for t in range(len(sessions) - window):
                session_t = sessions[t]
                session_t_next = sessions[t + window]

                # Only the boundary transition: last item of session t -> first item of session t+1
                if session_t and session_t_next:
                    edge_counter[(session_t[-1], session_t_next[0])] += 1

        if len(edge_counter) == 0:
            logger.warning("No cross-session edges found!")
            return self._empty_graph()

        edges = list(edge_counter.keys())
        weights = [edge_counter[e] for e in edges]

        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_type = torch.ones(len(edges), dtype=torch.long)  # 1 for cross-session
        edge_weight = torch.tensor(weights, dtype=torch.float)

        logger.info(f"Cross-session graph: {len(edges)} edges")

        return {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'edge_weight': edge_weight
        }

    def build_combined_graph(
        self,
        in_session_graph: Dict,
        cross_session_graph: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Combine in-session and cross-session graphs

        Returns:
            combined_graph with edge_type distinguishing the two
        """
        logger.info("Combining item graphs...")

        in_edges = in_session_graph['edge_index']
        cross_edges = cross_session_graph['edge_index']

        in_types = in_session_graph['edge_type']
        cross_types = cross_session_graph['edge_type']

        in_weights = in_session_graph['edge_weight']
        cross_weights = cross_session_graph['edge_weight']

        # Concatenate
        edge_index = torch.cat([in_edges, cross_edges], dim=1)
        edge_type = torch.cat([in_types, cross_types], dim=0)
        edge_weight = torch.cat([in_weights, cross_weights], dim=0)

        logger.info(f"Combined graph: {edge_index.shape[1]} total edges")
        logger.info(f"  In-session: {in_edges.shape[1]}")
        logger.info(f"  Cross-session: {cross_edges.shape[1]}")

        return {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'edge_weight': edge_weight
        }

    def compute_graph_statistics(
        self,
        graph: Dict[str, torch.Tensor],
        num_items: int
    ) -> Dict[str, float]:
        """
        Compute graph statistics

        Returns:
            stats: {
                'num_nodes': int,
                'num_edges': int,
                'avg_in_degree': float,
                'avg_out_degree': float,
                'max_in_degree': int,
                'max_out_degree': int
            }
        """
        edge_index = graph['edge_index']

        num_nodes = num_items
        num_edges = edge_index.shape[1]

        # Compute degrees
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()

        out_degrees = Counter(src_nodes)
        in_degrees = Counter(dst_nodes)

        avg_out_degree = num_edges / num_nodes if num_nodes > 0 else 0
        avg_in_degree = num_edges / num_nodes if num_nodes > 0 else 0

        max_out_degree = max(out_degrees.values()) if out_degrees else 0
        max_in_degree = max(in_degrees.values()) if in_degrees else 0

        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree,
            'max_in_degree': max_in_degree,
            'max_out_degree': max_out_degree
        }

        return stats

    def save_graph(self, graph: Dict, filename: str):
        """Save graph to cache"""
        file_path = self.cache_dir / filename
        torch.save(graph, file_path)
        logger.info(f"Saved graph to {file_path}")

    def load_graph(self, filename: str) -> Optional[Dict]:
        """Load graph from cache"""
        file_path = self.cache_dir / filename
        if file_path.exists():
            graph = torch.load(file_path)
            logger.info(f"Loaded graph from {file_path}")
            return graph
        return None

    def _empty_graph(self) -> Dict[str, torch.Tensor]:
        """Return empty graph"""
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long),
            'edge_type': torch.empty((0,), dtype=torch.long),
            'edge_weight': torch.empty((0,), dtype=torch.float)
        }


if __name__ == '__main__':
    import yaml

    # Test item graph building
    with open('configs/item_repr.yaml', 'r') as f:
        config = yaml.safe_load(f)

    builder = ItemGraphBuilder(config)

    # Load sequences
    train_sequences = builder.load_sequences('train')

    # Build sessions
    user_sessions = builder.build_session_sequences(train_sequences)

    # Build graphs
    in_session_graph = builder.build_in_session_graph(user_sessions)
    cross_session_graph = builder.build_cross_session_graph(user_sessions)

    # Combine
    combined_graph = builder.build_combined_graph(in_session_graph, cross_session_graph)

    # Load metadata
    with open(Path(config['dataset']['processed_dir']) / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    num_items = metadata['num_items']

    # Statistics
    stats = builder.compute_graph_statistics(combined_graph, num_items)

    print("\n" + "=" * 60)
    print("Item Graph Statistics")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key:20s}: {value}")

    # Save
    builder.save_graph(combined_graph, 'item_graph.pt')
    builder.save_graph(in_session_graph, 'in_session_graph.pt')
    builder.save_graph(cross_session_graph, 'cross_session_graph.pt')