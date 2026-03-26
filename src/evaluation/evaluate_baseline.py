"""
Clean Evaluation for Baseline LSTM

This module fixes the candidate set issue:
- Excludes padding token (index=0) from top-k
- Properly handles target indices
- Supports dynamic vocabulary size for CV
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CleanEvaluator:
    """
    Clean evaluator that properly handles candidate set.

    Key fixes:
    1. Excludes padding@0 from top-k selection
    2. Handles target indices correctly (subtract 1 for removed padding)
    3. Supports dynamic vocabulary size
    """

    def __init__(
        self,
        num_items: int,
        padding_idx: int = 0,
        exclude_padding: bool = True
    ):
        """
        Args:
            num_items: number of actual items (excluding padding)
            padding_idx: index of padding token (default 0)
            exclude_padding: whether to exclude padding from top-k
        """
        self.num_items = num_items
        self.padding_idx = padding_idx
        self.exclude_padding = exclude_padding

    def get_candidates(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get candidate logits for ranking.

        Args:
            logits: [batch_size, num_items + 1] raw logits

        Returns:
            candidates: [batch_size, num_items] candidate logits (excludes padding)
        """
        if self.exclude_padding:
            # Remove padding at index 0
            # logits[:, 0] is padding
            # logits[:, 1:num_items+1] are valid items
            return logits[:, 1:]
        else:
            return logits

    def compute_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        k_values: List[int] = [1, 5, 10, 15, 20, 50]
    ) -> Dict[str, float]:
        """
        Compute all metrics with clean candidate set.

        Args:
            logits: [batch_size, num_items + 1] raw logits from model
            targets: [batch_size] target indices (0-indexed, assuming padding at 0)
            k_values: list of k values

        Returns:
            metrics: dict of metric names and values (0-1 scale)
        """
        # Adjust targets to align with candidate indices
        adjusted_targets = targets.clone()
        if self.exclude_padding:
            # logits[:, 1:] are valid items (indices 1..num_items in logits)
            # candidates are reindexed to 0..num_items-1
            # targets are in range [1, num_items], so subtract 1 to match
            adjusted_targets = targets - 1

        # Get candidate logits
        candidates = self.get_candidates(logits)

        metrics = {}

        for k in k_values:
            # Precision@K
            metrics[f'Precision@{k}'] = self._precision_at_k(candidates, adjusted_targets, k)

            # NDCG@K
            metrics[f'NDCG@{k}'] = self._ndcg_at_k(candidates, adjusted_targets, k)

            # MRR@K
            metrics[f'MRR@{k}'] = self._mrr_at_k(candidates, adjusted_targets, k)

        return metrics

    def _precision_at_k(
        self,
        candidates: torch.Tensor,
        targets: torch.Tensor,
        k: int
    ) -> float:
        """Compute Precision@K"""
        batch_size = candidates.shape[0]

        # Get top-k predictions
        _, top_k_indices = torch.topk(candidates, min(k, candidates.shape[1]), dim=1)

        # Check if targets are in top-k
        targets_expanded = targets.unsqueeze(1).expand(-1, top_k_indices.shape[1])
        hits = (top_k_indices == targets_expanded).any(dim=1).float()

        return hits.mean().item()

    def _ndcg_at_k(
        self,
        candidates: torch.Tensor,
        targets: torch.Tensor,
        k: int
    ) -> float:
        """Compute NDCG@K"""
        batch_size = candidates.shape[0]
        k = min(k, candidates.shape[1])

        # Get top-k predictions
        _, top_k_indices = torch.topk(candidates, k, dim=1)

        # Find position of target in top-k
        targets_expanded = targets.unsqueeze(1).expand(-1, k)
        hit_mask = (top_k_indices == targets_expanded)

        # Get ranks (1-indexed) where hit
        ranks = torch.arange(1, k + 1, device=candidates.device, dtype=torch.float)
        hit_ranks = (ranks * hit_mask.float()).max(dim=1)[0]

        # Compute DCG
        dcg = torch.where(
            hit_mask.any(dim=1),
            1.0 / torch.log2(hit_ranks + 1.0),
            torch.zeros(batch_size, device=candidates.device)
        )

        # Ideal DCG = 1 (target at position 1)
        idcg = torch.ones(batch_size, device=candidates.device)

        return (dcg / idcg).mean().item()

    def _mrr_at_k(
        self,
        candidates: torch.Tensor,
        targets: torch.Tensor,
        k: int
    ) -> float:
        """Compute MRR@K"""
        batch_size = candidates.shape[0]
        k = min(k, candidates.shape[1])

        # Get top-k predictions
        _, top_k_indices = torch.topk(candidates, k, dim=1)

        # Find position of target in top-k
        targets_expanded = targets.unsqueeze(1).expand(-1, k)
        hit_mask = (top_k_indices == targets_expanded)

        # Get ranks (1-indexed) where hit
        ranks = torch.arange(1, k + 1, device=candidates.device, dtype=torch.float)
        hit_ranks = (ranks * hit_mask.float()).max(dim=1)[0]

        # Compute reciprocal rank
        rr = torch.where(
            hit_mask.any(dim=1),
            1.0 / hit_ranks,
            torch.zeros(batch_size, device=candidates.device)
        )

        return rr.mean().item()


def evaluate_model(
    model: nn.Module,
    data_loader,
    num_items: int,
    device: torch.device,
    k_values: List[int] = [1, 5, 10, 15, 20, 50],
    exclude_padding: bool = True
) -> Dict[str, float]:
    """
    Evaluate model with clean candidate set.

    Args:
        model: LSTM model
        data_loader: data loader
        num_items: number of items (excluding padding)
        device: device
        k_values: k values for metrics
        exclude_padding: whether to exclude padding from top-k

    Returns:
        metrics: dict of metrics
    """
    model.eval()

    evaluator = CleanEvaluator(
        num_items=num_items,
        padding_idx=0,
        exclude_padding=exclude_padding
    )

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            items = batch['items'].to(device)
            seq_lengths = batch['seq_length'].to(device)
            targets = batch['target'].to(device)

            # Forward pass
            logits = model(items, seq_lengths)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Concatenate
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics with clean candidate set
    metrics = evaluator.compute_metrics(all_logits, all_targets, k_values)

    return metrics


def print_metrics(
    metrics: Dict[str, float],
    title: str = "Metrics",
    percentage: bool = True
):
    """Print metrics in a nice format"""
    logger.info(f"\n{title}")
    logger.info("=" * 60)

    # Group by k
    k_values = sorted(set([int(k.split('@')[1]) for k in metrics.keys()]))

    for k in k_values:
        logger.info(f"@{k}:")
        for metric_type in ['Precision', 'NDCG', 'MRR']:
            key = f'{metric_type}@{k}'
            if key in metrics:
                value = metrics[key]
                if percentage:
                    logger.info(f"  {metric_type:10s}: {value * 100:6.2f}%")
                else:
                    logger.info(f"  {metric_type:10s}: {value:8.4f}")

    logger.info("=" * 60)


def print_main_results(
    metrics: Dict[str, float],
    title: str = "Main Results"
):
    """Print main results table"""
    logger.info(f"\n{title}")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<15} {'Value':<15}")
    logger.info("-" * 60)

    main_metrics = [
        'Precision@5', 'Precision@20',
        'NDCG@5', 'NDCG@20',
        'MRR@5', 'MRR@20'
    ]

    for metric_name in main_metrics:
        if metric_name in metrics:
            value = metrics[metric_name]
            logger.info(f"{metric_name:<15} {value * 100:6.2f}%")

    logger.info("=" * 60)


if __name__ == '__main__':
    print("Testing CleanEvaluator...")

    # Create dummy data
    batch_size = 100
    num_items = 1000

    logits = torch.randn(batch_size, num_items + 1)  # +1 for padding
    targets = torch.randint(1, num_items + 1, (batch_size,))  # Never 0

    # Test evaluator
    evaluator = CleanEvaluator(num_items=num_items, exclude_padding=True)

    metrics = evaluator.compute_metrics(logits, targets, k_values=[1, 5, 10, 20, 50])

    print("\nMetrics with clean candidate set:")
    for k in [5, 20]:
        print(f"  P@{k}: {metrics[f'Precision@{k}'] * 100:.2f}%")
        print(f"  N@{k}: {metrics[f'NDCG@{k}'] * 100:.2f}%")
        print(f"  M@{k}: {metrics[f'MRR@{k}'] * 100:.2f}%")

    print("\nCleanEvaluator test passed!")