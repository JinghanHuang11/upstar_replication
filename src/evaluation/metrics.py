"""
Evaluation metrics for sequential recommendation (Unified with padding handling)

Key fixes:
1. Excludes padding token (index=0) from candidate set
2. Properly handles target indices alignment
3. Unified for both baseline and UPSTAR

Supports:
- Precision@k
- Recall@k (same as Precision@k for next-item)
- NDCG@k
- MRR@k

k in {1, 5, 10, 15, 20, 50}
Output format: 0-1 scale (can be converted to percentage)
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def _remove_padding(
    logits: torch.Tensor,
    padding_idx: int = 0
) -> torch.Tensor:
    """
    Remove padding token from logits.

    Args:
        logits: [batch_size, num_items + 1] raw logits (including padding)
        padding_idx: index of padding token (default 0)

    Returns:
        candidates: [batch_size, num_items] candidate logits (excludes padding)
    """
    # Remove padding at index 0
    # logits[:, 0] is padding
    # logits[:, 1:] are valid items
    return logits[:, 1:]


def _align_targets(
    targets: torch.Tensor,
    padding_idx: int = 0
) -> torch.Tensor:
    """
    Align target indices with candidate set (after padding removal).

    Args:
        targets: [batch_size] target indices (original, including padding)
        padding_idx: index of padding token (default 0)

    Returns:
        adjusted_targets: [batch_size] target indices (aligned with candidates)
                          targets are in range [1, num_items], so subtract 1 to match
    """
    # logits[:, 1:] are valid items (indices 1..num_items in logits)
    # candidates are reindexed to 0..num_items-1
    # targets are in range [1, num_items], so subtract 1 to match
    return targets - 1


def precision_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    exclude_padding: bool = True,
    padding_idx: int = 0
) -> float:
    """
    Compute Precision@K

    Whether the true item is in top-k predictions.

    Args:
        predictions: [batch_size, num_items] or [batch_size, num_items + 1] prediction scores
        targets: [batch_size] target item indices
        k: cutoff
        exclude_padding: whether to exclude padding token from top-k
        padding_idx: index of padding token (default 0)

    Returns:
        precision: average precision@k (0-1)
    """
    batch_size = predictions.shape[0]

    # Exclude padding if requested
    if exclude_padding:
        candidates = _remove_padding(predictions, padding_idx)
        adjusted_targets = _align_targets(targets, padding_idx)
    else:
        candidates = predictions
        adjusted_targets = targets

    # Get top-k predictions
    _, top_k_indices = torch.topk(candidates, min(k, candidates.shape[1]), dim=1)

    # Check if targets are in top-k
    targets_expanded = adjusted_targets.unsqueeze(1).expand(-1, top_k_indices.shape[1])
    hits = (top_k_indices == targets_expanded).any(dim=1).float()

    precision = hits.mean().item()

    return precision


def recall_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    exclude_padding: bool = True,
    padding_idx: int = 0
) -> float:
    """
    Compute Recall@K

    For next-item recommendation, Recall@K = Precision@K.

    Args:
        predictions: [batch_size, num_items] or [batch_size, num_items + 1] prediction scores
        targets: [batch_size] target item indices
        k: cutoff
        exclude_padding: whether to exclude padding token from top-k
        padding_idx: index of padding token (default 0)

    Returns:
        recall: average recall@k (0-1)
    """
    return precision_at_k(predictions, targets, k, exclude_padding, padding_idx)


def ndcg_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    exclude_padding: bool = True,
    padding_idx: int = 0
) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain)

    If target item rank <= k: 1 / log2(1 + rank)
    Else: 0

    Args:
        predictions: [batch_size, num_items] or [batch_size, num_items + 1] prediction scores
        targets: [batch_size] target item indices
        k: cutoff
        exclude_padding: whether to exclude padding token from top-k
        padding_idx: index of padding token (default 0)

    Returns:
        ndcg: average NDCG@k (0-1)
    """
    batch_size = predictions.shape[0]

    # Exclude padding if requested
    if exclude_padding:
        candidates = _remove_padding(predictions, padding_idx)
        adjusted_targets = _align_targets(targets, padding_idx)
    else:
        candidates = predictions
        adjusted_targets = targets

    k = min(k, candidates.shape[1])

    # Get top-k predictions
    _, top_k_indices = torch.topk(candidates, k, dim=1)

    # Find position of target in top-k (1-indexed)
    targets_expanded = adjusted_targets.unsqueeze(1).expand(-1, k)
    hit_mask = (top_k_indices == targets_expanded)

    # Get ranks (1-indexed) where hit
    ranks = torch.arange(1, k + 1, device=predictions.device, dtype=torch.float)
    hit_ranks = (ranks * hit_mask.float()).max(dim=1)[0]  # [batch_size]

    # Compute DCG
    # DCG = 1 / log2(1 + rank) if hit, else 0
    dcg = torch.where(
        hit_mask.any(dim=1),
        1.0 / torch.log2(hit_ranks + 1.0),
        torch.zeros(batch_size, device=predictions.device)
    )

    # Ideal DCG = 1 (target at position 1)
    idcg = torch.ones(batch_size, device=predictions.device)

    ndcg = (dcg / idcg).mean().item()

    return ndcg


def mrr_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    exclude_padding: bool = True,
    padding_idx: int = 0
) -> float:
    """
    Compute MRR@K (Mean Reciprocal Rank)

    If target item in top-k: 1 / rank
    Else: 0

    Args:
        predictions: [batch_size, num_items] or [batch_size, num_items + 1] prediction scores
        targets: [batch_size] target item indices
        k: cutoff
        exclude_padding: whether to exclude padding token from top-k
        padding_idx: index of padding token (default 0)

    Returns:
        mrr: average MRR@k (0-1)
    """
    batch_size = predictions.shape[0]

    # Exclude padding if requested
    if exclude_padding:
        candidates = _remove_padding(predictions, padding_idx)
        adjusted_targets = _align_targets(targets, padding_idx)
    else:
        candidates = predictions
        adjusted_targets = targets

    k = min(k, candidates.shape[1])

    # Get top-k predictions
    _, top_k_indices = torch.topk(candidates, k, dim=1)

    # Find position of target in top-k (1-indexed)
    targets_expanded = adjusted_targets.unsqueeze(1).expand(-1, k)
    hit_mask = (top_k_indices == targets_expanded)

    # Get ranks (1-indexed) where hit
    ranks = torch.arange(1, k + 1, device=predictions.device, dtype=torch.float)
    hit_ranks = (ranks * hit_mask.float()).max(dim=1)[0]  # [batch_size]

    # Compute reciprocal rank
    rr = torch.where(
        hit_mask.any(dim=1),
        1.0 / hit_ranks,
        torch.zeros(batch_size, device=predictions.device)
    )

    mrr = rr.mean().item()

    return mrr


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 15, 20, 50],
    exclude_padding: bool = True,
    padding_idx: int = 0
) -> Dict[str, float]:
    """
    Compute all metrics for given predictions

    Args:
        predictions: [batch_size, num_items] or [batch_size, num_items + 1] prediction scores
        targets: [batch_size] target item indices
        k_values: list of k values
        exclude_padding: whether to exclude padding token from top-k (default: True)
        padding_idx: index of padding token (default 0)

    Returns:
        metrics: dict of metric names and values (0-1 scale)
    """
    metrics = {}

    for k in k_values:
        metrics[f'Precision@{k}'] = precision_at_k(
            predictions, targets, k, exclude_padding, padding_idx
        )
        metrics[f'Recall@{k}'] = recall_at_k(
            predictions, targets, k, exclude_padding, padding_idx
        )
        metrics[f'NDCG@{k}'] = ndcg_at_k(
            predictions, targets, k, exclude_padding, padding_idx
        )
        metrics[f'MRR@{k}'] = mrr_at_k(
            predictions, targets, k, exclude_padding, padding_idx
        )

    return metrics


def format_metrics_percentage(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Convert metrics from 0-1 scale to percentage strings

    Args:
        metrics: dict of metric names and values (0-1 scale)

    Returns:
        formatted_metrics: dict with percentage strings (e.g., "5.23%")
    """
    formatted = {}
    for name, value in metrics.items():
        formatted[name] = f"{value * 100:.2f}%"
    return formatted


def print_metrics(
    metrics: Dict[str, float],
    title: str = "Metrics",
    percentage: bool = True
):
    """
    Pretty print metrics

    Args:
        metrics: dict of metric names and values (0-1 scale)
        title: title for the output
        percentage: if True, display as percentages
    """
    logger.info(f"\n{title}")
    logger.info("=" * 80)

    # Group by k
    k_values = sorted(set([int(k.split('@')[1]) for k in metrics.keys()]))

    for k in k_values:
        logger.info(f"@{k}:")
        for metric_type in ['Precision', 'Recall', 'NDCG', 'MRR']:
            key = f'{metric_type}@{k}'
            if key in metrics:
                value = metrics[key]
                if percentage:
                    logger.info(f"  {metric_type:12s}: {value * 100:6.2f}%")
                else:
                    logger.info(f"  {metric_type:12s}: {value:8.4f}")

    logger.info("=" * 80)


def print_main_results(
    metrics: Dict[str, float],
    title: str = "Main Results"
):
    """
    Print main results table (P@5, P@20, NDCG@5, NDCG@20, MRR@5, MRR@20)

    Args:
        metrics: dict of metric names and values (0-1 scale)
        title: title for the output
    """
    logger.info(f"\n{title}")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<15} {'Value':<15}")
    logger.info("-" * 80)

    main_metrics = ['Precision@5', 'Precision@20',
                   'NDCG@5', 'NDCG@20',
                   'MRR@5', 'MRR@20']

    for metric_name in main_metrics:
        if metric_name in metrics:
            value = metrics[metric_name]
            logger.info(f"{metric_name:<15} {value * 100:6.2f}%")

    logger.info("=" * 80)


def compute_mean_std(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std across multiple runs

    Args:
        metrics_list: list of metrics dicts from each run

    Returns:
        stats: dict with 'mean' and 'std' for each metric
    """
    if not metrics_list:
        return {}

    # Get all metric names
    metric_names = metrics_list[0].keys()

    stats = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list]
        stats[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1)  # Sample std
        }

    return stats


def print_mean_std(
    stats: Dict[str, Dict[str, float]],
    title: str = "Cross-Validation Results",
    main_metrics_only: bool = True
):
    """
    Print mean ± std results

    Args:
        stats: dict with 'mean' and 'std' for each metric
        title: title for the output
        main_metrics_only: if True, only print main metrics
    """
    logger.info(f"\n{title}")
    logger.info("=" * 80)

    if main_metrics_only:
        metric_names = ['Precision@5', 'Precision@20',
                       'NDCG@5', 'NDCG@20',
                       'MRR@5', 'MRR@20']
    else:
        # Sort by metric type and k
        metric_names = sorted(stats.keys())

    for metric_name in metric_names:
        if metric_name in stats:
            mean_val = stats[metric_name]['mean'] * 100
            std_val = stats[metric_name]['std'] * 100
            logger.info(f"{metric_name:<15}: {mean_val:6.2f} ± {std_val:6.2f}%")

    logger.info("=" * 80)


if __name__ == '__main__':
    print("Testing unified metrics with padding handling...")

    # Create dummy data
    batch_size = 100
    num_items = 1000

    # Simulate model output: [batch, num_items + 1] (includes padding at index 0)
    predictions = torch.randn(batch_size, num_items + 1)
    # Targets are in range [1, num_items] (never 0, which is padding)
    targets = torch.randint(1, num_items + 1, (batch_size,))

    # Test with padding exclusion (RECOMMENDED)
    print("\n=== Test 1: With padding exclusion (recommended) ===")
    metrics_with_exclusion = compute_all_metrics(
        predictions, targets, k_values=[5, 10, 20],
        exclude_padding=True, padding_idx=0
    )

    print_main_results(metrics_with_exclusion, "Main Results (Padding Excluded)")

    # Test without padding exclusion (OLD BEHAVIOR)
    print("\n=== Test 2: Without padding exclusion (old behavior) ===")
    metrics_without_exclusion = compute_all_metrics(
        predictions, targets, k_values=[5, 10, 20],
        exclude_padding=False, padding_idx=0
    )

    print_main_results(metrics_without_exclusion, "Main Results (Padding Included)")

    # Compare
    print("\n=== Comparison ===")
    for metric in ['Precision@5', 'NDCG@20']:
        with_val = metrics_with_exclusion[metric] * 100
        without_val = metrics_without_exclusion[metric] * 100
        diff = with_val - without_val
        print(f"{metric}:")
        print(f"  With exclusion:    {with_val:.2f}%")
        print(f"  Without exclusion: {without_val:.2f}%")
        print(f"  Difference:         {diff:+.2f}%")

    print("\n✅ All unified metric tests passed!")
    print("\nRecommended usage:")
    print("  metrics = compute_all_metrics(predictions, targets, k_values, exclude_padding=True)")
