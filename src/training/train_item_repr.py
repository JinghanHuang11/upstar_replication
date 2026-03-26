"""
Training script for Item Representation Learning

Uses link prediction as supervision signal to train Item-GNN.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import pickle
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.item_gnn import ItemGNN
from src.graphs.item_graph import ItemGraphBuilder
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/item_repr.yaml')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def load_metadata(config: dict) -> dict:
    """Load dataset metadata"""
    processed_dir = Path(config['dataset']['processed_dir'])
    with open(processed_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    logger.info(f"Loaded metadata: {metadata['num_users']} users, {metadata['num_items']} items")

    return metadata


def build_or_load_item_graph(config: dict) -> dict:
    """Build or load item graph"""
    builder = ItemGraphBuilder(config)

    # Try to load from cache
    graph = builder.load_graph('item_graph.pt')

    if graph is None:
        logger.info("Building item graph from scratch...")

        # Load sequences
        train_sequences = builder.load_sequences('train')

        # Build sessions
        user_sessions = builder.build_session_sequences(train_sequences)

        # Build graphs
        in_session_graph = builder.build_in_session_graph(user_sessions)
        cross_session_graph = builder.build_cross_session_graph(user_sessions)

        # Combine
        graph = builder.build_combined_graph(in_session_graph, cross_session_graph)

        # Save
        builder.save_graph(graph, 'item_graph.pt')

    return graph


def sample_negative_edges(
    edge_index: torch.Tensor,
    num_items: int,
    num_negatives: int
) -> torch.Tensor:
    """
    Sample negative edges for link prediction

    Args:
        edge_index: [2, num_edges] - positive edges
        num_items: total number of items
        num_negatives: number of negative samples

    Returns:
        neg_edge_index: [2, num_negatives] - negative edges
    """
    # Create set of existing edges
    positive_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Sample negative edges
    neg_edges = []
    attempts = 0
    max_attempts = num_negatives * 10

    while len(neg_edges) < num_negatives and attempts < max_attempts:
        src = torch.randint(0, num_items, (1,)).item()
        dst = torch.randint(0, num_items, (1,)).item()

        if (src, dst) not in positive_edges and src != dst:
            neg_edges.append((src, dst))

        attempts += 1

    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).T

    return neg_edge_index


def train_epoch(
    model: nn.Module,
    graph: dict,
    optimizer: optim.Optimizer,
    num_items: int,
    device: torch.device,
    config: dict
) -> float:
    """Train for one epoch using link prediction"""
    model.train()

    edge_index = graph['edge_index'].to(device)
    edge_weight = graph.get('edge_weight', None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Sample positive and negative edges
    num_edges = edge_index.shape[1]
    num_samples = min(config['training']['batch_size_edges'], num_edges)

    # Random sample positive edges
    perm = torch.randperm(num_edges)[:num_samples]
    pos_edge_index = edge_index[:, perm]

    # Sample negative edges
    neg_edge_index = sample_negative_edges(
        edge_index,
        num_items,
        num_samples
    ).to(device)

    # Forward pass
    optimizer.zero_grad()

    # Get embeddings
    embeddings = model(edge_index, edge_weight, return_embeddings=True)

    # Compute link scores
    pos_src_emb = embeddings[pos_edge_index[0]]
    pos_dst_emb = embeddings[pos_edge_index[1]]
    pos_scores = (pos_src_emb * pos_dst_emb).sum(dim=1)

    neg_src_emb = embeddings[neg_edge_index[0]]
    neg_dst_emb = embeddings[neg_edge_index[1]]
    neg_scores = (neg_src_emb * neg_dst_emb).sum(dim=1)

    # BPR loss
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    graph: dict,
    num_items: int,
    device: torch.device,
    config: dict
) -> dict:
    """Evaluate using link prediction"""
    model.eval()

    edge_index = graph['edge_index'].to(device)
    edge_weight = graph.get('edge_weight', None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Get embeddings
    embeddings = model(edge_index, edge_weight, return_embeddings=True)

    # Sample test edges
    num_test = config['training'].get('num_eval_edges', 1000)

    # Positive edges
    perm = torch.randperm(edge_index.shape[1])[:num_test]
    pos_edge_index = edge_index[:, perm]

    pos_src_emb = embeddings[pos_edge_index[0]]
    pos_dst_emb = embeddings[pos_edge_index[1]]
    pos_scores = (pos_src_emb * pos_dst_emb).sum(dim=1)

    # Negative edges
    neg_edge_index = sample_negative_edges(
        edge_index,
        num_items,
        num_test
    ).to(device)

    neg_src_emb = embeddings[neg_edge_index[0]]
    neg_dst_emb = embeddings[neg_edge_index[1]]
    neg_scores = (neg_src_emb * neg_dst_emb).sum(dim=1)

    # Compute metrics
    # Positive scores should be higher than negative
    auc = (pos_scores > neg_scores).float().mean().item()

    return {
        'auc': auc,
        'pos_score_mean': pos_scores.mean().item(),
        'neg_score_mean': neg_scores.mean().item()
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_path: Path
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def save_item_embeddings(
    model: nn.Module,
    save_path: Path
):
    """Save item embeddings"""
    embeddings = model.get_item_embeddings()
    torch.save(embeddings, save_path)
    logger.info(f"Saved item embeddings to {save_path}")
    logger.info(f"  Embedding shape: {embeddings.shape}")


def analyze_embeddings(
    embeddings: torch.Tensor,
    num_examples: int = 5
):
    """Analyze learned item embeddings"""
    logger.info("\n" + "=" * 60)
    logger.info("Embedding Analysis")
    logger.info("=" * 60)

    # Statistics
    logger.info(f"Shape: {embeddings.shape}")
    logger.info(f"Mean: {embeddings.mean().item():.4f}")
    logger.info(f"Std: {embeddings.std().item():.4f}")
    logger.info(f"Min: {embeddings.min().item():.4f}")
    logger.info(f"Max: {embeddings.max().item():.4f}")

    # Nearest neighbors
    logger.info(f"\nTop {num_examples} items and their nearest neighbors:")

    # Compute similarity matrix
    norm_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarity = torch.mm(norm_embeddings, norm_embeddings.t())

    for i in range(min(num_examples, embeddings.shape[0])):
        # Get top neighbors (excluding self)
        sim_values, neighbors = similarity[i].topk(k=num_examples + 1)
        neighbors = neighbors[1:]  # Skip self

        logger.info(f"  Item {i}: neighbors = {neighbors.tolist()}")


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config['seed'])

    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'train_item_repr.log'

    global logger
    logger = get_logger(__name__, str(log_file))

    logger.info("=" * 60)
    logger.info("Item Representation Learning")
    logger.info("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load metadata
    metadata = load_metadata(config)
    num_items = metadata['num_items']

    # Build or load item graph
    graph = build_or_load_item_graph(config)

    # Log graph statistics
    from src.graphs.item_graph import ItemGraphBuilder
    builder = ItemGraphBuilder(config)
    stats = builder.compute_graph_statistics(graph, num_items)

    logger.info("\nItem Graph Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Create model
    model = ItemGNN(
        num_items=num_items,
        embed_dim=config['model']['embed_dim'],
        num_layers=config['model']['num_layers'],
        use_remember_gate=config['model']['use_remember_gate'],
        dropout=config['model']['dropout']
    ).to(device)

    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Training loop
    max_epochs = config['training']['max_epochs']
    best_auc = 0.0

    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{max_epochs}")

        # Train
        train_loss = train_epoch(
            model, graph, optimizer, num_items, device, config
        )
        logger.info(f"Train loss: {train_loss:.4f}")

        # Evaluate
        if (epoch + 1) % config['training']['eval_interval'] == 0:
            metrics = evaluate(model, graph, num_items, device, config)
            logger.info(f"Evaluation AUC: {metrics['auc']:.4f}")

            # Save if best
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                save_path = checkpoint_dir / 'best_model.pt'
                save_checkpoint(model, optimizer, epoch + 1, metrics, save_path)

    # Save final embeddings
    logger.info("\n" + "=" * 60)
    logger.info("Saving final embeddings...")
    logger.info("=" * 60)

    # Load best model
    best_checkpoint = torch.load(checkpoint_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Save embeddings
    output_dir = Path(config['logging']['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / 'item_embeddings.pt'
    save_item_embeddings(model, embedding_path)

    # Analyze embeddings
    embeddings = torch.load(embedding_path)
    analyze_embeddings(embeddings, num_examples=5)

    logger.info("=" * 60)
    logger.info("Item representation learning complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()