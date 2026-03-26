"""
Baseline LSTM model for next-item recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LSTMRec(nn.Module):
    """
    LSTM-based sequential recommendation model

    Architecture:
    - Item embedding layer
    - LSTM encoder
    - Output projection to item space
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        padding_idx: int = 0
    ):
        """
        Plain LSTM baseline for next-item recommendation

        **Paper-Aligned Default Hyperparameters** (Section 7.3):
        - hidden_dim: 128 (paper: hidden_size = 128)
        - num_layers: 4 (paper: num_layers = 4)

        **Baseline Simplicity** (intentionally kept simple):
        - Single-path LSTM (no S/E/O branches)
        - No fusion gate
        - No STB or motivation labels
        - Direct item embedding → LSTM → output projection

        Args:
            num_items: number of items
            embed_dim: item embedding dimension
            hidden_dim: LSTM hidden dimension (default: 128, paper-aligned)
            num_layers: number of LSTM layers (default: 4, paper-aligned)
            dropout: dropout rate
            padding_idx: padding index (usually 0)
        """
        super().__init__()

        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Item embedding (item 0 is padding)
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,  # +1 for padding
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )

        # LSTM encoder (single-path, no branches)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection (direct to item space)
        self.output_proj = nn.Linear(hidden_dim, num_items + 1)

        logger.info(f"Initialized LSTMRec model (Plain Baseline)")
        logger.info(f"  Items: {num_items}, Embed dim: {embed_dim}")
        logger.info(f"  Hidden dim: {hidden_dim} (paper-aligned: 128)")
        logger.info(f"  Layers: {num_layers} (paper-aligned: 4)")
        logger.info(f"  Architecture: Single-path LSTM (no S/E/O branches)")
        logger.info(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        items: torch.Tensor,
        seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            items: [batch_size, max_seq_len] item indices
            seq_lengths: [batch_size] actual sequence lengths

        Returns:
            logits: [batch_size, num_items] prediction scores
        """
        batch_size, max_seq_len = items.shape

        # Embed items: [batch_size, max_seq_len, embed_dim]
        item_embeds = self.item_embedding(items)

        # Apply dropout
        item_embeds = self.dropout(item_embeds)

        # Pack padded sequence for efficient LSTM processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            item_embeds,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM forward
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Get last hidden state: [num_layers, batch_size, hidden_dim]
        # Use the last layer's hidden state
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]

        # Apply dropout
        last_hidden = self.dropout(last_hidden)

        # Project to item space: [batch_size, num_items]
        logits = self.output_proj(last_hidden)

        return logits

    def predict(
        self,
        items: torch.Tensor,
        seq_lengths: torch.Tensor,
        k: int = 10
    ) -> torch.Tensor:
        """
        Get top-k predictions

        Args:
            items: [batch_size, max_seq_len]
            seq_lengths: [batch_size]
            k: number of predictions

        Returns:
            top_k_items: [batch_size, k] top-k item indices
        """
        # Get logits
        logits = self.forward(items, seq_lengths)

        # Get top-k
        _, top_k_items = torch.topk(logits, k, dim=1)

        return top_k_items

    def get_item_embeddings(self) -> torch.Tensor:
        """Get item embeddings (excluding padding)"""
        return self.item_embedding.weight[1:]  # Skip padding


if __name__ == '__main__':
    print("Testing LSTMRec model...")

    # Create model
    model = LSTMRec(
        num_items=1000,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2
    )

    # Test forward pass
    batch_size = 32
    max_seq_len = 50
    items = torch.randint(1, 1001, (batch_size, max_seq_len))
    seq_lengths = torch.randint(10, 50, (batch_size,))

    logits = model(items, seq_lengths)
    print(f"Logits shape: {logits.shape}")  # Should be [32, 1001]

    # Test predict
    top_k = model.predict(items, seq_lengths, k=10)
    print(f"Top-k shape: {top_k.shape}")  # Should be [32, 10]

    print("\nModel test passed!")