"""
Sequence Models for UPSTAR: S-model, E-model, O-model

Paper: UPSTAR - Section 3.1.4 (Next Item Prediction) + Section 7.3 (Implementation Details)

All models follow the same architecture:
- 4-layer LSTM (num_layers=4, per Section 7.3)
- Hidden size: 128 (hidden_dim=128, per Section 7.3)
- Embedding dimension: 64 (configurable)
- Dropout: 0.2 (default)

Each model processes a different subsequence:
- S-model: Stable preference subsequence (items with STB in top stable quantile)
- E-model: Exploratory intent subsequence (items with STB in bottom exploratory quantile)
- O-model: Entire original sequence (all items, including uncategorized)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SequenceEncoder(nn.Module):
    """
    4-layer LSTM encoder for sequence modeling

    Args:
        vocab_size: number of items
        embed_dim: embedding dimension
        hidden_dim: LSTM hidden size (128)
        num_layers: number of LSTM layers (4)
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Item embedding
        self.item_embedding = nn.Embedding(
            num_embeddings=vocab_size + 1,  # +1 for padding
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size + 1)

        logger.info(f"Initialized SequenceEncoder: "
                   f"vocab_size={vocab_size}, embed_dim={embed_dim}, "
                   f"hidden_dim={hidden_dim}, num_layers={num_layers}")

    def forward(
        self,
        items: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            items: [batch_size, max_seq_len] - item indices
            lengths: [batch_size] - actual sequence lengths

        Returns:
            hidden: [batch_size, hidden_dim] - sequence representation
            logits: [batch_size, vocab_size+1] - prediction scores
            probs: [batch_size, vocab_size+1] - probability distribution
        """
        batch_size, max_seq_len = items.shape

        # Embed items
        item_embeds = self.item_embedding(items)  # [B, L, D]
        item_embeds = self.dropout(item_embeds)

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            item_embeds,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM forward
        _, (hidden, _) = self.lstm(packed)

        # Get last hidden state
        hidden = hidden[-1]  # [batch_size, hidden_dim]

        # Project to vocab space
        logits = self.output_proj(hidden)  # [batch_size, vocab_size+1]

        # Compute probabilities
        probs = torch.softmax(logits, dim=1)  # [batch_size, vocab_size+1]

        return hidden, logits, probs


class SModel(nn.Module):
    """
    Stable Preference Model (S-model)

    Encodes stable preference subsequence.
    4-layer LSTM, hidden_size=128.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        logger.info("Initialized S-model (Stable Preference)")

    def forward(
        self,
        items: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            items: [batch_size, max_seq_len]
            lengths: [batch_size]

        Returns:
            z_stab: [batch_size, hidden_dim] - stable representation
            y_hat_stab: [batch_size, vocab_size+1] - logits
            p_stab: [batch_size, vocab_size+1] - probabilities
        """
        return self.encoder(items, lengths)


class EModel(nn.Module):
    """
    Exploratory Intent Model (E-model)

    Encodes exploratory intent subsequence.
    4-layer LSTM, hidden_size=128.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        logger.info("Initialized E-model (Exploratory Intent)")

    def forward(
        self,
        items: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            items: [batch_size, max_seq_len]
            lengths: [batch_size]

        Returns:
            z_expl: [batch_size, hidden_dim] - exploratory representation
            y_hat_expl: [batch_size, vocab_size+1] - logits
            p_expl: [batch_size, vocab_size+1] - probabilities
        """
        return self.encoder(items, lengths)


class OModel(nn.Module):
    """
    Other Model (O-model)

    Encodes entire original sequence.
    4-layer LSTM, hidden_size=128.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        logger.info("Initialized O-model (Other/Entire)")

    def forward(
        self,
        items: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            items: [batch_size, max_seq_len]
            lengths: [batch_size]

        Returns:
            z_other: [batch_size, hidden_dim] - entire representation
            y_hat_other: [batch_size, vocab_size+1] - logits
            p_other: [batch_size, vocab_size+1] - probabilities
        """
        return self.encoder(items, lengths)


if __name__ == '__main__':
    # Test models
    print("Testing Sequence Models...")

    vocab_size = 1000
    embed_dim = 64
    hidden_dim = 128
    num_layers = 4

    # Create dummy input
    batch_size = 4
    max_seq_len = 20
    items = torch.randint(1, vocab_size + 1, (batch_size, max_seq_len))
    lengths = torch.randint(5, max_seq_len, (batch_size,))

    # Test S-model
    print("\n1. Testing S-model...")
    s_model = SModel(vocab_size, embed_dim, hidden_dim, num_layers)
    z_stab, y_hat_stab, p_stab = s_model(items, lengths)
    print(f"  z_stab shape: {z_stab.shape}")      # [4, 128]
    print(f"  y_hat_stab shape: {y_hat_stab.shape}")  # [4, 1001]
    print(f"  p_stab shape: {p_stab.shape}")      # [4, 1001]

    # Test E-model
    print("\n2. Testing E-model...")
    e_model = EModel(vocab_size, embed_dim, hidden_dim, num_layers)
    z_expl, y_hat_expl, p_expl = e_model(items, lengths)
    print(f"  z_expl shape: {z_expl.shape}")
    print(f"  y_hat_expl shape: {y_hat_expl.shape}")
    print(f"  p_expl shape: {p_expl.shape}")

    # Test O-model
    print("\n3. Testing O-model...")
    o_model = OModel(vocab_size, embed_dim, hidden_dim, num_layers)
    z_other, y_hat_other, p_other = o_model(items, lengths)
    print(f"  z_other shape: {z_other.shape}")
    print(f"  y_hat_other shape: {y_hat_other.shape}")
    print(f"  p_other shape: {p_other.shape}")

    print("\nAll sequence models test passed!")
