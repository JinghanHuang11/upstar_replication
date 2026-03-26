"""
Fusion Module for UPSTAR

Learnable fusion gate to combine three path outputs.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FusionGate(nn.Module):
    """
    Learnable fusion gate for combining three paths

    Computes attention weights for each path:
    gate = softmax(W * [z_stab, z_expl, z_other] + b)

    Then:
    z_global = gate_stab * z_stab + gate_expl * z_expl + gate_other * z_other
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        use_gate: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_gate = use_gate

        if use_gate:
            # Gate network
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            )

            logger.info(f"Initialized FusionGate with learnable gate (hidden_dim={hidden_dim})")
        else:
            logger.info("Initialized FusionGate with equal weights (no learning)")

    def forward(
        self,
        z_stab: torch.Tensor,
        z_expl: torch.Tensor,
        z_other: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse three path representations

        Args:
            z_stab: [batch_size, hidden_dim] - stable representation
            z_expl: [batch_size, hidden_dim] - exploratory representation
            z_other: [batch_size, hidden_dim] - entire representation

        Returns:
            z_global: [batch_size, hidden_dim] - fused representation
            gate_info: {
                'gate_weights': [batch_size, 3] - fusion weights
                'gate_stab': [batch_size, 1]
                'gate_expl': [batch_size, 1]
                'gate_other': [batch_size, 1]
            }
        """
        batch_size = z_stab.shape[0]

        if self.use_gate:
            # Concatenate three representations
            concat = torch.cat([z_stab, z_expl, z_other], dim=1)  # [B, 3*hidden]

            # Compute gate weights
            gate_logits = self.gate_network(concat)  # [B, 3]
            gate_weights = torch.softmax(gate_logits, dim=1)  # [B, 3]

            # Split weights
            gate_stab = gate_weights[:, 0:1]  # [B, 1]
            gate_expl = gate_weights[:, 1:2]  # [B, 1]
            gate_other = gate_weights[:, 2:3]  # [B, 1]

        else:
            # Equal weights
            gate_stab = torch.ones(batch_size, 1, device=z_stab.device) * (1.0 / 3)
            gate_expl = torch.ones(batch_size, 1, device=z_stab.device) * (1.0 / 3)
            gate_other = torch.ones(batch_size, 1, device=z_stab.device) * (1.0 / 3)
            gate_weights = torch.cat([gate_stab, gate_expl, gate_other], dim=1)

        # Fuse representations
        z_global = (
            gate_stab * z_stab +
            gate_expl * z_expl +
            gate_other * z_other
        )  # [B, hidden_dim]

        gate_info = {
            'gate_weights': gate_weights,
            'gate_stab': gate_stab,
            'gate_expl': gate_expl,
            'gate_other': gate_other
        }

        return z_global, gate_info


class FusionModule(nn.Module):
    """
    Complete fusion module for UPSTAR

    Fuses both representations and logits
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        vocab_size: int = None,
        use_gate: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.use_gate = use_gate

        # Representation fusion gate
        self.repr_gate = FusionGate(hidden_dim, use_gate)

        # Logits fusion gate (optional, can use same gate or separate)
        if vocab_size is not None:
            # Note: logits have shape [B, vocab_size+1] (includes padding token)
            # So we need to initialize with vocab_size + 1
            self.logit_gate = FusionGate(vocab_size + 1, use_gate)

    def fuse_representations(
        self,
        z_stab: torch.Tensor,
        z_expl: torch.Tensor,
        z_other: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse hidden representations

        Args:
            z_stab, z_expl, z_other: [batch_size, hidden_dim]

        Returns:
            z_global: [batch_size, hidden_dim]
            gate_info: dict with gate weights
        """
        return self.repr_gate(z_stab, z_expl, z_other)

    def fuse_logits(
        self,
        y_hat_stab: torch.Tensor,
        y_hat_expl: torch.Tensor,
        y_hat_other: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse prediction logits

        Args:
            y_hat_stab, y_hat_expl, y_hat_other: [batch_size, vocab_size+1]

        Returns:
            y_hat_global: [batch_size, vocab_size+1]
            gate_info: dict with gate weights
        """
        if self.vocab_size is None:
            raise ValueError("vocab_size must be set for logit fusion")

        return self.logit_gate(y_hat_stab, y_hat_expl, y_hat_other)

    def fuse_all(
        self,
        z_stab: torch.Tensor,
        z_expl: torch.Tensor,
        z_other: torch.Tensor,
        y_hat_stab: torch.Tensor,
        y_hat_expl: torch.Tensor,
        y_hat_other: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse both representations and logits

        Returns:
            {
                'z_global': [batch_size, hidden_dim],
                'y_hat_global': [batch_size, vocab_size+1],
                'gate_repr': dict,
                'gate_logit': dict
            }
        """
        z_global, gate_repr = self.fuse_representations(z_stab, z_expl, z_other)
        y_hat_global, gate_logit = self.fuse_logits(y_hat_stab, y_hat_expl, y_hat_other)

        return {
            'z_global': z_global,
            'y_hat_global': y_hat_global,
            'gate_repr': gate_repr,
            'gate_logit': gate_logit
        }


if __name__ == '__main__':
    # Test fusion module
    print("Testing Fusion Module...")

    batch_size = 4
    hidden_dim = 128
    vocab_size = 1000

    # Create dummy inputs
    z_stab = torch.randn(batch_size, hidden_dim)
    z_expl = torch.randn(batch_size, hidden_dim)
    z_other = torch.randn(batch_size, hidden_dim)

    y_hat_stab = torch.randn(batch_size, vocab_size + 1)
    y_hat_expl = torch.randn(batch_size, vocab_size + 1)
    y_hat_other = torch.randn(batch_size, vocab_size + 1)

    # Test representation fusion
    print("\n1. Testing Representation Fusion...")
    fusion = FusionModule(hidden_dim=hidden_dim, use_gate=True)
    z_global, gate_info = fusion.fuse_representations(z_stab, z_expl, z_other)

    print(f"  z_global shape: {z_global.shape}")
    print(f"  gate_weights shape: {gate_info['gate_weights'].shape}")
    print(f"  Avg gate weights: stab={gate_info['gate_stab'].mean():.4f}, "
          f"expl={gate_info['gate_expl'].mean():.4f}, "
          f"other={gate_info['gate_other'].mean():.4f}")

    # Test logit fusion
    print("\n2. Testing Logit Fusion...")
    fusion_with_vocab = FusionModule(hidden_dim=hidden_dim, vocab_size=vocab_size, use_gate=True)
    y_hat_global, gate_logit = fusion_with_vocab.fuse_logits(y_hat_stab, y_hat_expl, y_hat_other)

    print(f"  y_hat_global shape: {y_hat_global.shape}")
    print(f"  Avg gate weights: stab={gate_logit['gate_stab'].mean():.4f}, "
          f"expl={gate_logit['gate_expl'].mean():.4f}, "
          f"other={gate_logit['gate_other'].mean():.4f}")

    # Test complete fusion
    print("\n3. Testing Complete Fusion...")
    result = fusion_with_vocab.fuse_all(
        z_stab, z_expl, z_other,
        y_hat_stab, y_hat_expl, y_hat_other
    )

    print(f"  z_global shape: {result['z_global'].shape}")
    print(f"  y_hat_global shape: {result['y_hat_global'].shape}")

    print("\nFusion module test passed!")
