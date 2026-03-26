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
    Per-dimension fusion gate for combining three paths (Paper Section 3.1.4)

    Computes per-dimension attention weights:
    f_s = softmax(Linear([z_stab, z_expl, z_other]))   -- [B, hidden_dim * 3]
    f_s = reshape to [B, hidden_dim, 3]

    Then for each dimension d:
    z_global[b, d] = f_s[b, d, 0] * z_stab[b, d]
                  + f_s[b, d, 1] * z_expl[b, d]
                  + f_s[b, d, 2] * z_other[b, d]
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
            # Gate network: outputs per-dimension weights (3 per dim)
            self.gate_network = nn.Linear(hidden_dim * 3, hidden_dim * 3, bias=True)

            logger.info(f"Initialized FusionGate with per-dimension gate (hidden_dim={hidden_dim})")
        else:
            logger.info("Initialized FusionGate with equal per-dimension weights (no learning)")

    def forward(
        self,
        z_stab: torch.Tensor,
        z_expl: torch.Tensor,
        z_other: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse three path representations (per-dimension)

        Args:
            z_stab: [batch_size, hidden_dim] - stable representation
            z_expl: [batch_size, hidden_dim] - exploratory representation
            z_other: [batch_size, hidden_dim] - entire representation

        Returns:
            z_global: [batch_size, hidden_dim] - fused representation
            gate_info: {
                'gate_weights': [batch_size, 3, hidden_dim] - per-dim fusion weights
                'gate_stab': [batch_size, hidden_dim]
                'gate_expl': [batch_size, hidden_dim]
                'gate_other': [batch_size, hidden_dim]
            }
        """
        batch_size = z_stab.shape[0]
        H = self.hidden_dim

        if self.use_gate:
            # Concatenate three representations
            concat = torch.cat([z_stab, z_expl, z_other], dim=1)  # [B, 3*hidden]

            # Compute per-dimension gate logits
            gate_logits = self.gate_network(concat)  # [B, 3*hidden]

            # Softmax over the 3 weights for each dimension
            # Reshape: [B, 3*hidden] -> [B, hidden, 3]
            gate_logits_per_dim = gate_logits.view(batch_size, H, 3)
            f_s = torch.softmax(gate_logits_per_dim, dim=2)  # [B, hidden, 3]

            # Split into per-path weights: each [B, hidden]
            gate_stab = f_s[:, :, 0]    # weight for stable path per dim
            gate_expl = f_s[:, :, 1]    # weight for exploratory path per dim
            gate_other = f_s[:, :, 2]  # weight for other path per dim

        else:
            # Equal per-dimension weights (1/3 each)
            gate_stab = torch.ones(batch_size, H, device=z_stab.device) * (1.0 / 3)
            gate_expl = torch.ones(batch_size, H, device=z_stab.device) * (1.0 / 3)
            gate_other = torch.ones(batch_size, H, device=z_stab.device) * (1.0 / 3)
            f_s = torch.stack([gate_stab, gate_expl, gate_other], dim=2)

        # Fuse representations per dimension
        z_global = (
            gate_stab * z_stab +
            gate_expl * z_expl +
            gate_other * z_other
        )  # [B, hidden_dim]

        gate_info = {
            'gate_weights': f_s,          # [B, hidden, 3] -- for compat / logging
            'gate_stab': gate_stab,       # [B, hidden]
            'gate_expl': gate_expl,
            'gate_other': gate_other
        }

        return z_global, gate_info


class FusionModule(nn.Module):
    """
    Complete fusion module for UPSTAR

    Paper Section 3.1.4:
    - Fuses per-dim representations: z_global = f_s[1]*z_stab + f_s[2]*z_expl + f_s[3]*z_other
    - Computes y_hat_global = softmax(Linear(z_global)) -- NOT weighted logits
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

        # Per-dimension representation fusion gate
        self.repr_gate = FusionGate(hidden_dim, use_gate)

        # y_hat_global = Linear(z_global) + softmax (per paper)
        self.global_proj = nn.Linear(hidden_dim, vocab_size + 1)

    def fuse_representations(
        self,
        z_stab: torch.Tensor,
        z_expl: torch.Tensor,
        z_other: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse hidden representations (per-dimension)

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
        y_hat_other: torch.Tensor,
        gate_info: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute global logits: y_hat_global = softmax(Linear(z_global))

        Note: per paper, y_hat_global is NOT a weighted average of branch logits.
        It is computed from the fused representation z_global.

        Args:
            y_hat_stab, y_hat_expl, y_hat_other: [batch_size, vocab_size+1] (unused, kept for compat)
            gate_info: gate weights from repr_gate

        Returns:
            y_hat_global: [batch_size, vocab_size+1]
            gate_info: same gate weights dict
        """
        raise NotImplementedError(
            "Use fuse_all() instead: y_hat_global = softmax(Linear(z_global))"
        )

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
        Fuse both representations and compute global logits.

        Per paper Section 3.1.4:
        - z_global = f_s[1] ⊙ z_stab + f_s[2] ⊙ z_expl + f_s[3] ⊙ z_other  (per-dim)
        - y_hat_global = softmax(Linear(z_global))

        Note: y_hat_stab/expl/other from individual LSTMs are still used for
        branch losses L_S&E&O; only the global prediction uses z_global.

        Returns:
            {
                'z_global': [batch_size, hidden_dim],
                'y_hat_global': [batch_size, vocab_size+1],
                'gate_repr': dict,
                'gate_logit': dict
            }
        """
        # Per-dim representation fusion
        z_global, gate_repr = self.repr_gate(z_stab, z_expl, z_other)

        # y_hat_global = softmax(Linear(z_global)) -- per paper
        y_hat_global = self.global_proj(z_global)

        return {
            'z_global': z_global,
            'y_hat_global': y_hat_global,
            'gate_repr': gate_repr,
            'gate_logit': gate_repr  # same per-dim weights
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
    fusion = FusionModule(hidden_dim=hidden_dim, vocab_size=vocab_size, use_gate=True)
    z_global, gate_info = fusion.fuse_representations(z_stab, z_expl, z_other)

    print(f"  z_global shape: {z_global.shape}")  # [B, 128]
    print(f"  gate_weights shape: {gate_info['gate_weights'].shape}")  # [B, 128, 3]
    print(f"  gate_stab shape: {gate_info['gate_stab'].shape}")  # [B, 128]
    print(f"  Avg gate weights (per dim mean): "
          f"stab={gate_info['gate_stab'].mean():.4f}, "
          f"expl={gate_info['gate_expl'].mean():.4f}, "
          f"other={gate_info['gate_other'].mean():.4f}")

    # Test complete fusion
    print("\n2. Testing Complete Fusion (y_hat_global = softmax(Linear(z_global)))...")
    result = fusion.fuse_all(
        z_stab, z_expl, z_other,
        y_hat_stab, y_hat_expl, y_hat_other
    )

    print(f"  z_global shape: {result['z_global'].shape}")
    print(f"  y_hat_global shape: {result['y_hat_global'].shape}")  # [B, vocab_size+1]
    print(f"  gate_weights shape: {result['gate_repr']['gate_weights'].shape}")  # [B, 128, 3]

    # Verify per-dim gate sums to 1
    f_s = result['gate_repr']['gate_weights']  # [B, 128, 3]
    dim_sum = f_s.sum(dim=2)  # [B, 128]
    print(f"  Per-dim gate sum (should be 1.0): min={dim_sum.min():.6f}, max={dim_sum.max():.6f}")

    print("\nFusion module test passed!")
