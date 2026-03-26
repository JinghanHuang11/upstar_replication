"""
UPSTAR Model: Three-path recommendation with global fusion

Paper: UPSTAR - Uncovering Purchase Motivations for Sequential Recommendation
Sections: 3.1.4 (Next Item Prediction) + 3.3 (Dual Teacher–Student Training)

Architecture:
1. Split user sequence into three paths based on motivation:
   - S-model: Encodes stable preference subsequence
   - E-model: Encodes exploratory intent subsequence
   - O-model: Encodes entire original sequence

2. Each path is a 4-layer LSTM (hidden_size=128, per paper Section 7.3)

3. Global Fusion Gate: Combines three paths with learnable weights
   - Learns adaptive weights for each path based on input
   - z_global = gate_stab * z_stab + gate_expl * z_expl + gate_other * z_other

4. Joint Training with Multiple Losses (per Section 3.3):
   - L_global: Cross-entropy on fused prediction
   - L_branch: Cross-entropy on each path (S, E, O)
   - L_orth: Orthogonality constraint (z_other ⟂ z_stab, z_other ⟂ z_expl)
   - L_distill: Dual teacher-student distillation
     - If target is stable: S teaches E
     - If target is exploratory: E teaches S
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import logging

from src.models.sequence_models import SModel, EModel, OModel
from src.models.fusion import FusionModule

logger = logging.getLogger(__name__)


class UPSTARModel(nn.Module):
    """
    UPSTAR: Uncovering Purchase Motivations for Sequential Recommendation

    Paper: Section 3.1.4 + 3.3

    Three-path model:
    - S-model: Encodes stable preference subsequence
    - E-model: Encodes exploratory intent subsequence
    - O-model: Encodes entire original sequence
    - Global Fusion: Combines three paths with learnable gate

    Implementation follows paper specifications:
    - LSTM: 4 layers, hidden_size=128 (Section 7.3)
    - Embedding: 64-dim (configurable)
    - Dropout: 0.2 (default)
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_gate: bool = True
    ):
        super().__init__()

        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gate = use_gate

        # Three path models
        self.s_model = SModel(
            vocab_size=num_items,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.e_model = EModel(
            vocab_size=num_items,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.o_model = OModel(
            vocab_size=num_items,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Fusion module
        self.fusion = FusionModule(
            hidden_dim=hidden_dim,
            vocab_size=num_items,
            use_gate=use_gate
        )

        # Total parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized UPSTAR model: {total_params:,} parameters")

    def forward(
        self,
        seq_stable: torch.Tensor,
        len_stable: torch.Tensor,
        seq_exploratory: torch.Tensor,
        len_exploratory: torch.Tensor,
        seq_entire: torch.Tensor,
        len_entire: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            seq_stable: [batch_size, max_seq_len] - stable subsequence
            len_stable: [batch_size] - stable lengths
            seq_exploratory: [batch_size, max_seq_len] - exploratory subsequence
            len_exploratory: [batch_size] - exploratory lengths
            seq_entire: [batch_size, max_seq_len] - entire sequence
            len_entire: [batch_size] - entire lengths

        Returns:
            {
                # Path outputs
                'z_stab': [batch_size, hidden_dim],
                'y_hat_stab': [batch_size, num_items+1],
                'p_stab': [batch_size, num_items+1],

                'z_expl': [batch_size, hidden_dim],
                'y_hat_expl': [batch_size, num_items+1],
                'p_expl': [batch_size, num_items+1],

                'z_other': [batch_size, hidden_dim],
                'y_hat_other': [batch_size, num_items+1],
                'p_other': [batch_size, num_items+1],

                # Global outputs
                'z_global': [batch_size, hidden_dim],
                'y_hat_global': [batch_size, num_items+1],
                'p_global': [batch_size, num_items+1],

                # Gate information
                'gate_weights': [batch_size, 3],
                'gate_repr': dict,
                'gate_logit': dict
            }
        """
        # S-model
        z_stab, y_hat_stab, p_stab = self.s_model(seq_stable, len_stable)

        # E-model
        z_expl, y_hat_expl, p_expl = self.e_model(seq_exploratory, len_exploratory)

        # O-model
        z_other, y_hat_other, p_other = self.o_model(seq_entire, len_entire)

        # Fusion
        result = self.fusion.fuse_all(
            z_stab, z_expl, z_other,
            y_hat_stab, y_hat_expl, y_hat_other
        )

        # Compute global probabilities
        p_global = torch.softmax(result['y_hat_global'], dim=1)

        output = {
            # S-model outputs
            'z_stab': z_stab,
            'y_hat_stab': y_hat_stab,
            'p_stab': p_stab,

            # E-model outputs
            'z_expl': z_expl,
            'y_hat_expl': y_hat_expl,
            'p_expl': p_expl,

            # O-model outputs
            'z_other': z_other,
            'y_hat_other': y_hat_other,
            'p_other': p_other,

            # Global outputs
            'z_global': result['z_global'],
            'y_hat_global': result['y_hat_global'],
            'p_global': p_global,

            # Gate information
            'gate_weights': result['gate_repr']['gate_weights'],
            'gate_repr': result['gate_repr'],
            'gate_logit': result['gate_logit']
        }

        return output

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        k: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Get top-k predictions from each path and global

        Args:
            batch: input batch
            k: number of predictions

        Returns:
            predictions: {
                'top_k_stab': [batch_size, k],
                'top_k_expl': [batch_size, k],
                'top_k_other': [batch_size, k],
                'top_k_global': [batch_size, k]
            }
        """
        with torch.no_grad():
            output = self.forward(
                batch['seq_stable'], batch['len_stable'],
                batch['seq_exploratory'], batch['len_exploratory'],
                batch['seq_entire'], batch['len_entire']
            )

        # Get top-k for each path
        _, top_k_stab = torch.topk(output['y_hat_stab'], k, dim=1)
        _, top_k_expl = torch.topk(output['y_hat_expl'], k, dim=1)
        _, top_k_other = torch.topk(output['y_hat_other'], k, dim=1)
        _, top_k_global = torch.topk(output['y_hat_global'], k, dim=1)

        return {
            'top_k_stab': top_k_stab,
            'top_k_expl': top_k_expl,
            'top_k_other': top_k_other,
            'top_k_global': top_k_global
        }


if __name__ == '__main__':
    # Test UPSTAR model
    print("Testing UPSTAR Model...")

    num_items = 1000
    batch_size = 4
    max_seq_len = 20

    # Create dummy input
    seq_stable = torch.randint(1, num_items + 1, (batch_size, max_seq_len))
    len_stable = torch.randint(5, max_seq_len, (batch_size,))

    seq_exploratory = torch.randint(1, num_items + 1, (batch_size, max_seq_len))
    len_exploratory = torch.randint(5, max_seq_len, (batch_size,))

    seq_entire = torch.randint(1, num_items + 1, (batch_size, max_seq_len))
    len_entire = torch.randint(5, max_seq_len, (batch_size,))

    # Create model
    model = UPSTARModel(
        num_items=num_items,
        embed_dim=64,
        hidden_dim=128,
        num_layers=4,
        use_gate=True
    )

    # Forward pass
    output = model(
        seq_stable, len_stable,
        seq_exploratory, len_exploratory,
        seq_entire, len_entire
    )

    print(f"\nOutput keys: {list(output.keys())}")

    print("\nRepresentations:")
    print(f"  z_stab:   {output['z_stab'].shape}")
    print(f"  z_expl:   {output['z_expl'].shape}")
    print(f"  z_other:  {output['z_other'].shape}")
    print(f"  z_global: {output['z_global'].shape}")

    print("\nLogits:")
    print(f"  y_hat_stab:   {output['y_hat_stab'].shape}")
    print(f"  y_hat_expl:   {output['y_hat_expl'].shape}")
    print(f"  y_hat_other:  {output['y_hat_other'].shape}")
    print(f"  y_hat_global: {output['y_hat_global'].shape}")

    print("\nProbabilities:")
    print(f"  p_stab:   {output['p_stab'].shape}")
    print(f"  p_expl:   {output['p_expl'].shape}")
    print(f"  p_other:  {output['p_other'].shape}")
    print(f"  p_global: {output['p_global'].shape}")

    print("\nGate weights (per-dim fusion):")
    print(f"  gate_weights (f_s): {output['gate_weights'].shape}")  # [B, hidden, 3]
    gate_repr = output['gate_repr']
    print(f"  Mean per-dim: stab={gate_repr['gate_stab'].mean():.4f}, "
          f"expl={gate_repr['gate_expl'].mean():.4f}, "
          f"other={gate_repr['gate_other'].mean():.4f}")

    print("\nUPSTAR model test passed!")
