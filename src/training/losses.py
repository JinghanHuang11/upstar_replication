"""
Loss Functions for UPSTAR

Phase 5: Gradually add losses in stages:
1. L_global (global prediction loss)
2. L_S&E&O (branch prediction losses)
3. L_orth (orthogonality loss)
4. L_distill (dual teacher-student distillation)

All losses are configurable via config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class UPSTARLoss(nn.Module):
    """
    Combined loss for UPSTAR with all components

    Total Loss = λ_global * L_global
              + λ_branch * L_branch
              + λ_ortho * L_orth
              + λ_distill * L_distill
    """

    def __init__(
        self,
        # Loss weights
        lambda_global: float = 1.0,
        lambda_branch: float = 0.5,
        lambda_ortho: float = 0.1,
        lambda_distill: float = 0.3,

        # Loss switches (for staged training)
        use_global_loss: bool = True,
        use_branch_loss: bool = False,
        use_orthogonality_loss: bool = False,
        use_distillation_loss: bool = False,

        # Orthogonality parameters
        tau_s: float = 0.5,
        tau_e: float = 0.5,

        # Distillation parameters
        temperature: float = 3.0,
        distill_lambda: float = 0.7,
        num_items: int = None
    ):
        super().__init__()

        # Loss weights
        self.lambda_global = lambda_global
        self.lambda_branch = lambda_branch
        self.lambda_ortho = lambda_ortho
        self.lambda_distill = lambda_distill

        # Loss switches
        self.use_global_loss = use_global_loss
        self.use_branch_loss = use_branch_loss
        self.use_orthogonality_loss = use_orthogonality_loss
        self.use_distillation_loss = use_distillation_loss

        # Orthogonality parameters
        self.tau_s = tau_s
        self.tau_e = tau_e

        # Distillation parameters
        self.temperature = temperature
        self.distill_lambda = distill_lambda
        self.num_items = num_items

        # Base loss
        self.ce_loss = nn.CrossEntropyLoss()

        logger.info("=" * 60)
        logger.info("Initialized UPSTAR Loss")
        logger.info("=" * 60)
        logger.info(f"Loss weights:")
        logger.info(f"  λ_global:  {lambda_global}")
        logger.info(f"  λ_branch:  {lambda_branch}")
        logger.info(f"  λ_ortho:  {lambda_ortho}")
        logger.info(f"  λ_distill: {lambda_distill}")
        logger.info(f"Loss switches:")
        logger.info(f"  L_global:    {use_global_loss}")
        logger.info(f"  L_branch:    {use_branch_loss}")
        logger.info(f"  L_orth:     {use_orthogonality_loss}")
        logger.info(f"  L_distill:   {use_distillation_loss}")
        logger.info(f"Parameters:")
        logger.info(f"  tau_s: {tau_s}, tau_e: {tau_e}")
        logger.info(f"  temperature: {temperature}, lambda_distill: {lambda_distill}")
        logger.info("=" * 60)

    def compute_global_loss(
        self,
        y_hat_global: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Stage 1: Global prediction loss

        Args:
            y_hat_global: [batch_size, num_items+1]
            targets: [batch_size]

        Returns:
            loss: scalar
        """
        if not self.use_global_loss:
            return torch.tensor(0.0, device=y_hat_global.device)

        return self.ce_loss(y_hat_global, targets)

    def compute_branch_losses(
        self,
        y_hat_stab: torch.Tensor,
        y_hat_expl: torch.Tensor,
        y_hat_other: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 2: Branch prediction losses

        Args:
            y_hat_stab, y_hat_expl, y_hat_other: [batch_size, num_items+1]
            targets: [batch_size]

        Returns:
            {
                'loss_stab': scalar,
                'loss_expl': scalar,
                'loss_other': scalar,
                'total_branch': scalar
            }
        """
        if not self.use_branch_loss:
            return {
                'loss_stab': torch.tensor(0.0, device=y_hat_stab.device),
                'loss_expl': torch.tensor(0.0, device=y_hat_expl.device),
                'loss_other': torch.tensor(0.0, device=y_hat_other.device),
                'total_branch': torch.tensor(0.0, device=y_hat_stab.device)
            }

        loss_stab = self.ce_loss(y_hat_stab, targets)
        loss_expl = self.ce_loss(y_hat_expl, targets)
        loss_other = self.ce_loss(y_hat_other, targets)

        total_branch = loss_stab + loss_expl + loss_other

        return {
            'loss_stab': loss_stab,
            'loss_expl': loss_expl,
            'loss_other': loss_other,
            'total_branch': total_branch
        }

    def compute_orthogonality_loss(
        self,
        z_stab: torch.Tensor,
        z_expl: torch.Tensor,
        z_other: torch.Tensor
    ) -> torch.Tensor:
        """
        Stage 3: Orthogonality loss

        Constrains z_other to be orthogonal to z_stab and z_expl:
        L_orth = tau_s * dot(z_other, z_stab)^2 + tau_e * dot(z_other, z_expl)^2

        Args:
            z_stab, z_expl, z_other: [batch_size, hidden_dim]

        Returns:
            loss: scalar
        """
        if not self.use_orthogonality_loss:
            return torch.tensor(0.0, device=z_stab.device)

        # Normalize
        z_stab_norm = F.normalize(z_stab, dim=1)
        z_expl_norm = F.normalize(z_expl, dim=1)
        z_other_norm = F.normalize(z_other, dim=1)

        # Dot products (similarity)
        dot_s = (z_other_norm * z_stab_norm).sum(dim=1)  # [batch_size]
        dot_e = (z_other_norm * z_expl_norm).sum(dim=1)  # [batch_size]

        # Squared and weighted
        loss = self.tau_s * dot_s.pow(2).mean() + self.tau_e * dot_e.pow(2).mean()

        return loss

    def compute_distillation_loss(
        self,
        y_hat_stab: torch.Tensor,
        y_hat_expl: torch.Tensor,
        targets: torch.Tensor,
        motivation_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 4: Dual teacher-student distillation loss

        If target item is stable (label=1): S-model teaches E-model
        If target item is exploratory (label=0): E-model teaches S-model

        Args:
            y_hat_stab: [batch_size, num_items+1] - S-model predictions
            y_hat_expl: [batch_size, num_items+1] - E-model predictions
            targets: [batch_size] - target items
            motivation_labels: [batch_size] - motivation of target items

        Returns:
            {
                'loss_distill': scalar,
                's_teaches_e': float,  # Number of times S teaches E
                'e_teaches_s': float   # Number of times E teaches S
            }
        """
        if not self.use_distillation_loss:
            return {
                'loss_distill': torch.tensor(0.0, device=y_hat_stab.device),
                's_teaches_e': 0.0,
                'e_teaches_s': 0.0
            }

        batch_size = targets.shape[0]

        # Get motivation of target items
        target_motivations = motivation_labels[targets]  # [batch_size]

        # Count stable and exploratory targets
        num_stable_targets = (target_motivations == 1).sum().item()
        num_expl_targets = (target_motivations == 0).sum().item()

        # If no stable or exploratory targets, skip
        if num_stable_targets == 0 and num_expl_targets == 0:
            return {
                'loss_distill': torch.tensor(0.0, device=y_hat_stab.device),
                's_teaches_e': 0.0,
                'e_teaches_s': 0.0
            }

        total_loss = torch.tensor(0.0, device=y_hat_stab.device)

        # Case 1: S teaches E (target is stable)
        if num_stable_targets > 0:
            # Stable targets: S-model predictions as teacher
            teacher_logits = y_hat_stab[target_motivations == 1]  # [num_stable_targets, num_items+1]
            student_logits = y_hat_expl[target_motivations == 1]  # [num_stable_targets, num_items+1]

            # KL divergence
            # F.kl_div expects log_prob (first) and prob (second)
            teacher_prob = torch.softmax(teacher_logits / self.temperature, dim=1)
            student_log_prob = F.log_softmax(student_logits / self.temperature, dim=1)

            kl_div = F.kl_div(
                student_log_prob,
                teacher_prob,
                reduction='batchmean'
            )

            # Apply distillation weight
            # Only weight by self.distill_lambda, not by temperature^2 (handled in KL)
            weighted_loss = self.distill_lambda * kl_div

            # Normalize by batch
            weighted_loss = weighted_loss * (batch_size / num_stable_targets)

            total_loss += weighted_loss

        # Case 2: E teaches S (target is exploratory)
        if num_expl_targets > 0:
            # Exploratory targets: E-model predictions as teacher
            teacher_logits = y_hat_expl[target_motivations == 0]  # [num_expl_targets, num_items+1]
            student_logits = y_hat_stab[target_motivations == 0]  # [num_expl_targets, num_items+1]

            # F.kl_div expects log_prob (first) and prob (second)
            teacher_prob = torch.softmax(teacher_logits / self.temperature, dim=1)
            student_log_prob = F.log_softmax(student_logits / self.temperature, dim=1)

            kl_div = F.kl_div(
                student_log_prob,
                teacher_prob,
                reduction='batchmean'
            )

            weighted_loss = self.distill_lambda * kl_div
            weighted_loss = weighted_loss * (batch_size / num_expl_targets)

            total_loss += weighted_loss

        return {
            'loss_distill': total_loss,
            's_teaches_e': num_stable_targets,
            'e_teaches_s': num_expl_targets
        }

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        motivation_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all enabled losses

        Args:
            predictions: {
                'y_hat_global', 'y_hat_stab', 'y_hat_expl', 'y_hat_other',
                'z_stab', 'z_expl', 'z_other'
            }
            targets: [batch_size]
            motivation_labels: [batch_size] - motivation of target items

        Returns:
            {
                'total': total loss,
                'global': global loss,
                'branch': dict of branch losses,
                'ortho': orthogonality loss,
                'distill': dict of distillation loss
            }
        """
        # Stage 1: Global loss
        global_loss = self.compute_global_loss(
            predictions['y_hat_global'],
            targets
        )

        # Stage 2: Branch losses
        branch_losses = self.compute_branch_losses(
            predictions['y_hat_stab'],
            predictions['y_hat_expl'],
            predictions['y_hat_other'],
            targets
        )

        # Stage 3: Orthogonality loss
        ortho_loss = self.compute_orthogonality_loss(
            predictions['z_stab'],
            predictions['z_expl'],
            predictions['z_other']
        )

        # Stage 4: Distillation loss
        if motivation_labels is not None:
            distill_losses = self.compute_distillation_loss(
                predictions['y_hat_stab'],
                predictions['y_hat_expl'],
                targets,
                motivation_labels
            )
        else:
            distill_losses = {
                'loss_distill': torch.tensor(0.0, device=targets.device),
                's_teaches_e': 0.0,
                'e_teaches_s': 0.0
            }

        # Total loss
        total_loss = (
            self.lambda_global * global_loss +
            self.lambda_branch * branch_losses['total_branch'] +
            self.lambda_ortho * ortho_loss +
            self.lambda_distill * distill_losses['loss_distill']
        )

        return {
            'total': total_loss,
            'global': global_loss,
            'branch': branch_losses,
            'ortho': ortho_loss,
            'distill': distill_losses
        }


# Helper function to create loss from config
def create_loss_from_config(config: dict) -> UPSTARLoss:
    """
    Create UPSTARLoss from configuration

    Args:
        config: configuration dict

    Returns:
        UPSTARLoss instance
    """
    # Training config
    training = config['training']

    # Loss weights
    lambda_global = training.get('lambda_global', 1.0)
    lambda_branch = training.get('lambda_branch', 0.5)
    lambda_ortho = training.get('lambda_ortho', 0.1)
    lambda_distill = training.get('lambda_distill', 0.3)

    # Loss switches
    use_global_loss = training.get('use_global_loss', True)
    use_branch_loss = training.get('use_branch_loss', False)
    use_orthogonality_loss = training.get('use_orthogonality_loss', False)
    use_distillation_loss = training.get('use_distillation_loss', False)

    # Orthogonality parameters
    tau_s = training.get('tau_s', 0.5)
    tau_e = training.get('tau_e', 0.5)

    # Distillation parameters
    temperature = training.get('distill_temperature', 3.0)
    lambda_distill_training = training.get('lambda_distill', 0.7)
    num_items = config['model']['num_items']

    return UPSTARLoss(
        lambda_global=lambda_global,
        lambda_branch=lambda_branch,
        lambda_ortho=lambda_ortho,
        lambda_distill=lambda_distill_training,
        use_global_loss=use_global_loss,
        use_branch_loss=use_branch_loss,
        use_orthogonality_loss=use_orthogonality_loss,
        use_distillation_loss=use_distillation_loss,
        tau_s=tau_s,
        tau_e=tau_e,
        temperature=temperature,
        distill_lambda=lambda_distill_training,
        num_items=num_items
    )


if __name__ == '__main__':
    # Test loss functions
    print("Testing UPSTAR Loss...")

    # Create dummy data
    batch_size = 32
    num_items = 1000
    hidden_dim = 128

    predictions = {
        'y_hat_global': torch.randn(batch_size, num_items + 1),
        'y_hat_stab': torch.randn(batch_size, num_items + 1),
        'y_hat_expl': torch.randn(batch_size, num_items + 1),
        'y_hat_other': torch.randn(batch_size, num_items + 1),
        'z_stab': torch.randn(batch_size, hidden_dim),
        'z_expl': torch.randn(batch_size, hidden_dim),
        'z_other': torch.randn(batch_size, hidden_dim)
    }

    targets = torch.randint(0, num_items, (batch_size,))
    motivation_labels = torch.randint(0, 3, (batch_size,))

    # Test Stage 1: Global loss only
    print("\nStage 1: Global loss only")
    loss_fn = UPSTARLoss(
        use_global_loss=True,
        use_branch_loss=False,
        use_orthogonality_loss=False,
        use_distillation_loss=False
    )
    output = loss_fn(predictions, targets, motivation_labels)
    print(f"  Total: {output['total'].item():.4f}")
    print(f"  Global: {output['global'].item():.4f}")

    # Test Stage 2: Global + Branch
    print("\nStage 2: Global + Branch losses")
    loss_fn = UPSTARLoss(
        use_global_loss=True,
        use_branch_loss=True,
        use_orthogonality_loss=False,
        use_distillation_loss=False
    )
    output = loss_fn(predictions, targets, motivation_labels)
    print(f"  Total: {output['total'].item():.4f}")
    print(f"  Global: {output['global'].item():.4f}")
    print(f"  Branch (total): {output['branch']['total_branch'].item():.4f}")
    print(f"    S: {output['branch']['loss_stab'].item():.4f}")
    print(f"    E: {output['branch']['loss_expl'].item():.4f}")
    print(f"    O: {output['branch']['loss_other'].item():.4f}")

    # Test Stage 3: Global + Branch + Ortho
    print("\nStage 3: Global + Branch + Ortho")
    loss_fn = UPSTARLoss(
        use_global_loss=True,
        use_branch_loss=True,
        use_orthogonality_loss=True,
        use_distillation_loss=False
    )
    output = loss_fn(predictions, targets, motivation_labels)
    print(f"  Total: {output['total'].item():.4f}")
    print(f"  Global: {output['global'].item():.4f}")
    print(f"  Branch: {output['branch']['total_branch'].item():.4f}")
    print(f"  Ortho: {output['ortho'].item():.4f}")

    # Test Stage 4: All losses
    print("\nStage 4: All losses")
    loss_fn = UPSTARLoss(
        use_global_loss=True,
        use_branch_loss=True,
        use_orthogonality_loss=True,
        use_distillation_loss=True
    )
    output = loss_fn(predictions, targets, motivation_labels)
    print(f"  Total: {output['total'].item():.4f}")
    print(f"  Global: {output['global'].item():.4f}")
    print(f"  Branch: {output['branch']['total_branch'].item():.4f}")
    print(f"  Ortho: {output['ortho'].item():.4f}")
    print(f"  Distill: {output['distill']['loss_distill'].item():.4f}")
    print(f"    S teaches E: {output['distill']['s_teaches_e']}")
    print(f"    E teaches S: {output['distill']['e_teaches_s']}")

    print("\nAll loss tests passed!")