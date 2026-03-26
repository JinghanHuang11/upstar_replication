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
        # Loss weights (per paper Section 3.3 formula):
        #   L_total = λ_global*L_global + λ_branch*L_S&E&O + λ_ortho*L_orth + λ_distill*L_distill
        #
        # Paper-derived defaults:
        #   λ_global  = 1.0  (no additional weight)
        #   λ_branch  = 0.7  (paper λ = 0.7)
        #   λ_ortho   = 1.0  (τ_s, τ_e already inside L_orth; no extra weight)
        #   λ_distill = 1.0  (distill_lambda already inside L_distill; no extra weight)
        #
        # Engineering knobs for experimentation:
        #   lambda_ortho   > 1.0: stronger orthogonality push
        #   lambda_distill > 1.0: stronger distillation signal
        lambda_global: float = 1.0,
        lambda_branch: float = 0.7,
        lambda_ortho: float = 1.0,
        lambda_distill: float = 1.0,

        # Loss switches (for staged training)
        use_global_loss: bool = True,
        use_branch_loss: bool = False,
        use_orthogonality_loss: bool = False,
        use_distillation_loss: bool = False,

        # Orthogonality parameters (inside L_orth)
        tau_s: float = 0.5,   # paper: τ_s = 0.5
        tau_e: float = 0.5,   # paper: τ_e = 0.5

        # Distillation parameters (inside L_distill)
        temperature: float = 3.0,
        distill_lambda: float = 0.7,  # paper: λ = 0.7
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

        # Orthogonality parameters (inside L_orth)
        self.tau_s = tau_s
        self.tau_e = tau_e

        # Distillation parameters (inside L_distill)
        self.temperature = temperature
        self.distill_lambda = distill_lambda
        self.num_items = num_items

        # Base loss
        self.ce_loss = nn.CrossEntropyLoss()

        logger.info("=" * 60)
        logger.info("Initialized UPSTAR Loss")
        logger.info("=" * 60)
        logger.info(f"Loss formula (paper Section 3.3):")
        logger.info(f"  L_total = λ_global*L_global + λ_branch*L_S&E&O + λ_ortho*L_orth + λ_distill*L_distill")
        logger.info(f"Loss weights:")
        logger.info(f"  λ_global  = {lambda_global}  (paper: 1.0)")
        logger.info(f"  λ_branch  = {lambda_branch}  (paper: λ=0.7 for L_S&E&O)")
        logger.info(f"  λ_ortho   = {lambda_ortho}  (τ_s, τ_e already inside L_orth)")
        logger.info(f"  λ_distill = {lambda_distill}  (distill_lambda already inside L_distill)")
        logger.info(f"Inner params:")
        logger.info(f"  tau_s={tau_s}, tau_e={tau_e}  (inside L_orth)")
        logger.info(f"  temperature={temperature}, distill_lambda={distill_lambda}  (inside L_distill)")
        logger.info(f"Loss switches:")
        logger.info(f"  L_global:    {use_global_loss}")
        logger.info(f"  L_branch:    {use_branch_loss}")
        logger.info(f"  L_orth:     {use_orthogonality_loss}")
        logger.info(f"  L_distill:   {use_distillation_loss}")
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
        Stage 3: Orthogonality loss (Paper Section 3.3)

        Per paper:
            L_orth = τ_s * z_other^T z_stab + τ_e * z_other^T z_expl

        Engineering: compute dot product per sample, then mean over batch.
        - z_other^T z_stab = sum_d z_other[b,d] * z_stab[b,d]   → scalar per sample
        - Loss = mean over batch of (τ_s * dot_s + τ_e * dot_e)
        - No normalization: paper uses raw dot product
        - No squared term: paper uses linear dot product

        Args:
            z_stab:   [batch_size, hidden_dim]
            z_expl:   [batch_size, hidden_dim]
            z_other:  [batch_size, hidden_dim]

        Returns:
            loss: scalar
        """
        if not self.use_orthogonality_loss:
            return torch.tensor(0.0, device=z_stab.device)

        # Per-sample dot products (no normalization -- paper literal)
        # z_other[b] · z_stab[b] = sum_d z_other[b,d] * z_stab[b,d]
        dot_s = (z_other * z_stab).sum(dim=1)    # [batch_size]
        dot_e = (z_other * z_expl).sum(dim=1)    # [batch_size]

        # Mean over batch, weighted by τ_s and τ_e
        loss = self.tau_s * dot_s.mean() + self.tau_e * dot_e.mean()

        return loss

    def compute_distillation_loss(
        self,
        y_hat_stab: torch.Tensor,
        y_hat_expl: torch.Tensor,
        targets: torch.Tensor,
        motivation_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 4: Dual teacher-student distillation loss (Paper Section 3.3)

        Conditional distillation based on target item motivation:
        - If target is stable    (label==1): S → E  (S teaches E)
        - If target is exploratory (label==0): E → S  (E teaches S)
        - If target is uncategorized (label==2): NO distillation

        KL divergence with temperature:
            L = λ * KL( softmax(teacher/T),  log_softmax(student/T) )

        Args:
            y_hat_stab:       [batch_size, num_items+1] - S-model logits
            y_hat_expl:       [batch_size, num_items+1] - E-model logits
            targets:           [batch_size]              - unused here
            motivation_labels: [batch_size]              - motivation of each target item
                                                         - 1=stable, 0=exploratory, 2=uncategorized

        Returns:
            {
                'loss_distill':  scalar,
                'loss_s_to_e':  scalar,   # L_S→E  (S teaches E on stable targets)
                'loss_e_to_s':  scalar,   # L_E→S  (E teaches S on exploratory targets)
                's_teaches_e':  float,    # batch count where S→E fires
                'e_teaches_s':  float     # batch count where E→S fires
            }
        """
        if not self.use_distillation_loss:
            return {
                'loss_distill': torch.tensor(0.0, device=y_hat_stab.device),
                'loss_s_to_e': torch.tensor(0.0, device=y_hat_stab.device),
                'loss_e_to_s': torch.tensor(0.0, device=y_hat_stab.device),
                's_teaches_e': 0.0,
                'e_teaches_s': 0.0
            }

        batch_size = targets.shape[0]
        device = y_hat_stab.device

        # ------------------------------------------------------------------
        # 1. Construct conditional masks from motivation_labels
        # ------------------------------------------------------------------
        # stable_mask:    samples whose target item has stable motivation (label==1)
        # expl_mask:      samples whose target item has exploratory motivation (label==0)
        # uncategorized:  samples where label==2 → NO distillation
        stable_mask = (motivation_labels == 1)   # [batch_size]
        expl_mask   = (motivation_labels == 0)    # [batch_size]

        num_stable = stable_mask.sum().item()
        num_expl   = expl_mask.sum().item()

        loss_s_to_e = torch.tensor(0.0, device=device)
        loss_e_to_s = torch.tensor(0.0, device=device)

        # ------------------------------------------------------------------
        # 2. L_S→E: S teaches E on stable targets
        #    teacher = S-model, student = E-model
        # ------------------------------------------------------------------
        if num_stable > 0:
            teacher_s_to_e = y_hat_stab[stable_mask].detach()      # teacher: detach, no grad
            student_s_to_e = y_hat_expl[stable_mask]                # student: keep grad

            # Soft targets + temperature
            p_teacher = torch.softmax(teacher_s_to_e / self.temperature, dim=1)
            log_p_student = torch.log_softmax(student_s_to_e / self.temperature, dim=1)

            # KL(teacher || student) = sum p * (log p - log q)
            # reduction='batchmean': divides by batch_size within each subset
            kl_s_to_e = torch.nn.functional.kl_div(
                log_p_student,
                p_teacher,
                reduction='batchmean'
            )

            # Apply distillation weight λ
            loss_s_to_e = self.distill_lambda * kl_s_to_e

        # ------------------------------------------------------------------
        # 3. L_E→S: E teaches S on exploratory targets
        #    teacher = E-model, student = S-model
        # ------------------------------------------------------------------
        if num_expl > 0:
            teacher_e_to_s = y_hat_expl[expl_mask].detach()        # teacher: detach, no grad
            student_e_to_s = y_hat_stab[expl_mask]                  # student: keep grad

            p_teacher = torch.softmax(teacher_e_to_s / self.temperature, dim=1)
            log_p_student = torch.log_softmax(student_e_to_s / self.temperature, dim=1)

            kl_e_to_s = torch.nn.functional.kl_div(
                log_p_student,
                p_teacher,
                reduction='batchmean'
            )

            loss_e_to_s = self.distill_lambda * kl_e_to_s

        # ------------------------------------------------------------------
        # 4. Combine: both terms summed, weighted by λ in each branch
        # ------------------------------------------------------------------
        loss_distill = loss_s_to_e + loss_e_to_s

        return {
            'loss_distill': loss_distill,
            'loss_s_to_e':  loss_s_to_e,    # S→E term
            'loss_e_to_s':  loss_e_to_s,    # E→S term
            's_teaches_e':  float(num_stable),
            'e_teaches_s':  float(num_expl)
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
                'loss_s_to_e': torch.tensor(0.0, device=targets.device),
                'loss_e_to_s': torch.tensor(0.0, device=targets.device),
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
    Create UPSTARLoss from configuration.

    All values should come from config['training'] which is aligned with
    the paper (see tafeng_upstar.yaml for documentation).

    Returns:
        UPSTARLoss instance
    """
    training = config['training']
    num_items = config['model']['num_items']

    return UPSTARLoss(
        # ── Loss weights ─────────────────────────────────────────────
        # L_total = λ_global*L_global + λ_branch*L_S&E&O + λ_ortho*L_orth + λ_distill*L_distill
        lambda_global=training.get('lambda_global', 1.0),   # paper: 1.0
        lambda_branch=training.get('lambda_branch', 0.7),   # paper: λ=0.7
        lambda_ortho=training.get('lambda_ortho', 1.0),    # τ_s,τ_e already inside L_orth
        lambda_distill=training.get('lambda_distill', 1.0),  # distill_lambda already inside L_distill

        # ── Loss switches (for staged curriculum) ──────────────────────
        use_global_loss=training.get('use_global_loss', True),
        use_branch_loss=training.get('use_branch_loss', False),
        use_orthogonality_loss=training.get('use_orthogonality_loss', False),
        use_distillation_loss=training.get('use_distillation_loss', False),

        # ── L_orth parameters (inside L_orth) ─────────────────────────
        tau_s=training.get('tau_s', 0.5),   # paper: τ_s = 0.5
        tau_e=training.get('tau_e', 0.5),   # paper: τ_e = 0.5

        # ── L_distill parameters (inside L_distill) ──────────────────
        temperature=training.get('distill_temperature', 3.0),
        distill_lambda=training.get('distill_lambda', 0.7),  # paper: λ = 0.7

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