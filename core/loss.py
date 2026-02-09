"""
Loss functions for EDA-Net.

Includes:
    - ImbalanceAwareFocalLoss: Focal loss for probability inputs
    - ImbalanceAwareFocalLoss_Logits: Focal loss for raw logit inputs
    - EntropyRegularization: Encourages varied per-sample temperatures
    - EDANetLoss: Combined loss (focal + entropy regularisation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImbalanceAwareFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with automatic class weight balancing.
    Loss = -α_t * (1 - p_t)^γ * log(p_t)
    Expects inputs to be probabilities (e.g. after Sigmoid).
    """
    def __init__(self, alpha=None, gamma=2.0, class_counts=None, reduction='mean'):
        super(ImbalanceAwareFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if class_counts is not None:
            total = sum(class_counts)
            self.alpha_pos = total / (2 * class_counts[1])
            self.alpha_neg = total / (2 * class_counts[0])
        elif alpha is not None:
            self.alpha_pos = alpha
            self.alpha_neg = 1 - alpha
        else:
            self.alpha_pos = 0.25
            self.alpha_neg = 0.75

    def forward(self, inputs, targets):
        p = inputs.clamp(min=1e-7, max=1-1e-7)
        ce_loss = -targets * torch.log(p) - (1 - targets) * torch.log(1 - p)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha_pos * targets + self.alpha_neg * (1 - targets)
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ImbalanceAwareFocalLoss_Logits(nn.Module):
    """
    Enhanced Focal Loss with automatic class weight balancing, designed for raw logits.
    Loss = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, class_counts=None, reduction='mean'):
        super(ImbalanceAwareFocalLoss_Logits, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if class_counts is not None:
            total = sum(class_counts)
            self.alpha_pos = total / (2 * class_counts[1])
            self.alpha_neg = total / (2 * class_counts[0])
        elif alpha is not None:
            self.alpha_pos = alpha
            self.alpha_neg = 1 - alpha
        else:
            self.alpha_pos = 0.25
            self.alpha_neg = 0.75

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha_pos * targets + self.alpha_neg * (1 - targets)
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class EntropyRegularization(nn.Module):
    """
    Regularisation term that encourages the EDT module to produce
    varied per-sample temperatures (not collapse to a constant τ).

    Minimises: -weight * Var(τ across batch)
    i.e., penalises low variance in temperature across the batch.
    """

    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight

    def forward(self, edt_info):
        """
        Args:
            edt_info: dict from model.last_edt_info containing
                      edt_info['edt_attention']['mean_tau'] of shape (B, 1)
        Returns:
            reg_loss: scalar regularisation loss
        """
        if edt_info is None:
            return torch.tensor(0.0)

        mean_tau = edt_info['edt_attention']['mean_tau']  # (B, 1)
        # Encourage variance across the batch
        tau_variance = mean_tau.var()
        # Negative variance → model is penalised for constant τ
        reg_loss = -self.weight * tau_variance

        return reg_loss


class EDANetLoss(nn.Module):
    """
    Combined loss for EDA-Net training:
        L_total = L_focal + λ_ent * L_entropy_reg
    """

    def __init__(self, gamma=2.0, class_counts=None, entropy_reg_weight=0.01):
        super().__init__()
        self.focal_loss = ImbalanceAwareFocalLoss_Logits(
            gamma=gamma, class_counts=class_counts
        )
        self.entropy_reg = EntropyRegularization(weight=entropy_reg_weight)

    def forward(self, logits, targets, edt_info=None):
        focal = self.focal_loss(logits, targets)
        ent_reg = self.entropy_reg(edt_info) if edt_info is not None else 0.0
        return focal + ent_reg, {'focal': focal.item(), 'entropy_reg': ent_reg.item() if torch.is_tensor(ent_reg) else 0.0}
