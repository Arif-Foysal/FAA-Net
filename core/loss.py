"""
Loss functions for EDA-Net.

Includes:
    - ImbalanceAwareFocalLoss: Focal loss for probability inputs
    - ImbalanceAwareFocalLoss_Logits: Focal loss for raw logit inputs
    - AsymmetricLoss: ASL for heavily imbalanced data (ICCV 2021)
    - EntropyRegularization: Encourages varied per-sample temperatures
    - SupervisedContrastiveLoss: SupCon loss (NeurIPS 2020)
    - EDANetLoss: Combined loss (focal + entropy reg + SupCon + diversity)
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


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) from Khosla et al., NeurIPS 2020.
    
    Pulls together samples of the same class in embedding space while
    pushing apart samples of different classes.
    
    L_supcon = Σ_i (1/|P(i)|) Σ_{p∈P(i)} -log(exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ))
    
    where P(i) = {p : y_p = y_i, p ≠ i} is the set of positives for anchor i.
    """
    
    def __init__(self, temperature=0.1, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, projection_dim) L2-normalized embeddings
            labels: (batch_size,) or (batch_size, 1) class labels
        Returns:
            loss: scalar SupCon loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)
        
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Labels batch size doesn't match features")
        
        # Mask: 1 if same class (excluding self), 0 otherwise
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Self-contrast mask (exclude diagonal)
        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask  # Remove self from positives
        
        # Check if there are any positives
        positives_per_sample = mask.sum(1)
        if (positives_per_sample == 0).all():
            # No positives in this batch — skip loss
            return torch.tensor(0.0, device=device)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.mm(features, features.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positives
        # Handle samples with no positives by setting their contribution to 0
        mean_log_prob_pos = (mask * log_prob).sum(1) / (positives_per_sample + 1e-8)
        
        # Only include samples that have positives
        has_positives = (positives_per_sample > 0).float()
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = (loss * has_positives).sum() / (has_positives.sum() + 1e-8)
        
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (ICCV 2021).
    
    Key idea: Different focusing parameters for positives vs negatives,
    plus a margin that shifts the decision boundary for negatives.
    
    L_ASL = -y * (1-p)^γ+ * log(p) - (1-y) * p_m^γ- * log(1-p_m)
    where p_m = max(p - m, 0) applies margin to negatives
    """
    
    def __init__(self, gamma_pos=1, gamma_neg=4, margin=0.05, eps=1e-8, 
                 class_counts=None, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.eps = eps
        self.reduction = reduction
        
        # Class weighting
        if class_counts is not None:
            total = sum(class_counts)
            self.alpha_pos = total / (2 * class_counts[1])
            self.alpha_neg = total / (2 * class_counts[0])
        else:
            self.alpha_pos = 1.0
            self.alpha_neg = 1.0
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: raw logits (batch_size, 1) or (batch_size,)
            targets: binary labels (batch_size, 1) or (batch_size,)
        """
        p = torch.sigmoid(inputs)
        
        # Positive loss
        p_pos = p.clamp(min=self.eps, max=1-self.eps)
        pos_loss = targets * (1 - p_pos) ** self.gamma_pos * torch.log(p_pos)
        
        # Negative loss with margin
        p_neg = (p - self.margin).clamp(min=self.eps)
        neg_loss = (1 - targets) * p_neg ** self.gamma_neg * torch.log(1 - p_neg.clamp(max=1-self.eps))
        
        # Weighted combination
        loss = -self.alpha_pos * pos_loss - self.alpha_neg * neg_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class EDANetLoss(nn.Module):
    """
    Combined loss for EDA-Net training (v2):
        L_total = L_focal + λ_ent * L_entropy_reg + λ_supcon * L_supcon 
                  + λ_anchor * L_anchor + λ_diversity * L_diversity
    
    Components:
        - Focal/ASL loss for classification
        - Entropy regularization (encourage varied temperatures)
        - Supervised contrastive loss (pull same-class embeddings together)
        - Prototype anchor loss (prevent prototype drift)
        - Head diversity loss (encourage head specialization)
    """

    def __init__(self, gamma=2.0, class_counts=None, entropy_reg_weight=0.01,
                 prototype_anchor_weight=0.01, supcon_weight=0.1, 
                 head_diversity_weight=0.01, use_asymmetric_loss=False):
        super().__init__()
        
        # Primary classification loss
        if use_asymmetric_loss:
            self.focal_loss = AsymmetricLoss(
                gamma_pos=1, gamma_neg=4, margin=0.05, class_counts=class_counts
            )
        else:
            self.focal_loss = ImbalanceAwareFocalLoss_Logits(
                gamma=gamma, class_counts=class_counts
            )
        
        self.entropy_reg = EntropyRegularization(weight=entropy_reg_weight)
        self.supcon_loss = SupervisedContrastiveLoss(temperature=0.1)
        
        # Loss weights
        self.prototype_anchor_weight = prototype_anchor_weight
        self.supcon_weight = supcon_weight
        self.head_diversity_weight = head_diversity_weight

    def forward(self, logits, targets, edt_info=None, prototype_anchor_loss=None,
                head_diversity_loss=None, labels_for_supcon=None):
        """
        Args:
            logits: model output (raw logits if output_logits=True)
            targets: ground truth labels (possibly smoothed)
            edt_info: dict from model.last_edt_info (optional)
            prototype_anchor_loss: pre-computed anchor loss (optional)
            head_diversity_loss: pre-computed diversity loss (optional)
            labels_for_supcon: original labels for SupCon (not smoothed)
        """
        # Classification loss
        focal = self.focal_loss(logits, targets)
        
        # Entropy regularization
        ent_reg = self.entropy_reg(edt_info) if edt_info is not None else 0.0
        
        # Prototype anchor loss
        anchor = self.prototype_anchor_weight * prototype_anchor_loss \
                 if prototype_anchor_loss is not None else 0.0
        
        # Head diversity loss
        diversity = self.head_diversity_weight * head_diversity_loss \
                    if head_diversity_loss is not None else 0.0
        
        # Supervised contrastive loss
        supcon = torch.tensor(0.0, device=logits.device)
        if edt_info is not None and 'projection' in edt_info and labels_for_supcon is not None:
            projection = edt_info['projection']
            supcon = self.supcon_weight * self.supcon_loss(projection, labels_for_supcon)
        
        # Total loss
        total = focal + ent_reg + anchor + diversity + supcon
        
        # Return breakdown for logging
        return total, {
            'focal': focal.item() if torch.is_tensor(focal) else focal,
            'entropy_reg': ent_reg.item() if torch.is_tensor(ent_reg) else 0.0,
            'prototype_anchor': anchor.item() if torch.is_tensor(anchor) else 0.0,
            'head_diversity': diversity.item() if torch.is_tensor(diversity) else 0.0,
            'supcon': supcon.item() if torch.is_tensor(supcon) else 0.0,
        }
