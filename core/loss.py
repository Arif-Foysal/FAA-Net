
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
            # Ensure class_counts are in (negative class count, positive class count) order
            total = sum(class_counts)
            self.alpha_pos = total / (2 * class_counts[1]) # Weight for positive class
            self.alpha_neg = total / (2 * class_counts[0]) # Weight for negative class
        elif alpha is not None:
            self.alpha_pos = alpha
            self.alpha_neg = 1 - alpha
        else:
            self.alpha_pos = 0.25 # Default for positive class
            self.alpha_neg = 0.75 # Default for negative class

    def forward(self, inputs, targets):
        # inputs are raw logits
        # targets are 0 or 1

        # Compute binary cross-entropy with logits
        # This handles numerical stability better than applying sigmoid then log
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Apply sigmoid to get probabilities for focal term
        p = torch.sigmoid(inputs)

        # p_t: probability of the ground truth class
        # if target is 1, p_t = p
        # if target is 0, p_t = 1 - p
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha term: alpha_pos for positive class, alpha_neg for negative class
        alpha_t = self.alpha_pos * targets + self.alpha_neg * (1 - targets)

        # Combine terms
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
