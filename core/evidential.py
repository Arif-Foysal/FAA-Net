"""
Evidential Deep Learning module for FAA-Net.

Replaces point-estimate probabilities with Dirichlet-parameterized
evidence outputs, enabling separation of aleatoric and epistemic uncertainty.

Instead of u = 1 - 2|p - 0.5| (geometric trick that collapses aleatoric and
epistemic uncertainty into a single number), this module computes:
    - Evidence e_k >= 0 for each class k
    - Dirichlet concentration alpha_k = e_k + 1 (uniform prior)
    - Dirichlet strength S = sum(alpha_k)
    - Epistemic uncertainty u = K / S (vacuous belief mass)
    - Aleatoric uncertainty = Var[Dir(alpha)] per class

Reference: Sensoy et al., "Evidential Deep Learning to Quantify
Classification Uncertainty", NeurIPS 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialOutputLayer(nn.Module):
    """
    Replaces the final sigmoid/softmax with an evidence-producing layer.

    For K classes, outputs K non-negative evidence values via softplus,
    which parameterize a Dirichlet distribution over class probabilities.
    """

    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_layer = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Features from the penultimate layer [batch, in_features]

        Returns:
            dict with keys:
                - 'evidence':   Raw evidence e_k >= 0              [batch, K]
                - 'alpha':      Dirichlet concentration alpha_k    [batch, K]
                - 'dirichlet_strength': S = sum(alpha)             [batch, 1]
                - 'expected_prob':      p_hat_k = alpha_k / S      [batch, K]
                - 'epistemic_uncertainty':  u = K / S              [batch, 1]
                - 'aleatoric_uncertainty':  per-class Var[Dir]     [batch, K]
                - 'attack_prob':  p_hat for the attack class (k=1) [batch]
        """
        # Softplus ensures evidence is non-negative (smoother than ReLU)
        evidence = F.softplus(self.evidence_layer(x))

        alpha = evidence + 1.0  # Uniform Dirichlet prior
        S = alpha.sum(dim=-1, keepdim=True)  # Dirichlet strength

        expected_prob = alpha / S

        # Epistemic uncertainty: vacuous belief mass
        # When evidence is zero for all classes, S = K and u = 1.0 (total ignorance)
        # When evidence is large, u -> 0.0 (confident)
        epistemic = self.num_classes / S

        # Aleatoric uncertainty: expected variance of the Dirichlet
        # Var[p_k] = alpha_k * (S - alpha_k) / (S^2 * (S + 1))
        aleatoric = (alpha * (S - alpha)) / (S.pow(2) * (S + 1))

        return {
            'evidence': evidence,
            'alpha': alpha,
            'dirichlet_strength': S,
            'expected_prob': expected_prob,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'attack_prob': expected_prob[:, 1],  # For binary: class 1 = attack
        }


class EvidentialLoss(nn.Module):
    """
    Bayes risk under sum-of-squares loss + KL divergence regularizer.

    L = sum_k [(y_k - p_hat_k)^2 + p_hat_k(1 - p_hat_k) / (S + 1)]
        + lambda_t * KL[Dir(alpha_tilde) || Dir(1)]

    where alpha_tilde removes evidence for the correct class.
    lambda_t is annealed from 0 to 1 to avoid the KL regularizer
    fighting the classification loss during early training.
    """

    def __init__(
        self,
        num_classes: int = 2,
        annealing_epochs: int = 10,
        annealing_start: float = 0.0,
        annealing_end: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.annealing_start = annealing_start
        self.annealing_end = annealing_end

    def get_lambda(self, epoch: int) -> float:
        """Anneal KL regularization coefficient linearly."""
        if epoch >= self.annealing_epochs:
            return self.annealing_end
        t = epoch / max(self.annealing_epochs, 1)
        return self.annealing_start + (self.annealing_end - self.annealing_start) * t

    def kl_divergence_dirichlet(
        self, alpha: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        KL[Dir(alpha_tilde) || Dir(1)] where alpha_tilde removes
        evidence for the correct class.

        Args:
            alpha:  Dirichlet params [batch, K]
            target: One-hot labels   [batch, K]
        """
        # Remove evidence for the true class: keep prior (1) for correct class
        alpha_tilde = target + (1.0 - target) * alpha

        ones = torch.ones_like(alpha_tilde)
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
        S_ones = ones.sum(dim=-1, keepdim=True)

        kl = (
            torch.lgamma(S_tilde) - torch.lgamma(S_ones)
            - (torch.lgamma(alpha_tilde) - torch.lgamma(ones)).sum(dim=-1, keepdim=True)
            + ((alpha_tilde - ones) * (
                torch.digamma(alpha_tilde) - torch.digamma(S_tilde)
            )).sum(dim=-1, keepdim=True)
        )

        return kl.squeeze(-1)

    def forward(
        self,
        output: dict,
        target: torch.Tensor,
        epoch: int = 0,
    ) -> dict:
        """
        Args:
            output: Dict from EvidentialOutputLayer.forward()
            target: Labels [batch] or [batch, 1] — will be converted to one-hot
            epoch:  Current training epoch (for annealing lambda_t)

        Returns:
            dict with 'total_loss', 'mse_loss', 'variance_loss', 'kl_loss', 'lambda_t'
        """
        # Handle various target shapes: [batch], [batch, 1], float or long
        if target.dim() == 2:
            target = target.squeeze(-1)
        target = target.long()

        target_onehot = F.one_hot(target, self.num_classes).float()

        alpha = output['alpha']
        S = output['dirichlet_strength']
        p_hat = output['expected_prob']

        # MSE term: pushes expected probability toward correct class
        mse = (target_onehot - p_hat).pow(2).sum(dim=-1)

        # Variance term: penalizes uncertainty in the probability estimate
        variance = (p_hat * (1.0 - p_hat) / (S + 1.0)).sum(dim=-1)

        # KL regularizer: penalizes evidence for wrong classes
        lambda_t = self.get_lambda(epoch)
        kl = self.kl_divergence_dirichlet(alpha, target_onehot)

        total = (mse + variance + lambda_t * kl).mean()

        return {
            'total_loss': total,
            'mse_loss': mse.mean(),
            'variance_loss': variance.mean(),
            'kl_loss': kl.mean(),
            'lambda_t': lambda_t,
        }


class FocalEvidentialLoss(EvidentialLoss):
    """
    Evidential loss with focal weighting for class imbalance.

    Combines FAA-Net's existing focal loss intuition with
    evidential uncertainty. Hard-to-classify minority samples
    (low evidence) receive higher loss weight, preserving the
    imbalance-awareness of the original architecture.
    """

    def __init__(
        self,
        num_classes: int = 2,
        annealing_epochs: int = 10,
        gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            annealing_epochs=annealing_epochs,
            **kwargs,
        )
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', torch.ones(num_classes))

    def forward(
        self,
        output: dict,
        target: torch.Tensor,
        epoch: int = 0,
    ) -> dict:
        """Same interface as EvidentialLoss but with focal modulation."""
        if target.dim() == 2:
            target = target.squeeze(-1)
        target = target.long()

        target_onehot = F.one_hot(target, self.num_classes).float()

        alpha = output['alpha']
        S = output['dirichlet_strength']
        p_hat = output['expected_prob']

        # Focal weight: (1 - p_correct)^gamma
        # Hard samples (low probability of correct class) get higher weight
        p_correct = (p_hat * target_onehot).sum(dim=-1)  # [batch]
        focal_weight = (1.0 - p_correct).pow(self.gamma)

        # Per-sample class weight from imbalance ratio
        sample_weight = self.class_weights[target]

        # MSE + variance (same as base)
        mse = (target_onehot - p_hat).pow(2).sum(dim=-1)
        variance = (p_hat * (1.0 - p_hat) / (S + 1.0)).sum(dim=-1)

        # Apply focal and class weighting to classification terms
        weighted_loss = focal_weight * sample_weight * (mse + variance)

        # KL regularizer (unweighted — regularization should apply uniformly)
        lambda_t = self.get_lambda(epoch)
        kl = self.kl_divergence_dirichlet(alpha, target_onehot)

        total = (weighted_loss + lambda_t * kl).mean()

        return {
            'total_loss': total,
            'mse_loss': mse.mean(),
            'variance_loss': variance.mean(),
            'kl_loss': kl.mean(),
            'focal_weight_mean': focal_weight.mean(),
            'lambda_t': lambda_t,
        }
