"""
Ablation model variants for EDA-Net.

Provides:
    - VanillaDNN_Ablation:     No attention baseline
    - FixedTempNet_Ablation:   Standard fixed-temperature attention (no EDT)
    - HeuristicEDTNet_Ablation: Heuristic τ mapping (no learned MLP)
    - EDANet_Ablation:         Full EDA-Net with output_logits=True
"""

import torch
import torch.nn as nn
from core.model import EDANet


class VanillaDNN_Ablation(nn.Module):
    """Standard DNN without any attention mechanism. Outputs raw logits."""

    def __init__(self, input_dim, hidden_units=[256, 128, 64], dropout_rate=0.3):
        super(VanillaDNN_Ablation, self).__init__()
        self.output_logits = True

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FixedTempNet_Ablation(EDANet):
    """
    EDA-Net architecture with FIXED temperature attention (no EDT).
    Serves as the "attention without EDT" ablation baseline.
    """

    def __init__(self, input_dim, **kwargs):
        kwargs.setdefault('edt_mode', 'fixed')
        kwargs.setdefault('output_logits', True)
        super().__init__(input_dim=input_dim, **kwargs)


class HeuristicEDTNet_Ablation(EDANet):
    """
    EDA-Net with HEURISTIC temperature mapping (τ = τ_max·(1−H̃) + τ_min).
    No learned MLP — tests whether the learned mapping adds value.
    """

    def __init__(self, input_dim, **kwargs):
        kwargs.setdefault('edt_mode', 'heuristic')
        kwargs.setdefault('output_logits', True)
        super().__init__(input_dim=input_dim, **kwargs)


class EDANet_Ablation(EDANet):
    """
    Full EDA-Net with learned EDT, configured for ablation (output_logits=True).
    """

    def __init__(self, input_dim, **kwargs):
        kwargs.setdefault('edt_mode', 'learned')
        kwargs.setdefault('output_logits', True)
        super().__init__(input_dim=input_dim, **kwargs)
