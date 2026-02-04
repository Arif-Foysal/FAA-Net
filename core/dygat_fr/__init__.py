"""
DyGAT-FR: Dynamic Graph Attention Network with Feedback Refinement
for Incremental Imbalanced Learning

This module extends FAA-Net's Focal-Aware Imbalance-Integrated Attention (FAIIA)
to dynamic graphs for continual learning under class imbalance.

Key Components:
- GraphFocalModulation: Edge-level focal modulation based on node uncertainty
- DyGATConv: Dynamic GAT convolution with temporal residuals
- GraphPrototypeAttention: Prototype cross-attention with momentum updates
- FeedbackRefinementModule: Human-AI feedback loop for attention refinement
- MinorityReplayBuffer: Memory replay for continual learning

Reference:
- FAA-Net: Focal-Aware Attention Network for Network Intrusion Detection
- DyGAT-FR extends this to dynamic graphs with incremental adaptation
"""

from .modules import (
    GraphFocalModulation,
    DyGATConv,
    GraphPrototypeAttention,
    FeedbackRefinementModule,
    MinorityReplayBuffer
)
from .model import DyGATFR, DyGATFRLoss
from .trainer import DyGATFRTrainer

__all__ = [
    'GraphFocalModulation',
    'DyGATConv', 
    'GraphPrototypeAttention',
    'FeedbackRefinementModule',
    'MinorityReplayBuffer',
    'DyGATFR',
    'DyGATFRLoss',
    'DyGATFRTrainer'
]

__version__ = '1.0.0'
