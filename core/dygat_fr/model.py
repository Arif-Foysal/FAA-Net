"""
DyGAT-FR: Full Model Architecture

Dynamic Graph Attention Network with Feedback Refinement
for Incremental Imbalanced Learning

This module implements the complete DyGAT-FR architecture:
1. Probability Estimator - Lightweight network for uncertainty estimation
2. Stacked DyGAT Layers - Dynamic GAT with focal modulation
3. Graph Prototype Attention - Cross-attention to minority prototypes
4. Feedback Refinement - Human-AI loop for attention adjustment
5. Classification Head - Final prediction layer

The architecture extends FAA-Net's FAIIA to dynamic graphs with:
- Temporal residual connections for incremental adaptation
- Momentum-based prototype updates to prevent forgetting
- Memory replay for continual learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any

from .modules import (
    DyGATConv,
    GraphPrototypeAttention,
    FeedbackRefinementModule,
    MinorityReplayBuffer,
    ClassConditionalGate
)


class DyGATFR(nn.Module):
    """
    DyGAT-FR: Dynamic Graph Attention Network with Feedback Refinement
    
    A graph neural network designed for incremental learning under class
    imbalance. Extends FAA-Net's focal-aware attention to dynamic graphs.
    
    Architecture Overview:
    ┌─────────────────────────────────────────────────────────┐
    │ Input: Node Features X, Edge Index E                    │
    ├─────────────────────────────────────────────────────────┤
    │ 1. Probability Estimator → p_init (uncertainty source)  │
    │                                                         │
    │ 2. DyGAT Layers (with focal modulation)                 │
    │    ├── Layer 1: in → hidden                            │
    │    ├── Layer 2-N: hidden → hidden                      │
    │    └── Layer N+1: hidden → out                         │
    │                                                         │
    │ 3. Prototype Cross-Attention                            │
    │    └── Attend to K-means minority prototypes           │
    │                                                         │
    │ 4. Feedback Refinement (optional)                       │
    │    └── Refine based on prediction errors               │
    │                                                         │
    │ 5. Classification Head → logits                         │
    └─────────────────────────────────────────────────────────┘
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension (total, divided by heads)
        out_channels: Output embedding dimension
        num_classes: Number of output classes (1 for binary)
        num_layers: Number of DyGAT layers
        heads: Number of attention heads
        n_prototypes: Number of minority prototypes
        focal_alpha: Focal modulation alpha
        focal_gamma: Focal modulation gamma
        dropout: Dropout probability
        prototype_momentum: Momentum for prototype updates
        use_feedback: Whether to use feedback refinement
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_classes: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        n_prototypes: int = 8,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dropout: float = 0.3,
        prototype_momentum: float = 0.9,
        use_feedback: bool = True
    ):
        super().__init__()
        
        # Store configuration
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_feedback = use_feedback
        
        # ============================================================
        # 1. Probability Estimator (migrated from FAA-Net)
        # Lightweight MLP for initial uncertainty estimation
        # ============================================================
        self.prob_estimator = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # ============================================================
        # 2. Input Projection
        # ============================================================
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)
        
        # ============================================================
        # 3. Stacked DyGAT Layers with Focal Modulation
        # ============================================================
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Compute per-head output dimension
        assert hidden_channels % heads == 0, "hidden_channels must be divisible by heads"
        head_dim = hidden_channels // heads
        out_head_dim = out_channels // heads
        
        # Build DyGAT layers
        for i in range(num_layers):
            if i == 0:
                # First layer
                self.convs.append(DyGATConv(
                    hidden_channels, head_dim,
                    heads=heads,
                    dropout=dropout,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma
                ))
                self.norms.append(nn.LayerNorm(hidden_channels))
            elif i == num_layers - 1:
                # Final layer
                self.convs.append(DyGATConv(
                    hidden_channels, out_head_dim,
                    heads=heads,
                    dropout=dropout,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma
                ))
                self.norms.append(nn.LayerNorm(out_channels))
            else:
                # Hidden layers
                self.convs.append(DyGATConv(
                    hidden_channels, head_dim,
                    heads=heads,
                    dropout=dropout,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma
                ))
                self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Residual projections for dimension changes
        self.residual_proj = nn.Linear(hidden_channels, out_channels)
        
        # ============================================================
        # 4. Graph Prototype Cross-Attention
        # ============================================================
        self.prototype_attention = GraphPrototypeAttention(
            embed_dim=out_channels,
            n_prototypes=n_prototypes,
            momentum=prototype_momentum,
            dropout=dropout
        )
        
        # Class-conditional gate (from FAA-Net)
        self.class_gate = ClassConditionalGate(
            dim=out_channels,
            reduction=4
        )
        
        # ============================================================
        # 5. Feedback Refinement Module (novel component)
        # ============================================================
        if use_feedback:
            self.feedback_module = FeedbackRefinementModule(
                embed_dim=out_channels,
                hidden_dim=64,
                dropout=dropout
            )
        
        # ============================================================
        # 6. Classification Head
        # ============================================================
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, num_classes if num_classes > 2 else 1)
        )
        
        # ============================================================
        # Auxiliary Components
        # ============================================================
        
        # Memory replay buffer for continual learning
        self.replay_buffer = MinorityReplayBuffer(max_size=1000)
        
        # Store previous embeddings for temporal residual
        self.register_buffer('prev_embeddings', None)
        
        # Attention info storage for visualization
        self.last_attention_info: Optional[Dict[str, Any]] = None
    
    def compute_focal_weights(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute focal weights from probability estimates.
        
        Focal weight peaks at p=0.5 (maximum uncertainty).
        Formula: w = 1 - 2|p - 0.5|
        
        Args:
            p: Probability estimates (N, 1)
        
        Returns:
            Focal weights (N,)
        """
        p_squeezed = p.squeeze(-1) if p.dim() > 1 else p
        return 1.0 - 2.0 * torch.abs(p_squeezed - 0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        h_prev: Optional[torch.Tensor] = None,
        feedback_errors: Optional[torch.Tensor] = None,
        feedback_losses: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for DyGAT-FR.
        
        Args:
            x: Node features (N, in_channels)
            edge_index: Graph connectivity (2, E)
            batch: Batch assignment for batched graphs (N,)
            h_prev: Previous time-step embeddings for temporal residual
            feedback_errors: Per-node prediction errors for feedback refinement
            feedback_losses: Per-node loss values for feedback refinement
            return_embeddings: Whether to return intermediate embeddings
        
        Returns:
            logits: Classification logits (N, num_classes) or (N, 1)
            (optional) embeddings: Final node embeddings if return_embeddings=True
        """
        # ============================================================
        # 1. Initial Probability Estimation
        # ============================================================
        p_init = self.prob_estimator(x)
        focal_weights = self.compute_focal_weights(p_init)
        
        # ============================================================
        # 2. Input Projection
        # ============================================================
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.gelu(h)
        
        # ============================================================
        # 3. Stacked DyGAT Layers
        # ============================================================
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_residual = h
            
            # DyGAT convolution with focal modulation
            h = conv(
                h, edge_index,
                node_probs=p_init.squeeze(-1),
                h_prev=h_prev
            )
            h = norm(h)
            h = F.gelu(h)
            
            # Residual connection
            if h.size(-1) == h_residual.size(-1):
                h = h + h_residual
            elif i == self.num_layers - 1:
                # Final layer: project residual
                h = h + self.residual_proj(h_residual)
        
        # ============================================================
        # 4. Prototype Cross-Attention
        # ============================================================
        h_proto, proto_attn = self.prototype_attention(h, focal_weights)
        
        # Class-conditional gating
        h_gated = self.class_gate(h_proto, p_init)
        
        # ============================================================
        # 5. Feedback Refinement (if enabled)
        # ============================================================
        if self.use_feedback and feedback_errors is not None:
            if feedback_losses is None:
                feedback_losses = torch.zeros_like(feedback_errors)
            h_gated = self.feedback_module(
                h_gated, feedback_errors, feedback_losses
            )
        
        # Store for next time-step
        self.prev_embeddings = h_gated.detach()
        
        # ============================================================
        # 6. Classification
        # ============================================================
        logits = self.classifier(h_gated)
        
        # Store attention info for analysis
        self.last_attention_info = {
            'initial_prob': p_init.detach(),
            'focal_weights': focal_weights.detach(),
            'prototype_attention': proto_attn.detach(),
            'gat_attention': [conv._alpha for conv in self.convs if conv._alpha is not None]
        }
        
        if return_embeddings:
            return logits, h_gated
        return logits
    
    def initialize_prototypes(self, minority_embeddings: torch.Tensor):
        """
        Initialize prototypes from minority class embeddings.
        
        Should be called before training with embeddings from the
        minority (attack) class to anchor attention.
        
        Args:
            minority_embeddings: Minority class node embeddings (M, embed_dim)
        """
        self.prototype_attention.initialize_prototypes(minority_embeddings)
    
    def update_for_increment(self, new_minority_embeddings: torch.Tensor):
        """
        Update prototypes for a new data increment.
        
        Uses momentum update to prevent catastrophic forgetting:
            P_new = μ * P_old + (1-μ) * P_increment
        
        Args:
            new_minority_embeddings: New minority embeddings from increment
        """
        self.prototype_attention.update_prototypes_momentum(new_minority_embeddings)
    
    def add_to_replay_buffer(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ):
        """Add minority samples to replay buffer for continual learning."""
        self.replay_buffer.add(features, labels)
    
    def sample_replay(
        self,
        n_samples: int,
        device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample from replay buffer."""
        return self.replay_buffer.sample(n_samples, device)
    
    def get_prototype_keys(self) -> torch.Tensor:
        """Get current prototype keys for analysis."""
        return self.prototype_attention.prototype_keys.detach()
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DyGATFRLoss(nn.Module):
    """
    Combined loss function for DyGAT-FR.
    
    Combines three objectives:
    1. Focal Loss: Addresses class imbalance by down-weighting easy examples
    2. Contrastive Loss: Pulls minority samples toward prototypes
    3. Replay Loss: Prevents forgetting via loss on replayed samples
    
    Total Loss:
        L = L_focal(pred, y) + λ * L_contrastive(h, P) + μ * L_replay
    
    Args:
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        contrastive_weight: Weight for contrastive loss (λ)
        replay_weight: Weight for replay loss (μ)
        num_classes: Number of output classes
    """
    
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        contrastive_weight: float = 0.1,
        replay_weight: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.contrastive_weight = contrastive_weight
        self.replay_weight = replay_weight
        self.num_classes = num_classes
    
    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss for imbalanced classification.
        
        FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
        
        Args:
            logits: Prediction logits (N, C) or (N, 1)
            targets: Ground truth labels (N,)
        
        Returns:
            Scalar focal loss
        """
        if self.num_classes == 2 or logits.size(-1) == 1:
            # Binary focal loss
            probs = torch.sigmoid(logits.squeeze(-1))
            targets_float = targets.float()
            
            # p_t: probability of correct class
            p_t = probs * targets_float + (1 - probs) * (1 - targets_float)
            
            # Alpha weighting
            alpha_t = (
                self.focal_alpha * targets_float +
                (1 - self.focal_alpha) * (1 - targets_float)
            )
            
            # Focal weight
            focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
            
            # Binary cross-entropy
            bce = F.binary_cross_entropy(probs, targets_float, reduction='none')
            
            return (focal_weight * bce).mean()
        else:
            # Multi-class focal loss
            probs = F.softmax(logits, dim=-1)
            ce = F.cross_entropy(logits, targets, reduction='none')
            
            # p_t for correct class
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze()
            
            # Focal weight
            focal_weight = (1 - p_t) ** self.focal_gamma
            
            return (focal_weight * ce).mean()
    
    def contrastive_loss(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        labels: torch.Tensor,
        minority_label: int = 1
    ) -> torch.Tensor:
        """
        Contrastive loss to pull minority samples toward prototypes.
        
        Maximizes cosine similarity between minority embeddings and
        their nearest prototype.
        
        Args:
            embeddings: Node embeddings (N, d)
            prototypes: Prototype embeddings (K, d)
            labels: Node labels (N,)
            minority_label: Label value for minority class
        
        Returns:
            Scalar contrastive loss
        """
        # Normalize for cosine similarity
        embeddings = F.normalize(embeddings, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        
        # Compute similarity
        sim = torch.matmul(embeddings, prototypes.T)  # (N, K)
        
        # Filter to minority samples
        minority_mask = labels == minority_label
        if minority_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        minority_sim = sim[minority_mask]
        
        # Maximize similarity to nearest prototype (negative for minimization)
        max_sim = minority_sim.max(dim=-1)[0]
        
        return -max_sim.mean()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
        replay_logits: Optional[torch.Tensor] = None,
        replay_targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Prediction logits (N, C)
            targets: Ground truth labels (N,)
            embeddings: Node embeddings for contrastive loss
            prototypes: Prototype embeddings for contrastive loss
            replay_logits: Predictions on replayed samples
            replay_targets: Labels for replayed samples
        
        Returns:
            Total loss scalar
        """
        # Main focal loss
        loss = self.focal_loss(logits, targets)
        
        # Contrastive loss
        if embeddings is not None and prototypes is not None:
            cont_loss = self.contrastive_loss(embeddings, prototypes, targets)
            loss = loss + self.contrastive_weight * cont_loss
        
        # Replay loss
        if replay_logits is not None and replay_targets is not None:
            replay_loss = self.focal_loss(replay_logits, replay_targets)
            loss = loss + self.replay_weight * replay_loss
        
        return loss


# ============================================================
# Lightweight variant for edge deployment
# ============================================================

class DyGATFRLite(DyGATFR):
    """
    Lightweight DyGAT-FR for edge deployment.
    
    Reduces computational overhead by:
    - Fewer layers (2 instead of 3)
    - Fewer heads (2 instead of 4)
    - Smaller hidden dimension
    - No feedback module
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_classes: int = 2,
        heads: int = 2,
        n_prototypes: int = 4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dropout: float = 0.2
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_classes=num_classes,
            num_layers=2,
            heads=heads,
            n_prototypes=n_prototypes,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            dropout=dropout,
            prototype_momentum=0.95,  # Higher momentum for stability
            use_feedback=False  # Disable feedback for simplicity
        )
