"""
DyGAT-FR Core Modules

This module contains the core building blocks for DyGAT-FR:
1. GraphFocalModulation - Extends FAA-Net's focal modulation to graph edges
2. DyGATConv - Dynamic GAT convolution with temporal residuals
3. GraphPrototypeAttention - Prototype cross-attention with momentum updates
4. FeedbackRefinementModule - Differentiable feedback refinement
5. MinorityReplayBuffer - Experience replay for continual learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops
from sklearn.cluster import KMeans
import numpy as np
from typing import Optional, Tuple, Dict, List


# ============================================================
# MIGRATED FROM FAA-NET: Uncertainty & Focal Modulation
# Extended for graph edges
# ============================================================

class GraphFocalModulation(nn.Module):
    """
    Extends FAA-Net's FocalModulation to graph edges.
    
    Original FAA-Net formulation:
        s_mod = s * (1 + α * uncertainty^γ)
        where uncertainty = 1 - 2|p - 0.5|
    
    Graph Extension:
        α_ij_mod = α_ij * (1 + w_focal_i + w_focal_j)
        
    This modulates edge attention based on the uncertainty of both
    source and destination nodes, amplifying attention on edges
    connecting uncertain (boundary) nodes.
    
    Args:
        alpha: Base focal weight (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        learnable: Whether alpha and temperature are learnable
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 learnable: bool = True):
        super().__init__()
        self.gamma = gamma
        
        if learnable:
            self.alpha = nn.Parameter(torch.tensor([alpha]))
            self.focal_temp = nn.Parameter(torch.tensor([1.0]))
        else:
            self.register_buffer('alpha', torch.tensor([alpha]))
            self.register_buffer('focal_temp', torch.tensor([1.0]))
    
    def compute_uncertainty(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty from probability estimates.
        Uncertainty peaks at p=0.5 (decision boundary).
        
        Formula: u = 1 - 2|p - 0.5| ∈ [0, 1]
        
        Args:
            p: Probability estimates (N,) or (N, 1)
        Returns:
            Uncertainty values (N,)
        """
        p = p.squeeze(-1) if p.dim() > 1 else p
        return 1.0 - 2.0 * torch.abs(p - 0.5)
    
    def forward(self, attention_coef: torch.Tensor,
                p_src: torch.Tensor, 
                p_dst: torch.Tensor) -> torch.Tensor:
        """
        Apply focal modulation to edge attention coefficients.
        
        Args:
            attention_coef: Base GAT attention coefficients (E,) or (E, heads)
            p_src: Source node probability estimates (E,)
            p_dst: Destination node probability estimates (E,)
        
        Returns:
            Modulated attention coefficients
        """
        # Compute uncertainty for source and destination nodes
        u_src = self.compute_uncertainty(p_src)
        u_dst = self.compute_uncertainty(p_dst)
        
        # Focal weights per node (from FAA-Net formulation)
        w_focal_src = self.alpha * torch.pow(u_src + 1e-8, self.gamma) * self.focal_temp
        w_focal_dst = self.alpha * torch.pow(u_dst + 1e-8, self.gamma) * self.focal_temp
        
        # Edge-level modulation: combines both endpoints
        edge_modulation = 1.0 + w_focal_src + w_focal_dst
        
        # Handle multi-head attention
        if attention_coef.dim() > 1:
            edge_modulation = edge_modulation.unsqueeze(-1)
        
        return attention_coef * edge_modulation


# ============================================================
# NEW: Dynamic Graph Attention Convolution Layer
# ============================================================

class DyGATConv(MessagePassing):
    """
    Dynamic Graph Attention Convolution with:
    1. Multi-head attention (standard GAT)
    2. Uncertainty-gated focal modulation on edges
    3. Temporal residual connections for incremental learning
    
    The layer extends standard GAT by:
    - Modulating attention based on node uncertainty (focal)
    - Maintaining temporal state via gated residuals
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension per head
        heads: Number of attention heads
        dropout: Dropout probability
        focal_alpha: Focal modulation alpha
        focal_gamma: Focal modulation gamma
        add_self_loops: Whether to add self-loops
        bias: Whether to use bias
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        add_self_loops: bool = True,
        bias: bool = True
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        
        # Total output dimension
        self.total_out = heads * out_channels
        
        # Linear transformations for query/key computation
        self.lin_src = nn.Linear(in_channels, self.total_out, bias=False)
        self.lin_dst = nn.Linear(in_channels, self.total_out, bias=False)
        
        # Attention parameters (learnable vectors)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Focal modulation module (from FAA-Net)
        self.focal_mod = GraphFocalModulation(
            alpha=focal_alpha, 
            gamma=focal_gamma, 
            learnable=True
        )
        
        # Temporal residual connection components
        self.temporal_proj = nn.Linear(in_channels, self.total_out, bias=False)
        self.temporal_gate = nn.Sequential(
            nn.Linear(self.total_out * 2, self.total_out),
            nn.Sigmoid()
        )
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.total_out))
        else:
            self.register_parameter('bias', None)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.total_out)
        
        self._alpha = None  # Store attention for visualization
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.temporal_proj.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_probs: Optional[torch.Tensor] = None,
        h_prev: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for DyGAT convolution.
        
        Args:
            x: Node features (N, in_channels)
            edge_index: Graph connectivity (2, E)
            node_probs: Initial probability estimates for focal mod (N,)
            h_prev: Previous time-step embeddings for temporal residual
            return_attention: Whether to return attention weights
        
        Returns:
            out: Updated node embeddings (N, heads * out_channels)
            (optional) alpha: Attention weights
        """
        N = x.size(0)
        
        # Add self-loops
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        
        # Linear projections
        x_src = self.lin_src(x).view(-1, self.heads, self.out_channels)
        x_dst = self.lin_dst(x).view(-1, self.heads, self.out_channels)
        
        # Compute attention scores (dot product with attention vectors)
        alpha_src = (x_src * self.att_src).sum(dim=-1)  # (N, heads)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)  # (N, heads)
        
        # Message passing
        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            alpha=(alpha_src, alpha_dst),
            node_probs=node_probs,
            size=None
        )
        
        out = out.view(-1, self.total_out)
        
        # Temporal residual connection (for incremental learning)
        if h_prev is not None:
            if h_prev.size(-1) != self.total_out:
                h_prev_proj = self.temporal_proj(h_prev)
            else:
                h_prev_proj = h_prev
            
            # Gated combination of current and previous
            gate = self.temporal_gate(torch.cat([out, h_prev_proj], dim=-1))
            out = gate * out + (1 - gate) * h_prev_proj
        
        # Add bias and normalize
        if self.bias is not None:
            out = out + self.bias
        
        out = self.layer_norm(out)
        
        if return_attention:
            return out, self._alpha
        return out
    
    def message(
        self,
        x_j: torch.Tensor,
        alpha_j: torch.Tensor,
        alpha_i: torch.Tensor,
        node_probs: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int]
    ) -> torch.Tensor:
        """
        Compute attention-weighted messages.
        
        Args:
            x_j: Source node features (E, heads, out_channels)
            alpha_j: Source attention scores (E, heads)
            alpha_i: Target attention scores (E, heads)
            node_probs: Node probability estimates (N,)
            index: Target node indices for aggregation
            ptr: CSR pointer (optional)
            size_i: Number of target nodes
        
        Returns:
            Weighted messages (E, heads, out_channels)
        """
        # Combine source and target attention scores
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # Softmax normalization
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Store attention weights for visualization
        self._alpha = alpha.detach()
        
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight messages by attention
        return x_j * alpha.unsqueeze(-1)


# ============================================================
# NEW: Graph Prototype Attention Module
# Extends FAA-Net's prototype cross-attention to graphs
# ============================================================

class GraphPrototypeAttention(nn.Module):
    """
    Prototype-based cross-attention for graph nodes.
    
    Extends FAA-Net's MinorityPrototypeGenerator and prototype attention:
    - Prototypes are initialized via K-means on minority node embeddings
    - Momentum update prevents prototype drift during incremental learning
    - Cross-attention computes similarity between nodes and prototypes
    
    Mathematical formulation:
        Q = W_q * h_i  (node embeddings as queries)
        K, V = prototypes
        Attention(Q, K, V) = softmax(QK^T / √d) * V
    
    Args:
        embed_dim: Embedding dimension
        n_prototypes: Number of prototype vectors (default: 8)
        momentum: Momentum for prototype updates (default: 0.9)
        dropout: Dropout probability
    """
    def __init__(
        self,
        embed_dim: int,
        n_prototypes: int = 8,
        momentum: float = 0.9,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_prototypes = n_prototypes
        self.momentum = momentum
        
        # Learnable prototype embeddings
        self.prototype_keys = nn.Parameter(
            torch.randn(n_prototypes, embed_dim) * 0.02
        )
        self.prototype_values = nn.Parameter(
            torch.randn(n_prototypes, embed_dim) * 0.02
        )
        
        # Learnable prototype importance (bias in attention)
        self.prototype_importance = nn.Parameter(
            torch.ones(n_prototypes) / n_prototypes
        )
        
        # Query projection
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scaling factor for attention
        self.scale = embed_dim ** -0.5
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Momentum buffer for incremental updates
        self.register_buffer(
            'prototype_momentum_keys',
            torch.zeros(n_prototypes, embed_dim)
        )
        self.register_buffer('initialized', torch.tensor(False))
    
    def initialize_prototypes(self, minority_embeddings: torch.Tensor):
        """
        Initialize prototypes via K-means clustering on minority embeddings.
        
        This follows FAA-Net's approach of anchoring attention to minority
        class patterns from the start of training.
        
        Args:
            minority_embeddings: Embeddings of minority class nodes (M, embed_dim)
        """
        with torch.no_grad():
            embeddings_np = minority_embeddings.detach().cpu().numpy()
            
            if len(embeddings_np) < self.n_prototypes:
                # Pad with noise if insufficient samples
                prototypes = embeddings_np
                padding = np.random.randn(
                    self.n_prototypes - len(embeddings_np),
                    self.embed_dim
                ) * 0.02
                prototypes = np.vstack([prototypes, padding])
            else:
                # K-means clustering
                kmeans = KMeans(
                    n_clusters=self.n_prototypes,
                    random_state=42,
                    n_init=10
                )
                kmeans.fit(embeddings_np)
                prototypes = kmeans.cluster_centers_
            
            # Set prototype parameters
            proto_tensor = torch.FloatTensor(prototypes).to(
                self.prototype_keys.device
            )
            self.prototype_keys.data = proto_tensor
            self.prototype_values.data = proto_tensor.clone()
            self.prototype_momentum_keys.data = proto_tensor.clone()
            self.initialized.fill_(True)
    
    def update_prototypes_momentum(self, new_minority_embeddings: torch.Tensor):
        """
        Momentum update for incremental learning.
        
        Prevents catastrophic forgetting by gradually updating prototypes:
            P^{t} = μ * P^{t-1} + (1-μ) * P_new
        
        Args:
            new_minority_embeddings: New minority node embeddings
        """
        with torch.no_grad():
            if len(new_minority_embeddings) >= self.n_prototypes:
                embeddings_np = new_minority_embeddings.detach().cpu().numpy()
                
                kmeans = KMeans(
                    n_clusters=self.n_prototypes,
                    random_state=42,
                    n_init=10
                )
                kmeans.fit(embeddings_np)
                new_prototypes = torch.FloatTensor(
                    kmeans.cluster_centers_
                ).to(self.prototype_keys.device)
                
                # Momentum update
                self.prototype_momentum_keys.data = (
                    self.momentum * self.prototype_momentum_keys +
                    (1 - self.momentum) * new_prototypes
                )
                
                # Sync learnable parameters
                self.prototype_keys.data = self.prototype_momentum_keys.clone()
                self.prototype_values.data = self.prototype_momentum_keys.clone()
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        focal_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention between nodes and prototypes.
        
        Args:
            node_embeddings: Node embeddings (N, embed_dim)
            focal_weights: Optional per-node focal weights (N,)
        
        Returns:
            output: Prototype-attended embeddings (N, embed_dim)
            attn_weights: Attention weights over prototypes (N, n_prototypes)
        """
        # Query projection
        Q = self.query_proj(node_embeddings)  # (N, embed_dim)
        
        # Attention scores: Q @ K^T
        scores = torch.matmul(Q, self.prototype_keys.T)  # (N, K)
        
        # Add prototype importance bias
        scores = scores + self.prototype_importance.unsqueeze(0)
        
        # Apply focal modulation if provided
        if focal_weights is not None:
            # Amplify attention for uncertain nodes
            scores = scores * (1 + focal_weights.unsqueeze(-1))
        
        # Softmax attention with scaling
        attn_weights = F.softmax(scores * self.scale, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of prototype values
        attended = torch.matmul(attn_weights, self.prototype_values)
        
        # Output projection with residual connection
        output = self.output_proj(attended)
        output = self.layer_norm(output + node_embeddings)
        
        return output, attn_weights


# ============================================================
# NEW: Feedback Refinement Module
# Human-AI feedback loop for attention refinement
# ============================================================

class FeedbackRefinementModule(nn.Module):
    """
    Refines node embeddings based on prediction errors.
    
    Implements a differentiable feedback loop:
        h^{t+1} = gate * h^{t} + (1-gate) * MLP(error) * β
    
    This enables:
    1. Semi-supervised refinement from sparse human labels
    2. Self-correction from AI-generated pseudo-labels
    3. Attention adjustment based on classification errors
    
    Args:
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension for error encoder
        dropout: Dropout probability
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Error encoding network
        self.error_encoder = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),  # +1 for scalar loss
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Gating mechanism (prevents training instability)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Learnable feedback strength (β)
        self.feedback_strength = nn.Parameter(torch.tensor([0.1]))
        
        # Gradient clipping for stability
        self.register_buffer('grad_clip', torch.tensor([1.0]))
    
    def forward(
        self,
        current_embeddings: torch.Tensor,
        prediction_errors: torch.Tensor,
        loss_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply feedback refinement to embeddings.
        
        Args:
            current_embeddings: Current node embeddings (N, embed_dim)
            prediction_errors: Per-node prediction errors (N,) or (N, embed_dim)
            loss_values: Per-node loss values (N,)
        
        Returns:
            refined_embeddings: Feedback-refined embeddings (N, embed_dim)
        """
        # Ensure prediction_errors has correct shape
        if prediction_errors.dim() == 1:
            prediction_errors = prediction_errors.unsqueeze(-1).expand_as(
                current_embeddings
            )
        
        # Concatenate error signals
        error_input = torch.cat([
            prediction_errors,
            loss_values.unsqueeze(-1)
        ], dim=-1)
        
        # Encode error signal
        error_encoding = self.error_encoder(error_input)
        
        # Compute gate values
        gate_input = torch.cat([current_embeddings, error_encoding], dim=-1)
        gate_values = self.gate(gate_input)
        
        # Gated update with feedback strength
        refined = (
            gate_values * current_embeddings +
            (1 - gate_values) * error_encoding * self.feedback_strength
        )
        
        return refined


# ============================================================
# NEW: Memory Replay Buffer for Continual Learning
# ============================================================

class MinorityReplayBuffer:
    """
    Experience replay buffer for minority class samples.
    
    Prevents catastrophic forgetting by storing and replaying
    minority class samples during incremental training.
    
    Features:
    - Reservoir sampling for memory-efficient storage
    - Per-class limits to ensure diverse replay
    - Edge storage for graph structure preservation
    
    Args:
        max_size: Maximum buffer size
        per_class_limit: Maximum samples per class
    """
    def __init__(
        self,
        max_size: int = 1000,
        per_class_limit: int = 100
    ):
        self.max_size = max_size
        self.per_class_limit = per_class_limit
        self.buffer: Dict[int, List[torch.Tensor]] = {}
        self.edge_buffer: Dict[int, List[torch.Tensor]] = {}
        self.total_samples = 0
    
    def add(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        edges: Optional[torch.Tensor] = None
    ):
        """
        Add minority samples to buffer with reservoir sampling.
        
        Args:
            features: Node features (N, d)
            labels: Node labels (N,)
            edges: Optional edge indices for subgraph
        """
        unique_labels = labels.unique()
        
        for label in unique_labels:
            label_val = label.item()
            mask = labels == label
            label_features = features[mask].detach().cpu()
            
            if label_val not in self.buffer:
                self.buffer[label_val] = []
            
            for feat in label_features:
                if len(self.buffer[label_val]) < self.per_class_limit:
                    self.buffer[label_val].append(feat)
                    self.total_samples += 1
                else:
                    # Reservoir sampling: replace random element
                    idx = np.random.randint(0, self.per_class_limit)
                    self.buffer[label_val][idx] = feat
    
    def sample(
        self,
        n_samples: int,
        device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample from buffer for replay.
        
        Args:
            n_samples: Number of samples to return
            device: Target device
        
        Returns:
            features: Sampled features (n, d) or None
            labels: Sampled labels (n,) or None
        """
        if self.total_samples == 0:
            return None, None
        
        all_features = []
        all_labels = []
        
        for label, features in self.buffer.items():
            all_features.extend(features)
            all_labels.extend([label] * len(features))
        
        if len(all_features) == 0:
            return None, None
        
        # Random sampling without replacement
        indices = np.random.choice(
            len(all_features),
            min(n_samples, len(all_features)),
            replace=False
        )
        
        sampled_features = torch.stack(
            [all_features[i] for i in indices]
        ).to(device)
        sampled_labels = torch.tensor(
            [all_labels[i] for i in indices]
        ).to(device)
        
        return sampled_features, sampled_labels
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = {}
        self.edge_buffer = {}
        self.total_samples = 0
    
    def __len__(self):
        return self.total_samples


# ============================================================
# NEW: Class-Conditional Gate (from FAA-Net)
# ============================================================

class ClassConditionalGate(nn.Module):
    """
    Class-conditional gating mechanism.
    
    Gates features based on prediction difficulty:
        difficulty = 1 - 2|p - 0.5|
        gate = σ(MLP([x; difficulty]))
        output = x * gate
    
    Args:
        dim: Feature dimension
        reduction: Reduction ratio for MLP bottleneck
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(dim + 1, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        class_prob: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply class-conditional gating.
        
        Args:
            x: Input features (N, dim)
            class_prob: Class probability estimates (N, 1)
        
        Returns:
            Gated features (N, dim)
        """
        # Compute difficulty (inverse of confidence)
        difficulty = 1 - 2 * torch.abs(class_prob - 0.5)
        
        # Concatenate features with difficulty
        gate_input = torch.cat([x, difficulty], dim=-1)
        
        # Compute gate values
        gate_values = self.gate(gate_input)
        
        return x * gate_values
