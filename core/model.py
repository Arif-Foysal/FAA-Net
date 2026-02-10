"""
EDA-Net: Entropy-Dynamic Attention Network for Network Intrusion Detection.

Replaces fixed-temperature attention with Entropy-Dynamic Temperature (EDT)
attention that adapts softmax sharpness per-sample based on the information
entropy of attention logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .edt_attention import EDTAttention


class MinorityPrototypeGenerator:
    """Generate prototypes from minority class samples using K-Means clustering."""

    def __init__(self, n_prototypes=8, random_state=42):
        self.n_prototypes = n_prototypes
        self.kmeans = KMeans(n_clusters=n_prototypes, random_state=random_state, n_init=10)
        self.prototypes = None

    def fit(self, X_minority):
        if len(X_minority) < self.n_prototypes:
            self.prototypes = X_minority[:self.n_prototypes]
        else:
            self.kmeans.fit(X_minority)
            self.prototypes = self.kmeans.cluster_centers_
        return self.prototypes

    def get_prototypes_tensor(self, device):
        return torch.FloatTensor(self.prototypes).to(device)


# ---------------------------------------------------------------------------
#  EDT Attention Head & Multi-Head Wrapper
# ---------------------------------------------------------------------------

class EDTAttentionHead(nn.Module):
    """
    Single head of Entropy-Dynamic Temperature prototype attention.

    Uses learnable prototype keys/values and modulates the softmax
    temperature per sample based on entropy of the raw attention logits.
    """

    def __init__(self, input_dim, attention_dim=32, n_prototypes=4,
                 tau_min=0.1, tau_max=5.0, tau_hidden_dim=32,
                 edt_mode='learned', dropout=0.1, normalize_entropy=True):
        super().__init__()
        self.attention_dim = attention_dim
        self.n_prototypes = n_prototypes

        # Query projection
        self.query = nn.Linear(input_dim, attention_dim)

        # Learnable prototype keys and values
        self.prototype_keys = nn.Parameter(
            torch.randn(n_prototypes, attention_dim) * 0.02
        )
        self.prototype_values = nn.Parameter(
            torch.randn(n_prototypes, attention_dim) * 0.02
        )

        # EDT attention mechanism (core innovation)
        self.edt_attention = EDTAttention(
            d_k=attention_dim,
            n_keys=n_prototypes,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_hidden_dim=tau_hidden_dim,
            edt_mode=edt_mode,
            dropout=dropout,
            normalize_entropy=normalize_entropy
        )

        # Output projection + normalisation
        self.output_proj = nn.Linear(attention_dim, attention_dim)
        self.layer_norm = nn.LayerNorm(attention_dim)

    def initialize_prototypes(self, prototype_features):
        """Initialise prototype keys/values from data-derived minority features."""
        with torch.no_grad():
            proto_tensor = prototype_features[:self.n_prototypes]
            if proto_tensor.shape[0] < self.n_prototypes:
                padding = torch.randn(
                    self.n_prototypes - proto_tensor.shape[0],
                    proto_tensor.shape[1], device=proto_tensor.device
                ) * 0.02
                proto_tensor = torch.cat([proto_tensor, padding], dim=0)
            projected = self.query(proto_tensor)
            self.prototype_keys.data = projected.clone()
            self.prototype_values.data = projected.clone()
            # Store initial positions for prototype anchoring regularization
            self.register_buffer('_initial_keys', projected.clone())
            self.register_buffer('_initial_values', projected.clone())

    def prototype_anchor_loss(self):
        """MSE between current and initial prototype positions (prevents drift)."""
        if not hasattr(self, '_initial_keys') or self._initial_keys is None:
            return torch.tensor(0.0, device=self.prototype_keys.device)
        return (
            F.mse_loss(self.prototype_keys, self._initial_keys) +
            F.mse_loss(self.prototype_values, self._initial_values)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            output:       (batch_size, attention_dim)
            attn_weights: (batch_size, n_prototypes)
            edt_info:     dict with entropy and tau
        """
        q = self.query(x)  # (B, attention_dim)
        output, attn_weights, edt_info = self.edt_attention(
            q, self.prototype_keys, self.prototype_values
        )
        output = self.output_proj(output)
        output = self.layer_norm(output)
        return output, attn_weights, edt_info


class MultiHeadEDT(nn.Module):
    """Multi-head EDT attention with head fusion and residual connection."""

    def __init__(self, input_dim, num_heads=4, attention_dim=32, n_prototypes=4,
                 tau_min=0.1, tau_max=5.0, tau_hidden_dim=32,
                 edt_mode='learned', dropout=0.1, normalize_entropy=True):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim

        self.heads = nn.ModuleList([
            EDTAttentionHead(
                input_dim=input_dim,
                attention_dim=attention_dim,
                n_prototypes=n_prototypes,
                tau_min=tau_min,
                tau_max=tau_max,
                tau_hidden_dim=tau_hidden_dim,
                edt_mode=edt_mode,
                dropout=dropout,
                normalize_entropy=normalize_entropy
            )
            for _ in range(num_heads)
        ])

        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        self.final_proj = nn.Linear(attention_dim * num_heads, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def initialize_all_prototypes(self, prototype_features, device):
        proto_tensor = torch.FloatTensor(prototype_features).to(device)
        for head in self.heads:
            head.initialize_prototypes(proto_tensor)

    def prototype_anchor_loss(self):
        """Sum prototype anchor losses across all attention heads."""
        return sum(head.prototype_anchor_loss() for head in self.heads)

    def forward(self, x):
        head_outputs = []
        head_attentions = []
        head_edt_infos = []

        for head in self.heads:
            h_out, h_attn, h_edt = head(x)
            head_outputs.append(h_out)
            head_attentions.append(h_attn)
            head_edt_infos.append(h_edt)

        combined = torch.cat(head_outputs, dim=-1)
        output = self.final_proj(combined)
        output = self.layer_norm(output + x)  # Residual connection

        # Aggregate EDT info across heads for analysis
        all_entropy = torch.stack([info['entropy'] for info in head_edt_infos], dim=0)
        all_tau = torch.stack([info['tau'] for info in head_edt_infos], dim=0)

        edt_info = {
            'head_weights': F.softmax(self.head_weights, dim=0).detach(),
            'head_attentions': [a.detach() for a in head_attentions],
            'mean_entropy': all_entropy.mean(dim=0).detach(),   # (B, 1)
            'mean_tau': all_tau.mean(dim=0).detach(),           # (B, 1)
            'per_head_entropy': [info['entropy'] for info in head_edt_infos],
            'per_head_tau': [info['tau'] for info in head_edt_infos],
        }

        return output, edt_info


# ---------------------------------------------------------------------------
#  EDA-Net: Main Model
# ---------------------------------------------------------------------------

class EDANet(nn.Module):
    """
    EDA-Net: Entropy-Dynamic Attention Network for Network Intrusion Detection.

    Architecture:
        Input → BatchNorm → Multi-Head EDT Attention → SE Block
              → Hidden MLP Blocks (with residuals) → Classifier Head

    The EDT attention adaptively modulates softmax temperature per sample
    based on the information entropy of the attention logits:
        - Ambiguous samples (high entropy) → sharp attention (low τ)
        - Confident samples (low entropy)  → smooth attention (high τ)
    """

    def __init__(self, input_dim, num_heads=4, attention_dim=32, n_prototypes=8,
                 hidden_units=[256, 128, 64], dropout_rate=0.3, attention_dropout=0.1,
                 tau_min=0.1, tau_max=5.0, tau_hidden_dim=32, edt_mode='learned',
                 normalize_entropy=True, num_classes=1, output_logits=False):
        super().__init__()

        self.input_dim = input_dim
        self.output_logits = output_logits
        self.edt_mode = edt_mode

        # Input normalisation
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Multi-Head EDT Attention (core innovation)
        self.edt_attention = MultiHeadEDT(
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            n_prototypes=n_prototypes,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_hidden_dim=tau_hidden_dim,
            edt_mode=edt_mode,
            dropout=attention_dropout,
            normalize_entropy=normalize_entropy
        )

        # Squeeze-and-Excitation block
        self.se_block = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )

        # Hidden blocks with residuals
        self.hidden_blocks = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_units:
            block = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.hidden_blocks.append(block)

            if prev_dim != hidden_dim:
                self.residual_projections.append(nn.Linear(prev_dim, hidden_dim))
            else:
                self.residual_projections.append(nn.Identity())
            prev_dim = hidden_dim

        # Final classifier
        self.classifier_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, num_classes)
        )

        self.last_edt_info = None

    def prototype_anchor_loss(self):
        """Prototype anchoring regularization: prevents prototype drift from initialization."""
        return self.edt_attention.prototype_anchor_loss()

    def forward(self, x, return_edt_info=False):
        # Input normalisation
        x = self.input_norm(x)

        # EDT Attention
        attended, edt_info = self.edt_attention(x)

        # Squeeze-and-Excitation
        se_weights = self.se_block(attended)
        attended = attended * se_weights

        # Hidden blocks with residuals
        h = attended
        for block, res_proj in zip(self.hidden_blocks, self.residual_projections):
            residual = res_proj(h)
            h = block(h) + residual

        # Final classification
        logits = self.classifier_head(h)

        if self.output_logits:
            output = logits
        else:
            output = torch.sigmoid(logits)

        # Store for post-hoc analysis
        self.last_edt_info = {
            'edt_attention': edt_info,
            'se_weights': se_weights.detach()
        }

        if return_edt_info:
            return output, self.last_edt_info
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
