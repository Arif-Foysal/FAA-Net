"""
Entropy-Dynamic Temperature (EDT) Attention Module.

Core innovation of EDA-Net: Modulate the softmax temperature dynamically
based on the information entropy of attention logits, enabling per-sample
adaptive attention sharpness for network intrusion detection.

Mathematical Formulation:
    1. Feature Entropy:  H(x) = -Σ p(x) log p(x)
       where p(x) = softmax(QK^T / √d_k)

    2. Dynamic Temperature:  τ(x) = τ_min + (τ_max - τ_min) · σ(MLP(H̃(x)))
       where H̃ is normalized entropy in [0, 1]

    3. EDT-Modulated Attention:  Attn = softmax(QK^T / (τ(x) · √d_k)) · V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EntropyDynamicTemperature(nn.Module):
    """
    Maps normalized entropy → dynamic temperature via a learned MLP.

    Behaviour:
        High entropy (ambiguous / hard sample) → low temperature → sharper attention
        Low entropy (confident / easy sample) → high temperature → smoother attention

    Modes:
        'learned'   : MLP maps entropy to temperature (full EDT)
        'heuristic' : Analytic inverse mapping τ = τ_max · (1 - H̃) + τ_min
        'fixed'     : Learnable but entropy-independent scalar temperature
    """

    def __init__(self, tau_min=0.1, tau_max=5.0, hidden_dim=32, mode='learned'):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.mode = mode

        if mode == 'learned':
            self.entropy_to_temp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            # Initialise to produce mid-range temperatures
            nn.init.xavier_uniform_(self.entropy_to_temp[0].weight)
            nn.init.zeros_(self.entropy_to_temp[0].bias)
            nn.init.xavier_uniform_(self.entropy_to_temp[2].weight)
            nn.init.zeros_(self.entropy_to_temp[2].bias)

        elif mode == 'fixed':
            self.fixed_tau = nn.Parameter(torch.tensor([1.0]))

    def forward(self, entropy_normalized):
        """
        Args:
            entropy_normalized: (batch_size, 1) normalised entropy in [0, 1]
        Returns:
            tau: (batch_size, 1) dynamic temperature in [tau_min, tau_max]
        """
        if self.mode == 'learned':
            scale = torch.sigmoid(self.entropy_to_temp(entropy_normalized))  # (0, 1)
            tau = self.tau_min + (self.tau_max - self.tau_min) * scale

        elif self.mode == 'heuristic':
            # Analytic inverse: high entropy → low tau
            tau = self.tau_max * (1.0 - entropy_normalized) + self.tau_min

        elif self.mode == 'fixed':
            tau = self.fixed_tau.abs().clamp(min=self.tau_min, max=self.tau_max)
            tau = tau.expand_as(entropy_normalized)

        else:
            raise ValueError(f"Unknown EDT mode: {self.mode}")

        return tau

    def extra_repr(self):
        return f"mode={self.mode}, tau_range=[{self.tau_min}, {self.tau_max}]"


class EDTAttention(nn.Module):
    """
    Entropy-Dynamic Temperature Scaled Dot-Product Attention.

    Computes attention with a per-sample dynamic temperature derived
    from the information entropy of the raw attention logits. This replaces
    the fixed 1/√d_k scaling in standard attention.
    """

    def __init__(self, d_k, n_keys, tau_min=0.1, tau_max=5.0,
                 tau_hidden_dim=32, edt_mode='learned', dropout=0.1,
                 normalize_entropy=True):
        """
        Args:
            d_k: Dimension of queries/keys
            n_keys: Number of keys (prototypes)
            tau_min: Minimum temperature (lower bound)
            tau_max: Maximum temperature (upper bound)
            tau_hidden_dim: Hidden dimension for entropy→temperature MLP
            edt_mode: 'learned', 'heuristic', or 'fixed'
            dropout: Attention dropout rate
            normalize_entropy: Whether to normalize entropy to [0, 1]
        """
        super().__init__()
        self.d_k = d_k
        self.n_keys = n_keys
        self.base_scale = d_k ** -0.5  # 1 / √d_k
        self.normalize_entropy = normalize_entropy
        self.max_entropy = math.log(max(n_keys, 2))  # log(K) for normalisation

        self.edt = EntropyDynamicTemperature(
            tau_min=tau_min,
            tau_max=tau_max,
            hidden_dim=tau_hidden_dim,
            mode=edt_mode
        )
        self.dropout = nn.Dropout(dropout)

    def compute_entropy(self, logits):
        """
        Compute (optionally normalised) information entropy of attention logits.

        Args:
            logits: (batch_size, n_keys) raw attention logits
        Returns:
            entropy: (batch_size, 1) entropy value
        """
        # Temporary softmax with standard scaling for entropy computation
        p = F.softmax(logits * self.base_scale, dim=-1)  # (B, K)

        # H(x) = -Σ p(x) log p(x)
        log_p = torch.log(p + 1e-8)
        entropy = -torch.sum(p * log_p, dim=-1, keepdim=True)  # (B, 1)

        if self.normalize_entropy:
            # Normalise to [0, 1] by dividing by max possible entropy log(K)
            entropy = entropy / (self.max_entropy + 1e-8)
            entropy = entropy.clamp(0.0, 1.0)

        return entropy

    def forward(self, q, keys, values):
        """
        Args:
            q:      (batch_size, d_k)      query vectors
            keys:   (n_keys, d_k)          key vectors (prototypes)
            values: (n_keys, d_v)          value vectors (prototypes)
        Returns:
            output:       (batch_size, d_v)     attended output
            attn_weights: (batch_size, n_keys)  attention weights
            edt_info:     dict with entropy and tau for analysis
        """
        # Step 0: Raw attention logits
        raw_logits = torch.matmul(q, keys.T)  # (B, K)

        # Step 1: Compute per-sample entropy
        entropy = self.compute_entropy(raw_logits)  # (B, 1)

        # Step 2: Dynamic temperature from entropy
        tau = self.edt(entropy)  # (B, 1)

        # Step 3: EDT-modulated attention
        scaled_logits = raw_logits * self.base_scale / (tau + 1e-8)  # (B, K)
        attn_weights = F.softmax(scaled_logits, dim=-1)  # (B, K)
        attn_weights = self.dropout(attn_weights)

        # Step 4: Weighted combination of values
        output = torch.matmul(attn_weights, values)  # (B, d_v)

        edt_info = {
            'entropy': entropy.detach(),      # (B, 1)
            'tau': tau.detach(),              # (B, 1)
            'raw_logits': raw_logits.detach() # (B, K)
        }

        return output, attn_weights, edt_info
