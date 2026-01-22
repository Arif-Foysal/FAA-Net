
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class MinorityPrototypeGenerator:
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


class FocalModulation(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, learnable=True):
        super(FocalModulation, self).__init__()
        self.gamma = gamma
        if learnable:
            self.alpha = nn.Parameter(torch.tensor([alpha]))
            self.focal_temp = nn.Parameter(torch.tensor([1.0]))
        else:
            self.register_buffer('alpha', torch.tensor([alpha]))
            self.register_buffer('focal_temp', torch.tensor([1.0]))

    def forward(self, attention_scores, minority_prob):
        focal_weight = self.alpha * torch.pow(1 - minority_prob + 1e-8, self.gamma)
        focal_weight = focal_weight * self.focal_temp
        while focal_weight.dim() < attention_scores.dim():
            focal_weight = focal_weight.unsqueeze(-1)
        return attention_scores * (1 + focal_weight)


class ClassConditionalGate(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ClassConditionalGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim + 1, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x, class_prob):
        difficulty = 1 - 2 * torch.abs(class_prob - 0.5)
        gate_input = torch.cat([x, difficulty], dim=-1)
        gate_values = self.gate(gate_input)
        return x * gate_values


class FAIIAHead(nn.Module):
    def __init__(self, input_dim, attention_dim=32, n_prototypes=4,
                 focal_alpha=0.60, focal_gamma=2.0, dropout=0.1):
        super(FAIIAHead, self).__init__()
        self.attention_dim = attention_dim
        self.n_prototypes = n_prototypes
        self.scale = attention_dim ** -0.5
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.prototype_keys = nn.Parameter(torch.randn(n_prototypes, attention_dim) * 0.02)
        self.prototype_values = nn.Parameter(torch.randn(n_prototypes, attention_dim) * 0.02)
        self.prototype_importance = nn.Parameter(torch.ones(n_prototypes) / n_prototypes)
        self.focal_mod = FocalModulation(alpha=focal_alpha, gamma=focal_gamma, learnable=True)
        self.output_proj = nn.Linear(attention_dim * 2, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention_dim)

    def initialize_prototypes(self, prototype_features):
        with torch.no_grad():
            proto_tensor = prototype_features[:self.n_prototypes]
            if proto_tensor.shape[0] < self.n_prototypes:
                padding = torch.randn(self.n_prototypes - proto_tensor.shape[0],
                                     proto_tensor.shape[1], device=proto_tensor.device) * 0.02
                proto_tensor = torch.cat([proto_tensor, padding], dim=0)
            self.prototype_keys.data = self.query(proto_tensor)
            self.prototype_values.data = self.value(proto_tensor)

    def forward(self, x, minority_prob=None):
        batch_size = x.shape[0]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_scores = torch.bmm(q.unsqueeze(1), k.unsqueeze(2)).squeeze(-1)
        if minority_prob is not None:
            attention_scores = self.focal_mod(attention_scores, minority_prob)
        attention_weights = F.softmax(attention_scores * self.scale, dim=-1)
        attention_weights = self.dropout(attention_weights)
        self_attended = v * attention_weights

        proto_scores = torch.matmul(q.unsqueeze(1), self.prototype_keys.T).squeeze(1)
        if minority_prob is not None:
            proto_scores = self.focal_mod(proto_scores, minority_prob)
        proto_weights = F.softmax(proto_scores * self.scale, dim=-1) * F.softmax(self.prototype_importance, dim=0)
        proto_weights = self.dropout(proto_weights)
        proto_attended = torch.matmul(
            proto_weights.unsqueeze(1),
            self.prototype_values.unsqueeze(0).expand(batch_size, -1, -1)
        ).squeeze(1)

        combined = torch.cat([self_attended, proto_attended], dim=-1)
        output = self.output_proj(combined)
        output = self.layer_norm(output)
        return output, attention_weights.squeeze(1) if attention_weights.dim() > 1 else attention_weights


class MultiHeadFAIIA(nn.Module):
    def __init__(self, input_dim, num_heads=4, attention_dim=32, n_prototypes=4,
                 focal_alpha=0.60, focal_gamma=2.0, dropout=0.1):
        super(MultiHeadFAIIA, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.heads = nn.ModuleList([
            FAIIAHead(
                input_dim=input_dim,
                attention_dim=attention_dim,
                n_prototypes=n_prototypes,
                focal_alpha=focal_alpha * (1 + 0.1 * i),
                focal_gamma=focal_gamma,
                dropout=dropout
            )
            for i in range(num_heads)
        ])
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        self.final_proj = nn.Linear(attention_dim * num_heads, input_dim)
        self.class_gate = ClassConditionalGate(input_dim, reduction=4)
        self.layer_norm = nn.LayerNorm(input_dim)

    def initialize_all_prototypes(self, prototype_features, device):
        proto_tensor = torch.FloatTensor(prototype_features).to(device)
        for head in self.heads:
            head.initialize_prototypes(proto_tensor)

    def forward(self, x, minority_prob=None):
        head_outputs = []
        head_attentions = []
        for head in self.heads:
            h_out, h_attn = head(x, minority_prob)
            head_outputs.append(h_out)
            head_attentions.append(h_attn)
        combined = torch.cat(head_outputs, dim=-1)
        output = self.final_proj(combined)
        if minority_prob is not None:
            output = self.class_gate(output, minority_prob)
        output = self.layer_norm(output + x)
        attention_info = {
            'head_weights': F.softmax(self.head_weights, dim=0).detach(),
            'head_attentions': [a.detach() for a in head_attentions],
            'prototype_importance': [h.prototype_importance.detach() for h in self.heads]
        }
        return output, attention_info


class EDANv3(nn.Module):
    """
    E-DAN v3: Edge-Deployable Attention Network with FAIIA
    """
    def __init__(self, input_dim, num_heads=4, attention_dim=32, n_prototypes=8,
                 hidden_units=[256, 128, 64], dropout_rate=0.3, attention_dropout=0.1,
                 focal_alpha=0.60, focal_gamma=2.0, num_classes=1, output_logits=False):
        super(EDANv3, self).__init__()

        self.input_dim = input_dim
        self.output_logits = output_logits

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Initial probability estimator (for focal modulation)
        self.prob_estimator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # FAIIA: Focal-Aware Imbalance-Integrated Attention (NOVEL)
        self.faiia = MultiHeadFAIIA(
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            n_prototypes=n_prototypes,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            dropout=attention_dropout
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
        # We separate the final activation to support output_logits=True
        self.classifier_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, num_classes)
        )
        # Note: Sigmoid is applied in forward if output_logits is False

        self.last_attention_info = None

    def forward(self, x, return_attention=False):
        # Input normalization
        x = self.input_norm(x)

        # Initial probability estimate for focal modulation
        p_init = self.prob_estimator(x)

        # FAIIA with initial probability
        attended, attention_info = self.faiia(x, minority_prob=p_init)

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

        self.last_attention_info = {
            'initial_prob': p_init.detach(),
            'faiia_attention': attention_info,
            'se_weights': se_weights.detach()
        }

        if return_attention:
            return output, self.last_attention_info
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
