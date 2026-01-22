
import torch
import torch.nn as nn
from core.model import EDANv3

class VanillaDNN_Ablation(nn.Module):
    """Standard DNN without attention mechanism, outputs raw logits."""
    def __init__(self, input_dim, hidden_units=[256, 128, 64], dropout_rate=0.3):
        super(VanillaDNN_Ablation, self).__init__()

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
            nn.Dropout(dropout_rate/2),
            nn.Linear(32, 1), # No Sigmoid for raw logits
            nn.Identity() # Output raw logits
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def count_parameters(self):
        """Returns the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EDANv3_Ablation(EDANv3):
    """
    EDAN v3 with FAIIA, modified to output raw logits for ablation studies.
    Inherits from EDANv3 but sets output_logits=True by default.
    """
    def __init__(self, input_dim, num_heads=4, attention_dim=32, n_prototypes=8,
                 hidden_units=[256, 128, 64], dropout_rate=0.3, attention_dropout=0.1,
                 focal_alpha=0.60, focal_gamma=2.0, num_classes=1):
        super(EDANv3_Ablation, self).__init__(
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            n_prototypes=n_prototypes,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            num_classes=num_classes,
            output_logits=True 
        )
