"""
Configuration for EDA-Net (Entropy-Dynamic Attention Network) and Ablation Studies.
"""

# ---------------------------------------------------------------------------
#  Main EDA-Net Configuration
# ---------------------------------------------------------------------------

EDA_CONFIG = {
    # EDT Attention Parameters
    'num_heads': 4,
    'attention_dim': 32,
    'n_prototypes': 8,
    'tau_min': 0.5,             # Minimum temperature (sharper attention)
    'tau_max': 2.0,             # Maximum temperature (smoother attention)
    'tau_hidden_dim': 32,       # Hidden dim of entropy→temperature MLP
    'edt_mode': 'learned',      # 'learned', 'heuristic', or 'fixed'
    'normalize_entropy': True,  # Normalise entropy to [0, 1]

    # Network Architecture
    'hidden_units': [256, 128, 64],
    'dropout_rate': 0.3,
    'attention_dropout': 0.1,

    # Training
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 256,
    'epochs': 150,
    'patience': 20,
    'label_smoothing': 0.05,

    # Loss
    'focal_gamma': 2.0,
    'entropy_reg_weight': 0.01,  # Weight for entropy regularisation term
    'prototype_anchor_weight': 0.01,  # Weight for prototype anchoring loss
}

# Backward compatibility alias
V3_CONFIG = EDA_CONFIG

# ---------------------------------------------------------------------------
#  Ablation Configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    # A1: Fixed temperature — standard attention baseline (no EDT)
    'fixed_temp': {**EDA_CONFIG, 'edt_mode': 'fixed'},

    # A2: Heuristic EDT — analytic τ = τ_max·(1−H̃) + τ_min (no learning)
    'heuristic': {**EDA_CONFIG, 'edt_mode': 'heuristic'},

    # A3: Learned EDT without entropy normalisation
    'no_entropy_norm': {**EDA_CONFIG, 'normalize_entropy': False},

    # A4: Narrow τ range — tighter adaptation
    'narrow_tau': {**EDA_CONFIG, 'tau_min': 0.8, 'tau_max': 1.5},

    # A5: Wide τ range — old default range
    'wide_tau': {**EDA_CONFIG, 'tau_min': 0.1, 'tau_max': 5.0},
}

# ---------------------------------------------------------------------------
#  Dataset Configuration
# ---------------------------------------------------------------------------

# Features dropped due to high correlation (> 0.95)
DROPPED_FEATURES = [
    'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_src_dport_ltm', 'ct_srv_dst',
    'dbytes', 'dloss', 'dwin', 'sbytes', 'sloss'
]

RANDOM_STATE = 42
