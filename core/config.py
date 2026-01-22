
# Configuration for EDANv3 and Ablation Studies

V3_CONFIG = {
    'num_heads': 4,
    'attention_dim': 32,
    'n_prototypes': 8,
    'hidden_units': [256, 128, 64],
    'dropout_rate': 0.3,
    'attention_dropout': 0.1,
    'focal_alpha': 0.60,
    'focal_gamma': 2.0,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 256,
    'epochs': 150,
    'patience': 20,
    'label_smoothing': 0.05,
}

# Features dropped due to high correlation (> 0.95)
DROPPED_FEATURES = [
    'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_src_dport_ltm', 'ct_srv_dst',
    'dbytes', 'dloss', 'dwin', 'sbytes', 'sloss'
]

RANDOM_STATE = 42
