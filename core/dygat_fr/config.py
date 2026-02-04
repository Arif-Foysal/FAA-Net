"""
DyGAT-FR Configuration

Default hyperparameters and configuration for DyGAT-FR experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DyGATFRConfig:
    """Configuration for DyGAT-FR model and training."""
    
    # Model Architecture
    hidden_channels: int = 128
    out_channels: int = 64
    num_layers: int = 3
    heads: int = 4
    n_prototypes: int = 8
    dropout: float = 0.3
    attention_dropout: float = 0.1
    
    # Focal Parameters (from FAA-Net)
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Prototype Parameters
    prototype_momentum: float = 0.9
    prototype_init_method: str = 'kmeans'  # 'kmeans', 'random', 'medoid'
    
    # Feedback Module
    use_feedback: bool = True
    feedback_strength: float = 0.1
    feedback_start_epoch: int = 10
    
    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 256  # For mini-batch training
    curriculum_epochs: int = 5
    early_stopping_patience: int = 10
    
    # Loss Weights
    contrastive_weight: float = 0.1
    replay_weight: float = 0.1
    
    # Incremental Learning
    n_increments: int = 5
    minority_drift: bool = True
    drift_intensity: float = 0.3
    replay_buffer_size: int = 1000
    replay_ratio: float = 0.1
    
    # Graph Construction
    k_neighbors: int = 10
    graph_metric: str = 'euclidean'
    include_self_loops: bool = True
    
    # Device
    device: str = 'auto'
    
    # Random Seed
    seed: int = 42
    
    def get_device(self) -> torch.device:
        """Get the computing device."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    
    # Experiment
    name: str = 'dygat_fr_experiment'
    dataset: str = 'unsw-nb15'
    
    # Paths
    data_dir: str = '.'
    output_dir: str = 'results/dygat_fr'
    checkpoint_dir: str = 'checkpoints/dygat_fr'
    
    # Logging
    log_interval: int = 10
    save_checkpoints: bool = True
    save_best_only: bool = True
    
    # Visualization
    plot_training: bool = True
    plot_attention: bool = True
    
    # Evaluation
    eval_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1_score',
        'auc_roc', 'avg_precision', 'fpr'
    ])
    per_class_eval: bool = True


# Default configurations for different scenarios
CONFIGS = {
    'default': DyGATFRConfig(),
    
    'lightweight': DyGATFRConfig(
        hidden_channels=64,
        out_channels=32,
        num_layers=2,
        heads=2,
        n_prototypes=4,
        use_feedback=False
    ),
    
    'high_capacity': DyGATFRConfig(
        hidden_channels=256,
        out_channels=128,
        num_layers=4,
        heads=8,
        n_prototypes=16,
        dropout=0.4
    ),
    
    'fast_adaptation': DyGATFRConfig(
        prototype_momentum=0.7,
        feedback_start_epoch=5,
        curriculum_epochs=3,
        lr=2e-3
    ),
    
    'stable_incremental': DyGATFRConfig(
        prototype_momentum=0.95,
        replay_ratio=0.2,
        replay_buffer_size=2000,
        early_stopping_patience=15
    )
}


def get_config(name: str = 'default') -> DyGATFRConfig:
    """Get a predefined configuration."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]
