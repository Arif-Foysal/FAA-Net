"""
DyGAT-FR Utility Functions

Utility functions for evaluation, visualization, and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import seaborn as sns


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        balanced_accuracy_score, matthews_corrcoef
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['avg_precision'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.0
            metrics['avg_precision'] = 0.0
    
    # Confusion matrix derived metrics
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics


def compute_per_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_types: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, float]:
    """
    Compute recall for each attack type.
    
    Args:
        y_true: Ground truth binary labels (0/1)
        y_pred: Predicted binary labels
        attack_types: Original attack type labels (for multi-class breakdown)
        class_names: Mapping from type ID to name
    
    Returns:
        Per-class recall dictionary
    """
    if attack_types is None:
        # Binary case
        return {
            'normal': recall_score(1 - y_true, 1 - y_pred, zero_division=0),
            'attack': recall_score(y_true, y_pred, zero_division=0)
        }
    
    results = {}
    unique_types = np.unique(attack_types)
    
    for attack_type in unique_types:
        mask = attack_types == attack_type
        type_true = y_true[mask]
        type_pred = y_pred[mask]
        
        if len(type_true) > 0:
            # For attacks (type > 0), recall = correct attack predictions
            if attack_type > 0:
                recall = (type_pred == 1).mean()
            else:
                recall = (type_pred == 0).mean()
            
            name = class_names.get(attack_type, f'Type_{attack_type}') if class_names else f'Type_{attack_type}'
            results[name] = {
                'recall': recall,
                'count': len(type_true)
            }
    
    return results


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot training history curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 Score
    ax = axes[1]
    if 'train_f1' in history:
        ax.plot(history['train_f1'], label='Train F1', color='blue')
    if 'val_f1' in history:
        ax.plot(history['val_f1'], label='Val F1', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Recall
    ax = axes[2]
    if 'train_recall' in history:
        ax.plot(history['train_recall'], label='Train Recall', color='blue')
    if 'val_recall' in history:
        ax.plot(history['val_recall'], label='Val Recall', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Recall vs Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Normal', 'Attack'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Class name labels
        save_path: Path to save figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = 'DyGAT-FR',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot ROC and Precision-Recall curves.
    
    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        model_name: Model name for legend
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ROC Curve
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PR Curve
    ax = axes[1]
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(recall, precision, label=f'{model_name} (AP = {ap:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prototype_attention(
    attention_weights: torch.Tensor,
    labels: torch.Tensor,
    n_samples: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Visualize prototype attention patterns.
    
    Args:
        attention_weights: Attention to prototypes (N, K)
        labels: Sample labels (N,)
        n_samples: Number of samples to visualize
        save_path: Path to save figure
        figsize: Figure size
    """
    attn = attention_weights[:n_samples].cpu().numpy()
    labs = labels[:n_samples].cpu().numpy()
    
    # Sort by label
    sort_idx = np.argsort(labs)
    attn = attn[sort_idx]
    labs = labs[sort_idx]
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    im = plt.imshow(attn, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, label='Attention Weight')
    
    # Mark class boundaries
    boundary = np.where(np.diff(labs))[0]
    for b in boundary:
        plt.axhline(y=b + 0.5, color='white', linewidth=2)
    
    plt.xlabel('Prototype Index')
    plt.ylabel('Sample Index')
    plt.title('Prototype Attention Patterns by Class')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_incremental_performance(
    increment_results: List[Dict[str, Any]],
    metric: str = 'f1',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot performance across increments.
    
    Args:
        increment_results: List of per-increment results
        metric: Metric to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    increments = [r['increment'] for r in increment_results]
    values = [r['final_metrics'][metric] for r in increment_results]
    
    plt.figure(figsize=figsize)
    plt.plot(increments, values, 'o-', markersize=10, linewidth=2)
    
    # Add value labels
    for i, v in zip(increments, values):
        plt.annotate(f'{v:.3f}', (i, v), textcoords="offset points",
                    xytext=(0, 10), ha='center')
    
    plt.xlabel('Increment')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} Across Increments')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def format_results_table(
    results: Dict[str, Any],
    precision: int = 4
) -> str:
    """
    Format results as a markdown table.
    
    Args:
        results: Dictionary of metric name -> value
        precision: Decimal precision
    
    Returns:
        Markdown table string
    """
    lines = ['| Metric | Value |', '|--------|-------|']
    
    for key, value in results.items():
        if isinstance(value, float):
            lines.append(f'| {key} | {value:.{precision}f} |')
        else:
            lines.append(f'| {key} | {value} |')
    
    return '\n'.join(lines)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        path: Save path
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, Dict]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into (optional)
        path: Checkpoint path
        device: Target device
    
    Returns:
        Tuple of (model, optimizer, epoch, metrics)
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return (
        model,
        optimizer,
        checkpoint.get('epoch', 0),
        checkpoint.get('metrics', {})
    )


class EarlyStopping:
    """
    Early stopping utility.
    
    Args:
        patience: Number of epochs to wait
        min_delta: Minimum improvement required
        mode: 'min' or 'max' for metric
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
