"""
Utility functions for EDA-Net: evaluation, seeding, analysis, and I/O.
"""

import torch
import numpy as np
import pandas as pd
import random
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_model(model, X_tensor, y_true, device, threshold=0.5):
    """
    Evaluates the model on the given tensor data.
    Returns a dictionary of metrics, probabilities, and predictions.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        outputs = model(X_tensor)

        if getattr(model, 'output_logits', False):
            y_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        else:
            y_probs = outputs.cpu().numpy().flatten()

        y_pred = (y_probs > threshold).astype(int)

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, y_probs),
        'Avg Precision': average_precision_score(y_true, y_probs)
    }
    return metrics, y_probs, y_pred


def collect_edt_analysis(model, X_tensor, y_true, device, batch_size=1024):
    """
    Run inference and collect per-sample EDT metrics (entropy, tau)
    for post-hoc analysis and paper figures.

    Args:
        model: EDANet instance
        X_tensor: (N, D) input tensor
        y_true: (N,) ground truth labels
        device: torch.device
        batch_size: inference batch size

    Returns:
        analysis_df: DataFrame with columns
            ['y_true', 'y_prob', 'entropy', 'tau', 'y_pred']
    """
    model.eval()
    all_probs, all_entropies, all_taus = [], [], []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(device)
            outputs, edt_info = model(batch, return_edt_info=True)

            if getattr(model, 'output_logits', False):
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            else:
                probs = outputs.cpu().numpy().flatten()

            all_probs.extend(probs)

            edt_attn = edt_info.get('edt_attention', {})
            if 'mean_entropy' in edt_attn:
                all_entropies.extend(edt_attn['mean_entropy'].cpu().numpy().flatten())
            if 'mean_tau' in edt_attn:
                all_taus.extend(edt_attn['mean_tau'].cpu().numpy().flatten())

    y_true_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)

    analysis_df = pd.DataFrame({
        'y_true': y_true_arr,
        'y_prob': all_probs,
        'y_pred': (np.array(all_probs) > 0.5).astype(int),
    })

    if all_entropies:
        analysis_df['entropy'] = all_entropies
    if all_taus:
        analysis_df['tau'] = all_taus

    return analysis_df


def save_training_history(history, filename):
    """Saves training history dict to CSV."""
    df = pd.DataFrame(history)
    df.to_csv(filename, index=False)


def save_predictions(y_true, y_probs, filename):
    """Saves ground truth and probabilities for ROC/Recall analysis."""
    data = {
        'y_true': y_true,
        'y_probs': y_probs
    }
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        data['y_true'] = y_true.values

    np.savez(filename, **data)


def save_edt_analysis(analysis_df, filename):
    """Save EDT analysis dataframe."""
    analysis_df.to_csv(filename, index=False)


def print_metrics(metrics, title="Evaluation Results"):
    print(f"\n{title}:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"  {metric:<15}: {value:.4f}")
