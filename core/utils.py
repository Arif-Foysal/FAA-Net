
import torch
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_optimal_threshold(y_true, y_probs, metric='f1', thresholds=None):
    """
    Search for the classification threshold that maximizes the given metric.

    Evidential expected_prob is calibrated differently from sigmoid output,
    so the default 0.5 threshold is usually suboptimal. This finds the best one.

    Args:
        y_true:     Ground truth binary labels (numpy array)
        y_probs:    Predicted probabilities (numpy array)
        metric:     'f1' or 'recall' — the metric to optimize
        thresholds: Array of thresholds to try (default: 0.1 to 0.9 step 0.01)

    Returns:
        best_threshold: Threshold that maximizes the metric
        best_score:     Score at that threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    y_true = np.array(y_true).flatten()
    y_probs = np.array(y_probs).flatten()

    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_probs > t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score


def evaluate_model(model, X_tensor, y_true, device, threshold=0.5,
                   optimize_threshold=False):
    """
    Evaluates the model on the given tensor data.
    Supports both standard and evidential output modes.
    Returns a dictionary of metrics, probabilities, and predictions.
    For evidential models, also returns uncertainty info.

    Args:
        model:              The model to evaluate
        X_tensor:           Input features tensor
        y_true:             Ground truth labels
        device:             Compute device
        threshold:          Classification threshold (default 0.5)
        optimize_threshold: If True, search for the threshold that maximizes F1.
                            Useful for evidential models where 0.5 is miscalibrated.
    """
    model.eval()
    is_evidential = getattr(model, 'evidential', False)

    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        outputs = model(X_tensor)

        if is_evidential:
            y_probs = outputs['attack_prob'].cpu().numpy().flatten()
            epistemic_u = outputs['epistemic_uncertainty'].cpu().numpy().flatten()
            aleatoric_u = outputs['aleatoric_uncertainty'].cpu().numpy()
            evidence = outputs['evidence'].cpu().numpy()
        elif getattr(model, 'output_logits', False):
            y_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        else:
            y_probs = outputs.cpu().numpy().flatten()

    # Handle y_true format
    y_true_np = y_true
    if hasattr(y_true, 'values'):
        y_true_np = y_true.values
    y_true_np = np.array(y_true_np).flatten()

    # Optimize threshold if requested (especially for evidential models)
    if optimize_threshold:
        threshold, best_f1 = find_optimal_threshold(y_true_np, y_probs, metric='f1')

    y_pred = (y_probs > threshold).astype(int)

    metrics = {
        'Accuracy': accuracy_score(y_true_np, y_pred),
        'Precision': precision_score(y_true_np, y_pred, zero_division=0),
        'Recall': recall_score(y_true_np, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true_np, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true_np, y_probs),
        'Avg Precision': average_precision_score(y_true_np, y_probs),
        'Threshold': threshold,
    }

    if is_evidential:
        metrics['Mean Epistemic U'] = float(epistemic_u.mean())
        metrics['Mean Evidence (Normal)'] = float(evidence[:, 0].mean())
        metrics['Mean Evidence (Attack)'] = float(evidence[:, 1].mean())

    return metrics, y_probs, y_pred


def evaluate_evidential_uncertainty(model, X_tensor, y_true, device,
                                      y_categories=None, threshold=0.5):
    """
    Detailed evidential uncertainty analysis.

    Returns a DataFrame with per-sample uncertainty breakdown,
    plus aggregate statistics by category if y_categories is provided.

    Args:
        model:        Trained EDANv3 with evidential=True
        X_tensor:     Input features tensor
        y_true:       Ground truth binary labels
        device:       Compute device
        y_categories: Optional array of attack category labels per sample
        threshold:    Classification threshold for attack_prob

    Returns:
        results_df:   DataFrame with per-sample decisions and uncertainty
        summary:      Dict of aggregate statistics
    """
    model.eval()
    assert getattr(model, 'evidential', False), \
        "evaluate_evidential_uncertainty requires a model with evidential=True"

    with torch.no_grad():
        X_tensor = X_tensor.to(device)

        # Process in batches for memory efficiency
        batch_size = 2048
        all_results = {
            'attack_prob': [], 'epistemic_u': [],
            'aleatoric_u_normal': [], 'aleatoric_u_attack': [],
            'evidence_normal': [], 'evidence_attack': [],
            'dirichlet_strength': [],
        }

        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size]
            output = model(batch)

            all_results['attack_prob'].append(output['attack_prob'].cpu().numpy())
            all_results['epistemic_u'].append(
                output['epistemic_uncertainty'].cpu().numpy().flatten()
            )
            all_results['aleatoric_u_normal'].append(
                output['aleatoric_uncertainty'][:, 0].cpu().numpy()
            )
            all_results['aleatoric_u_attack'].append(
                output['aleatoric_uncertainty'][:, 1].cpu().numpy()
            )
            all_results['evidence_normal'].append(
                output['evidence'][:, 0].cpu().numpy()
            )
            all_results['evidence_attack'].append(
                output['evidence'][:, 1].cpu().numpy()
            )
            all_results['dirichlet_strength'].append(
                output['dirichlet_strength'].cpu().numpy().flatten()
            )

    results = {k: np.concatenate(v) for k, v in all_results.items()}
    df = pd.DataFrame(results)

    # Ground truth
    y_true_np = y_true
    if hasattr(y_true, 'values'):
        y_true_np = y_true.values
    df['y_true'] = np.array(y_true_np).flatten()

    # Predictions
    df['y_pred'] = (df['attack_prob'] > threshold).astype(int)
    df['correct'] = (df['y_pred'] == df['y_true']).astype(int)

    if y_categories is not None:
        if hasattr(y_categories, 'values'):
            y_categories = y_categories.values
        df['category'] = np.array(y_categories).flatten()

    # Summary statistics
    summary = {
        'overall_epistemic_u_mean': df['epistemic_u'].mean(),
        'overall_epistemic_u_std': df['epistemic_u'].std(),
        'correct_epistemic_u_mean': df[df['correct'] == 1]['epistemic_u'].mean(),
        'incorrect_epistemic_u_mean': df[df['correct'] == 0]['epistemic_u'].mean(),
        'normal_epistemic_u_mean': df[df['y_true'] == 0]['epistemic_u'].mean(),
        'attack_epistemic_u_mean': df[df['y_true'] == 1]['epistemic_u'].mean(),
    }

    if 'category' in df.columns:
        summary['epistemic_u_by_category'] = (
            df.groupby('category')['epistemic_u'].mean().to_dict()
        )

    return df, summary


def calibrate_epistemic_threshold(model, X_known_tensor, device, percentile=95.0):
    """
    Calibrate epistemic uncertainty threshold from known (in-distribution) data.

    Args:
        model:          Trained EDANv3 with evidential=True
        X_known_tensor: Feature tensor of known/in-distribution traffic
        device:         Compute device
        percentile:     Percentile of validation epistemic uncertainty
                        to use as the "known traffic" ceiling

    Returns:
        threshold: Calibrated epistemic uncertainty threshold
    """
    model.eval()
    assert getattr(model, 'evidential', False), \
        "calibrate_epistemic_threshold requires a model with evidential=True"

    with torch.no_grad():
        X_known_tensor = X_known_tensor.to(device)
        all_epistemic = []
        batch_size = 2048

        for i in range(0, len(X_known_tensor), batch_size):
            batch = X_known_tensor[i:i + batch_size]
            output = model(batch)
            all_epistemic.append(
                output['epistemic_uncertainty'].cpu().numpy().flatten()
            )

        epistemic_scores = np.concatenate(all_epistemic)

    threshold = np.percentile(epistemic_scores, percentile)

    print(f"  Calibrated epistemic threshold: {threshold:.4f}")
    print(f"    Based on {percentile}th percentile of {len(X_known_tensor)} known samples")
    print(f"    Mean epistemic U: {epistemic_scores.mean():.4f}")
    print(f"    Max  epistemic U: {epistemic_scores.max():.4f}")

    return threshold


def save_training_history(history, filename):
    """Saves training history dict to a CSV or JSON."""
    df = pd.DataFrame(history)
    df.to_csv(filename, index=False)

def save_predictions(y_true, y_probs, filename, epistemic_u=None, evidence=None):
    """Saves ground truth, probabilities, and optionally uncertainty data."""
    data = {
        'y_true': y_true,
        'y_probs': y_probs
    }
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        data['y_true'] = y_true.values

    if epistemic_u is not None:
        data['epistemic_u'] = epistemic_u
    if evidence is not None:
        data['evidence_normal'] = evidence[:, 0]
        data['evidence_attack'] = evidence[:, 1]

    np.savez(filename, **data)

def print_metrics(metrics, title="Evaluation Results"):
    print(f"\n{title}:")
    print("-" * 40)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric:<25}: {value:.4f}")
        else:
            print(f"  {metric:<25}: {value}")
