
import torch
import numpy as np
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

def evaluate_model(model, X_tensor, y_true, device, threshold=0.5):
    """
    Evaluates the model on the given tensor data.
    Returns a dictionary of metrics.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        # Check if the model returns a tuple (output, attn_info) or just output
        # We assume for evaluation we just want output
        # But some forward methods might return tuple if config is set, 
        # so we inspect the output or assume forward(..., return_attention=False) returns just output
        
        # However, for EDANv3, standard forward returns output only unless return_attention=True.
        outputs = model(X_tensor)
        
        # If output is logits (not bounded 0-1), we apply sigmoid for probabilities
        # Check if model has a 'output_logits' attribute which is true
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
    return metrics

def print_metrics(metrics, title="Evaluation Results"):
    print(f"\n{title}:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"  {metric:<15}: {value:.4f}")
