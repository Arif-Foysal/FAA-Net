
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from core.config import V3_CONFIG, RANDOM_STATE
from core.data_loader import load_and_preprocess_data, create_dataloaders
from core.ablation import VanillaDNN_Ablation, EDANv3_Ablation
from core.model import MinorityPrototypeGenerator, EDANv3
from core.loss import ImbalanceAwareFocalLoss_Logits
from core.evidential import FocalEvidentialLoss, EvidentialLoss
from core.trainer import train_model
from core.utils import (set_all_seeds, evaluate_model, print_metrics,
                        save_predictions, save_training_history)

def run_experiment(name, model, train_loader, val_loader, test_tensor, y_test, config, criterion, device):
    print(f"\n--- Running Experiment: {name} ---")
    is_evidential = getattr(model, 'evidential', False)
    model, history = train_model(model, train_loader, val_loader, config, criterion, device)
    metrics, y_probs, y_pred = evaluate_model(
        model, test_tensor, y_test, device,
        optimize_threshold=is_evidential,
    )
    print_metrics(metrics, f"{name} Results")
    
    save_dir = "."
    if os.path.exists("/content/drive/MyDrive"):
        save_dir = "/content/drive/MyDrive/FAIIA_Models"
        os.makedirs(save_dir, exist_ok=True)
    
    # Save Model
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save Predictions (for Ablation Table T3 and Analysis)
    pred_path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}_predictions.npz")
    save_predictions(y_test, y_probs, pred_path)
    print(f"Predictions saved to {pred_path}")

    return metrics, history

def main():
    print("="*60)
    print("FAIIA-IDS: Ablation Study")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(RANDOM_STATE)

    # Data
    data_dir = "/content" if os.path.exists("/content") else "."
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = load_and_preprocess_data(data_dir=data_dir)
    
    train_loader, val_loader, _, X_test_tensor = create_dataloaders(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        batch_size=V3_CONFIG['batch_size']
    )

    input_dim = X_train_scaled.shape[1]
    
    # Class counts for loss weighting
    count_positive = y_train.sum()
    count_negative = len(y_train) - count_positive
    class_counts = [count_negative, count_positive]
    
    # Pos weight for BCEWithLogitsLoss
    pos_weight = torch.tensor([count_negative / count_positive], device=device, dtype=torch.float32)

    # Class weights for evidential loss
    # Use sqrt-dampened weights: full inverse-frequency is too aggressive
    # and starves evidence accumulation (evidence 12/6 vs 92/73 without focal)
    total_samples = count_positive + count_negative
    w_neg = np.sqrt(total_samples / (2 * count_negative))
    w_pos = np.sqrt(total_samples / (2 * count_positive))
    evidential_class_weights = torch.tensor(
        [w_neg, w_pos], dtype=torch.float32
    ).to(device)

    results = {}

    # --- Experiment 1: Vanilla DNN + BCE ---
    model_1 = VanillaDNN_Ablation(input_dim=input_dim).to(device)
    criterion_1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    results['Vanilla DNN + BCE'], _ = run_experiment(
        "VanillaDNN_BCE", model_1, train_loader, val_loader, X_test_tensor, y_test, V3_CONFIG, criterion_1, device
    )

    # --- Experiment 2: Vanilla DNN + Focal Loss ---
    model_2 = VanillaDNN_Ablation(input_dim=input_dim).to(device)
    criterion_2 = ImbalanceAwareFocalLoss_Logits(class_counts=class_counts, gamma=2.0)
    results['Vanilla DNN + Focal'], history_vanilla = run_experiment(
        "VanillaDNN_Focal", model_2, train_loader, val_loader, X_test_tensor, y_test, V3_CONFIG, criterion_2, device
    )
    # Save Vanilla History
    save_training_history_path = "vanilladnn_history.csv"
    if os.path.exists("/content/drive/MyDrive/FAIIA_Models"):
        save_training_history_path = os.path.join("/content/drive/MyDrive/FAIIA_Models", "vanilladnn_history.csv")
    pd.DataFrame(history_vanilla).to_csv(save_training_history_path, index=False)
    print(f"Saved Vanilla DNN history to {save_training_history_path}")

    # --- Setup for EDAN Ablation (Prototypes) ---
    minority_mask = y_train.values == 1
    X_minority = X_train_scaled[minority_mask]
    proto_gen = MinorityPrototypeGenerator(n_prototypes=V3_CONFIG['n_prototypes'], random_state=RANDOM_STATE)
    prototypes = proto_gen.fit(X_minority)

    # --- Experiment 3: FAIIA (EDAN v3) + BCE ---
    model_3 = EDANv3_Ablation(
        input_dim=input_dim, num_heads=V3_CONFIG['num_heads'], attention_dim=V3_CONFIG['attention_dim']
    ).to(device)
    model_3.faiia.initialize_all_prototypes(prototypes, device)
    criterion_3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    results['FAIIA + BCE'], _ = run_experiment(
        "FAIIA_BCE", model_3, train_loader, val_loader, X_test_tensor, y_test, V3_CONFIG, criterion_3, device
    )

    # --- Experiment 4: FAIIA (EDAN v3) + Focal Loss ---
    model_4 = EDANv3_Ablation(
        input_dim=input_dim, num_heads=V3_CONFIG['num_heads'], attention_dim=V3_CONFIG['attention_dim']
    ).to(device)
    model_4.faiia.initialize_all_prototypes(prototypes, device)
    criterion_4 = ImbalanceAwareFocalLoss_Logits(class_counts=class_counts, gamma=2.0)
    results['FAIIA + Focal'], _ = run_experiment(
        "FAIIA_Focal", model_4, train_loader, val_loader, X_test_tensor, y_test, V3_CONFIG, criterion_4, device
    )

    # --- Experiment 5: FAIIA + Evidential Loss (no focal) ---
    model_5 = EDANv3(
        input_dim=input_dim, num_heads=V3_CONFIG['num_heads'],
        attention_dim=V3_CONFIG['attention_dim'],
        n_prototypes=V3_CONFIG['n_prototypes'],
        hidden_units=V3_CONFIG['hidden_units'],
        dropout_rate=V3_CONFIG['dropout_rate'],
        attention_dropout=V3_CONFIG['attention_dropout'],
        focal_alpha=V3_CONFIG['focal_alpha'],
        focal_gamma=V3_CONFIG['focal_gamma'],
        evidential=True,
    ).to(device)
    model_5.faiia.initialize_all_prototypes(prototypes, device)
    criterion_5 = EvidentialLoss(
        num_classes=2,
        annealing_epochs=V3_CONFIG.get('annealing_epochs', 10),
    ).to(device)
    results['FAIIA + Evidential'], _ = run_experiment(
        "FAIIA_Evidential", model_5, train_loader, val_loader, X_test_tensor, y_test, V3_CONFIG, criterion_5, device
    )

    # --- Experiment 6: FAIIA + Focal Evidential Loss (full model) ---
    model_6 = EDANv3(
        input_dim=input_dim, num_heads=V3_CONFIG['num_heads'],
        attention_dim=V3_CONFIG['attention_dim'],
        n_prototypes=V3_CONFIG['n_prototypes'],
        hidden_units=V3_CONFIG['hidden_units'],
        dropout_rate=V3_CONFIG['dropout_rate'],
        attention_dropout=V3_CONFIG['attention_dropout'],
        focal_alpha=V3_CONFIG['focal_alpha'],
        focal_gamma=V3_CONFIG['focal_gamma'],
        evidential=True,
    ).to(device)
    model_6.faiia.initialize_all_prototypes(prototypes, device)
    criterion_6 = FocalEvidentialLoss(
        num_classes=2,
        annealing_epochs=V3_CONFIG.get('annealing_epochs', 10),
        gamma=V3_CONFIG.get('evidential_focal_gamma', 2.0),
        class_weights=evidential_class_weights,
    ).to(device)
    results['FAIIA + Focal Evidential'], _ = run_experiment(
        "FAIIA_FocalEvidential", model_6, train_loader, val_loader, X_test_tensor, y_test, V3_CONFIG, criterion_6, device
    )

    # Summary
    print("\n=== Ablation Study Summary ===")
    df = pd.DataFrame(results).T
    print(df)
    df.to_csv('ablation_summary.csv')

if __name__ == "__main__":
    main()
