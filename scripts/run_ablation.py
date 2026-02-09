"""
EDA-Net: Comprehensive Ablation Study.

Experiments:
    1. Vanilla DNN + BCE          (no attention baseline)
    2. Vanilla DNN + Focal        (focal loss baseline)
    3. Fixed-Temp Attention + Focal   (attention w/o EDT)
    4. Heuristic EDT + Focal      (analytic τ, no learned MLP)
    5. EDA-Net (Learned EDT) + Focal  (full model)
    6. EDA-Net + Focal (narrow τ) (sensitivity: τ ∈ [0.5, 2.0])
    7. EDA-Net + Focal (wide τ)   (sensitivity: τ ∈ [0.01, 10.0])
    8. EDA-Net + Focal (no norm)  (without entropy normalisation)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pandas as pd
from core.config import EDA_CONFIG, ABLATION_CONFIGS, RANDOM_STATE
from core.data_loader import load_and_preprocess_data, create_dataloaders
from core.ablation import (VanillaDNN_Ablation, FixedTempNet_Ablation,
                           HeuristicEDTNet_Ablation, EDANet_Ablation)
from core.model import MinorityPrototypeGenerator
from core.loss import ImbalanceAwareFocalLoss_Logits, EDANetLoss
from core.trainer import train_model
from core.utils import (set_all_seeds, evaluate_model, print_metrics,
                        save_predictions, save_training_history,
                        collect_edt_analysis, save_edt_analysis)


def run_experiment(name, model, train_loader, val_loader, test_tensor, y_test,
                   config, criterion, device, use_edt_loss=False):
    """Run a single ablation experiment and return results."""
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"{'='*60}")

    model, history = train_model(
        model, train_loader, val_loader, config, criterion, device,
        use_edt_loss=use_edt_loss
    )
    metrics, y_probs, y_pred = evaluate_model(model, test_tensor, y_test, device)
    print_metrics(metrics, f"{name} Results")

    # Save directory
    save_dir = "."
    if os.path.exists("/content/drive/MyDrive"):
        save_dir = "/content/drive/MyDrive/EDANet_Models"
        os.makedirs(save_dir, exist_ok=True)

    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('τ', 'tau').lower()

    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, f"{safe_name}.pt"))

    # Save predictions
    save_predictions(y_test, y_probs, os.path.join(save_dir, f"{safe_name}_predictions.npz"))

    # Save history
    save_training_history(history, os.path.join(save_dir, f"{safe_name}_history.csv"))

    # Save EDT analysis if applicable
    if hasattr(model, 'last_edt_info') and model.last_edt_info is not None:
        analysis_df = collect_edt_analysis(model, test_tensor, y_test, device)
        save_edt_analysis(analysis_df, os.path.join(save_dir, f"{safe_name}_edt_analysis.csv"))

    return metrics, history


def make_edanet(input_dim, config):
    """Create an EDANet ablation variant from config."""
    return EDANet_Ablation(
        input_dim=input_dim,
        num_heads=config['num_heads'],
        attention_dim=config['attention_dim'],
        n_prototypes=config['n_prototypes'],
        tau_min=config['tau_min'],
        tau_max=config['tau_max'],
        tau_hidden_dim=config['tau_hidden_dim'],
        edt_mode=config['edt_mode'],
        normalize_entropy=config.get('normalize_entropy', True),
        hidden_units=config['hidden_units'],
        dropout_rate=config['dropout_rate'],
        attention_dropout=config['attention_dropout'],
    )


def main():
    print("=" * 60)
    print("EDA-Net: Ablation Study")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(RANDOM_STATE)

    # Data
    data_dir = "/content" if os.path.exists("/content") else "."
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = load_and_preprocess_data(data_dir=data_dir)

    train_loader, val_loader, _, X_test_tensor = create_dataloaders(
        X_train_scaled, y_train, X_test_scaled, y_test,
        batch_size=EDA_CONFIG['batch_size']
    )

    input_dim = X_train_scaled.shape[1]

    # Class counts for loss
    count_positive = y_train.sum()
    count_negative = len(y_train) - count_positive
    class_counts = [count_negative, count_positive]
    pos_weight = torch.tensor([count_negative / count_positive], device=device, dtype=torch.float32)

    # Prototypes (for attention-based models)
    minority_mask = y_train.values == 1
    X_minority = X_train_scaled[minority_mask]
    proto_gen = MinorityPrototypeGenerator(n_prototypes=EDA_CONFIG['n_prototypes'], random_state=RANDOM_STATE)
    prototypes = proto_gen.fit(X_minority)

    results = {}

    # ---- Experiment 1: Vanilla DNN + BCE ----
    set_all_seeds(RANDOM_STATE)
    model_1 = VanillaDNN_Ablation(input_dim=input_dim).to(device)
    criterion_1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    results['Vanilla DNN + BCE'], _ = run_experiment(
        "VanillaDNN_BCE", model_1, train_loader, val_loader,
        X_test_tensor, y_test, EDA_CONFIG, criterion_1, device
    )

    # ---- Experiment 2: Vanilla DNN + Focal ----
    set_all_seeds(RANDOM_STATE)
    model_2 = VanillaDNN_Ablation(input_dim=input_dim).to(device)
    criterion_2 = ImbalanceAwareFocalLoss_Logits(class_counts=class_counts, gamma=2.0)
    results['Vanilla DNN + Focal'], hist_vanilla = run_experiment(
        "VanillaDNN_Focal", model_2, train_loader, val_loader,
        X_test_tensor, y_test, EDA_CONFIG, criterion_2, device
    )

    # ---- Experiment 3: Fixed-Temp Attention + Focal ----
    set_all_seeds(RANDOM_STATE)
    model_3 = FixedTempNet_Ablation(
        input_dim=input_dim,
        num_heads=EDA_CONFIG['num_heads'],
        attention_dim=EDA_CONFIG['attention_dim'],
        n_prototypes=EDA_CONFIG['n_prototypes'],
    ).to(device)
    model_3.edt_attention.initialize_all_prototypes(prototypes, device)
    criterion_3 = ImbalanceAwareFocalLoss_Logits(class_counts=class_counts, gamma=2.0)
    results['Fixed-Temp Attn + Focal'], _ = run_experiment(
        "FixedTemp_Focal", model_3, train_loader, val_loader,
        X_test_tensor, y_test, EDA_CONFIG, criterion_3, device
    )

    # ---- Experiment 4: Heuristic EDT + Focal ----
    set_all_seeds(RANDOM_STATE)
    model_4 = HeuristicEDTNet_Ablation(
        input_dim=input_dim,
        num_heads=EDA_CONFIG['num_heads'],
        attention_dim=EDA_CONFIG['attention_dim'],
        n_prototypes=EDA_CONFIG['n_prototypes'],
    ).to(device)
    model_4.edt_attention.initialize_all_prototypes(prototypes, device)
    criterion_4 = ImbalanceAwareFocalLoss_Logits(class_counts=class_counts, gamma=2.0)
    results['Heuristic EDT + Focal'], _ = run_experiment(
        "Heuristic_EDT_Focal", model_4, train_loader, val_loader,
        X_test_tensor, y_test, EDA_CONFIG, criterion_4, device
    )

    # ---- Experiment 5: Full EDA-Net (Learned EDT) + Focal ----
    set_all_seeds(RANDOM_STATE)
    model_5 = make_edanet(input_dim, EDA_CONFIG).to(device)
    model_5.edt_attention.initialize_all_prototypes(prototypes, device)
    criterion_5 = EDANetLoss(
        gamma=EDA_CONFIG['focal_gamma'],
        class_counts=class_counts,
        entropy_reg_weight=EDA_CONFIG['entropy_reg_weight']
    )
    results['EDA-Net (Full)'], hist_edanet = run_experiment(
        "EDANet_Full", model_5, train_loader, val_loader,
        X_test_tensor, y_test, EDA_CONFIG, criterion_5, device,
        use_edt_loss=True
    )

    # ---- Experiment 6: EDA-Net Narrow τ [0.5, 2.0] ----
    set_all_seeds(RANDOM_STATE)
    cfg_narrow = ABLATION_CONFIGS['narrow_tau']
    model_6 = make_edanet(input_dim, cfg_narrow).to(device)
    model_6.edt_attention.initialize_all_prototypes(prototypes, device)
    criterion_6 = EDANetLoss(gamma=cfg_narrow['focal_gamma'], class_counts=class_counts,
                              entropy_reg_weight=cfg_narrow['entropy_reg_weight'])
    results['EDA-Net (Narrow τ)'], _ = run_experiment(
        "EDANet_NarrowTau", model_6, train_loader, val_loader,
        X_test_tensor, y_test, cfg_narrow, criterion_6, device,
        use_edt_loss=True
    )

    # ---- Experiment 7: EDA-Net Wide τ [0.01, 10.0] ----
    set_all_seeds(RANDOM_STATE)
    cfg_wide = ABLATION_CONFIGS['wide_tau']
    model_7 = make_edanet(input_dim, cfg_wide).to(device)
    model_7.edt_attention.initialize_all_prototypes(prototypes, device)
    criterion_7 = EDANetLoss(gamma=cfg_wide['focal_gamma'], class_counts=class_counts,
                              entropy_reg_weight=cfg_wide['entropy_reg_weight'])
    results['EDA-Net (Wide τ)'], _ = run_experiment(
        "EDANet_WideTau", model_7, train_loader, val_loader,
        X_test_tensor, y_test, cfg_wide, criterion_7, device,
        use_edt_loss=True
    )

    # ---- Experiment 8: EDA-Net No Entropy Normalisation ----
    set_all_seeds(RANDOM_STATE)
    cfg_nonorm = ABLATION_CONFIGS['no_entropy_norm']
    model_8 = make_edanet(input_dim, cfg_nonorm).to(device)
    model_8.edt_attention.initialize_all_prototypes(prototypes, device)
    criterion_8 = EDANetLoss(gamma=cfg_nonorm['focal_gamma'], class_counts=class_counts,
                              entropy_reg_weight=cfg_nonorm['entropy_reg_weight'])
    results['EDA-Net (No Norm)'], _ = run_experiment(
        "EDANet_NoNorm", model_8, train_loader, val_loader,
        X_test_tensor, y_test, cfg_nonorm, criterion_8, device,
        use_edt_loss=True
    )

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    df = pd.DataFrame(results).T
    print(df.to_string())

    save_dir = "."
    if os.path.exists("/content/drive/MyDrive/EDANet_Models"):
        save_dir = "/content/drive/MyDrive/EDANet_Models"
    df.to_csv(os.path.join(save_dir, 'ablation_summary.csv'))
    print(f"\nAblation summary saved to {os.path.join(save_dir, 'ablation_summary.csv')}")


if __name__ == "__main__":
    main()
