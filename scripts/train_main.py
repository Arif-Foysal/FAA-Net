"""
EDA-Net: Main Model Training Script.

Trains the full EDA-Net (Entropy-Dynamic Attention Network) on UNSW-NB15
and saves all artifacts needed for paper figures and evaluation.
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from core.config import EDA_CONFIG, RANDOM_STATE
from core.data_loader import load_and_preprocess_data, create_dataloaders
from core.model import EDANet, MinorityPrototypeGenerator
from core.loss import EDANetLoss
from core.trainer import train_model
from core.utils import (set_all_seeds, evaluate_model, print_metrics,
                        save_training_history, save_predictions,
                        collect_edt_analysis, save_edt_analysis)


def main():
    print("=" * 60)
    print("EDA-Net: Entropy-Dynamic Attention Network — Training")
    print("=" * 60)

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_all_seeds(RANDOM_STATE)

    # 2. Data Loading
    data_dir = "/content" if os.path.exists("/content") else "."
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = load_and_preprocess_data(data_dir=data_dir)

    # 3. Create DataLoaders
    train_loader, val_loader, test_loader, X_test_tensor = create_dataloaders(
        X_train_scaled, y_train, X_test_scaled, y_test,
        batch_size=EDA_CONFIG['batch_size']
    )

    # 4. Extract Minority Prototypes
    print("\nExtracting Minority Prototypes...")
    minority_mask = y_train.values == 1
    X_minority = X_train_scaled[minority_mask]
    X_majority = X_train_scaled[~minority_mask]

    print(f"  Minority samples: {len(X_minority)}")
    print(f"  Majority samples: {len(X_majority)}")

    proto_gen = MinorityPrototypeGenerator(
        n_prototypes=EDA_CONFIG['n_prototypes'], random_state=RANDOM_STATE
    )
    minority_prototypes = proto_gen.fit(X_minority)

    # 5. Initialise Model
    input_dim = X_train_scaled.shape[1]
    model = EDANet(
        input_dim=input_dim,
        num_heads=EDA_CONFIG['num_heads'],
        attention_dim=EDA_CONFIG['attention_dim'],
        n_prototypes=EDA_CONFIG['n_prototypes'],
        hidden_units=EDA_CONFIG['hidden_units'],
        dropout_rate=EDA_CONFIG['dropout_rate'],
        attention_dropout=EDA_CONFIG['attention_dropout'],
        tau_min=EDA_CONFIG['tau_min'],
        tau_max=EDA_CONFIG['tau_max'],
        tau_hidden_dim=EDA_CONFIG['tau_hidden_dim'],
        edt_mode=EDA_CONFIG['edt_mode'],
        normalize_entropy=EDA_CONFIG['normalize_entropy'],
        num_classes=1,
        output_logits=True
    ).to(device)

    # Initialise prototypes from minority cluster centres
    model.edt_attention.initialize_all_prototypes(minority_prototypes, device)

    print(f"\nModel initialised with {model.count_parameters():,} parameters.")
    print(f"  EDT mode: {EDA_CONFIG['edt_mode']}")
    print(f"  τ range: [{EDA_CONFIG['tau_min']}, {EDA_CONFIG['tau_max']}]")

    # 6. Loss Function (Focal + Entropy Regularisation)
    class_counts = [len(X_majority), len(X_minority)]
    criterion = EDANetLoss(
        gamma=EDA_CONFIG['focal_gamma'],
        class_counts=class_counts,
        entropy_reg_weight=EDA_CONFIG['entropy_reg_weight']
    )

    # 7. Train
    model, history = train_model(
        model, train_loader, val_loader, EDA_CONFIG, criterion, device,
        use_edt_loss=True
    )

    # 8. Evaluate
    print("\nEvaluating on Test Set...")
    metrics, y_probs, y_pred = evaluate_model(model, X_test_tensor, y_test, device)
    print_metrics(metrics, "EDA-Net Test Results")

    # 9. Collect EDT analysis (entropy, tau per sample)
    print("\nCollecting EDT analysis data...")
    analysis_df = collect_edt_analysis(model, X_test_tensor, y_test, device)

    # 10. Save artifacts
    save_dir = "."
    if os.path.exists("/content/drive/MyDrive"):
        save_dir = "/content/drive/MyDrive/EDANet_Models"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving artifacts to Google Drive: {save_dir}")

    # Metrics CSV
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, 'edanet_metrics.csv'), index=False)

    # Model checkpoint
    save_path = os.path.join(save_dir, 'edanet_main.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Training history (includes tau/entropy curves)
    hist_path = os.path.join(save_dir, 'edanet_history.csv')
    save_training_history(history, hist_path)
    print(f"Training history saved to {hist_path}")

    # Predictions (for ROC/PR curves)
    pred_path = os.path.join(save_dir, 'edanet_predictions.npz')
    save_predictions(y_test, y_probs, pred_path)
    print(f"Predictions saved to {pred_path}")

    # EDT analysis (for temperature/entropy figures)
    analysis_path = os.path.join(save_dir, 'edanet_edt_analysis.csv')
    save_edt_analysis(analysis_df, analysis_path)
    print(f"EDT analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
