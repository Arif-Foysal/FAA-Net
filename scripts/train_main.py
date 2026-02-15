
import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from core.config import V3_CONFIG, RANDOM_STATE
from core.data_loader import load_and_preprocess_data, create_dataloaders
from core.model import EDANv3, MinorityPrototypeGenerator
from core.loss import ImbalanceAwareFocalLoss_Logits
from core.evidential import FocalEvidentialLoss
from core.trainer import train_model
from core.utils import (set_all_seeds, evaluate_model, print_metrics,
                        save_training_history, save_predictions,
                        evaluate_evidential_uncertainty, calibrate_epistemic_threshold)

def main():
    print("="*60)
    print("FAIIA-IDS: Main Model Training (EDAN v3)")
    print("="*60)

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_all_seeds(RANDOM_STATE)

    use_evidential = V3_CONFIG.get('evidential', False)
    if use_evidential:
        print("Mode: Evidential Deep Learning (EDL)")
    else:
        print("Mode: Standard (sigmoid/logit output)")

    # 2. Data Loading
    # Check if we are in colab or local
    data_dir = "/content"
    if not os.path.exists(data_dir):
        data_dir = "." # Fallback to current dir if not in Colab
        
    X_train_scaled, X_test_scaled, y_train, y_test, y_train_cat, y_test_cat = load_and_preprocess_data(data_dir=data_dir)
    
    # 3. Create DataLoaders
    train_loader, val_loader, test_loader, X_test_tensor = create_dataloaders(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        batch_size=V3_CONFIG['batch_size']
    )

    # 4. Extract Minority Prototypes
    print("\nExtracting Minority Prototypes...")
    minority_mask = y_train.values == 1
    X_minority = X_train_scaled[minority_mask]
    X_majority = X_train_scaled[~minority_mask]
    
    print(f"  Minority samples: {len(X_minority)}")
    print(f"  Majority samples: {len(X_majority)}")
    
    proto_gen = MinorityPrototypeGenerator(n_prototypes=V3_CONFIG['n_prototypes'], random_state=RANDOM_STATE)
    minority_prototypes = proto_gen.fit(X_minority)
    
    # 5. Initialize Model
    input_dim = X_train_scaled.shape[1]
    model = EDANv3(
        input_dim=input_dim,
        num_heads=V3_CONFIG['num_heads'],
        attention_dim=V3_CONFIG['attention_dim'],
        n_prototypes=V3_CONFIG['n_prototypes'],
        hidden_units=V3_CONFIG['hidden_units'],
        dropout_rate=V3_CONFIG['dropout_rate'],
        attention_dropout=V3_CONFIG['attention_dropout'],
        focal_alpha=V3_CONFIG['focal_alpha'],
        focal_gamma=V3_CONFIG['focal_gamma'],
        num_classes=1,
        output_logits=True if not use_evidential else False,
        evidential=use_evidential,
    ).to(device)
    
    # Initialize prototypes
    model.faiia.initialize_all_prototypes(minority_prototypes, device)
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")

    # 6. Loss Function
    class_counts = [len(X_majority), len(X_minority)]

    if use_evidential:
        # Compute class weights for focal evidential loss
        # Use sqrt-dampened weights to avoid starving evidence accumulation
        total = sum(class_counts)
        class_weights = torch.tensor(
            [np.sqrt(total / (2 * class_counts[0])),   # Weight for normal class
             np.sqrt(total / (2 * class_counts[1]))],  # Weight for attack (minority) class
            dtype=torch.float32
        ).to(device)

        criterion = FocalEvidentialLoss(
            num_classes=2,
            annealing_epochs=V3_CONFIG.get('annealing_epochs', 10),
            gamma=V3_CONFIG.get('evidential_focal_gamma', 2.0),
            class_weights=class_weights,
        ).to(device)
        print(f"  Loss: FocalEvidentialLoss (γ={V3_CONFIG.get('evidential_focal_gamma', 2.0)}, "
              f"anneal={V3_CONFIG.get('annealing_epochs', 10)} epochs)")
        print(f"  Class weights: normal={class_weights[0]:.3f}, attack={class_weights[1]:.3f}")
    else:
        criterion = ImbalanceAwareFocalLoss_Logits(
            gamma=V3_CONFIG['focal_gamma'],
            class_counts=class_counts
        )

    # 7. Train
    model, history = train_model(
        model, train_loader, val_loader, V3_CONFIG, criterion, device
    )

    # 8. Evaluate
    print("\nEvaluating on Test Set...")
    metrics, y_probs, y_pred = evaluate_model(
        model, X_test_tensor, y_test, device,
        optimize_threshold=use_evidential,
    )
    print_metrics(metrics, "EDANv3 Test Results")

    # 9. Evidential uncertainty analysis
    if use_evidential:
        print("\n" + "="*60)
        print("Evidential Uncertainty Analysis")
        print("="*60)

        # Detailed per-sample analysis
        results_df, summary = evaluate_evidential_uncertainty(
            model, X_test_tensor, y_test, device,
            y_categories=y_test_cat,
        )

        print(f"\n  Overall epistemic U:  {summary['overall_epistemic_u_mean']:.4f} ± {summary['overall_epistemic_u_std']:.4f}")
        print(f"  Correct predictions:  {summary['correct_epistemic_u_mean']:.4f}")
        print(f"  Incorrect predictions: {summary['incorrect_epistemic_u_mean']:.4f}")
        print(f"  Normal samples:       {summary['normal_epistemic_u_mean']:.4f}")
        print(f"  Attack samples:       {summary['attack_epistemic_u_mean']:.4f}")

        if 'epistemic_u_by_category' in summary:
            print("\n  Epistemic U by attack category:")
            for cat, u_val in sorted(summary['epistemic_u_by_category'].items(),
                                      key=lambda x: x[1], reverse=True):
                print(f"    Category {cat:>3}: {u_val:.4f}")

        # Calibrate threshold for zero-day detection
        print("\nCalibrating epistemic threshold on training data...")
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        epistemic_threshold = calibrate_epistemic_threshold(
            model, X_train_tensor, device,
            percentile=V3_CONFIG.get('calibration_percentile', 95.0),
        )

        # Extract uncertainty arrays for saving
        with torch.no_grad():
            test_output = model(X_test_tensor.to(device))
            epistemic_u = test_output['epistemic_uncertainty'].cpu().numpy().flatten()
            evidence = test_output['evidence'].cpu().numpy()

    # 10. Save artifacts
    save_dir = "."
    if os.path.exists("/content/drive/MyDrive"):
        save_dir = "/content/drive/MyDrive/FAIIA_Models"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving artifacts to Google Drive: {save_dir}")

    # Save Metrics CSV
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, 'edan_v3_metrics.csv'), index=False)

    # Save Model
    save_path = os.path.join(save_dir, 'edan_v3_main.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save History (for convergence plots F3-F5)
    hist_path = os.path.join(save_dir, 'edan_v3_history.csv')
    save_training_history(history, hist_path)
    print(f"Training history saved to {hist_path}")

    # Save Predictions (for ROC/PR F7-F8 and Per-Attack Analysis)
    pred_path = os.path.join(save_dir, 'edan_v3_predictions.npz')
    if use_evidential:
        save_predictions(y_test, y_probs, pred_path,
                         epistemic_u=epistemic_u, evidence=evidence)
    else:
        save_predictions(y_test, y_probs, pred_path)
    print(f"Predictions saved to {pred_path}")

    # Save evidential analysis results
    if use_evidential:
        uncertainty_path = os.path.join(save_dir, 'edan_v3_uncertainty_analysis.csv')
        results_df.to_csv(uncertainty_path, index=False)
        print(f"Uncertainty analysis saved to {uncertainty_path}")

        # Save calibrated threshold
        threshold_path = os.path.join(save_dir, 'epistemic_threshold.txt')
        with open(threshold_path, 'w') as f:
            f.write(f"{epistemic_threshold:.6f}\n")
        print(f"Epistemic threshold saved to {threshold_path}")

if __name__ == "__main__":
    main()
