
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from core.config import V3_CONFIG, RANDOM_STATE
from core.data_loader import load_and_preprocess_data, create_dataloaders
from core.model import EDANv3, MinorityPrototypeGenerator
from core.loss import ImbalanceAwareFocalLoss
from core.trainer import train_model
from core.utils import set_all_seeds, evaluate_model, print_metrics

def main():
    print("="*60)
    print("FAIIA-IDS: Main Model Training (EDAN v3)")
    print("="*60)

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_all_seeds(RANDOM_STATE)

    # 2. Data Loading
    # Check if we are in colab or local
    data_dir = "/content"
    if not os.path.exists(data_dir):
        data_dir = "." # Fallback to current dir if not in Colab
        
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = load_and_preprocess_data(data_dir=data_dir)
    
    # 3. Create DataLoaders
    train_loader, val_loader, test_loader, X_test_tensor = create_dataloaders(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        batch_size=V3_CONFIG['batch_size']
    )

    # 4. Extract Minority Prototypes
    print("\nExtracting Minority Prototypes...")
    # Re-construct X_train from scaler to extract minority? 
    # load_and_preprocess_data returns scaled data.
    # We need to filter X_train_scaled by y_train
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
        output_logits=False # Main model uses Sigmoid
    ).to(device)
    
    # Initialize prototypes
    model.faiia.initialize_all_prototypes(minority_prototypes, device)
    
    print(f"\nModel initialized with {model.count_parameters():,} parameters.")

    # 6. Loss Function
    # For EDANv3 standard, we use ImbalanceAwareFocalLoss (expects probs)
    class_counts = [len(X_majority), len(X_minority)]
    criterion = ImbalanceAwareFocalLoss(
        gamma=V3_CONFIG['focal_gamma'],
        class_counts=class_counts # Use calculated weights
    )

    # 7. Train
    model, history = train_model(
        model, train_loader, val_loader, V3_CONFIG, criterion, device
    )

    # 8. Evaluate
    print("\nEvaluating on Test Set...")
    metrics = evaluate_model(model, X_test_tensor, y_test, device)
    print_metrics(metrics, "EDANv3 Test Results")

    # Save model
    save_dir = "."
    if os.path.exists("/content/drive/MyDrive"):
        save_dir = "/content/drive/MyDrive/FAIIA_Models"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving model to Google Drive: {save_dir}")
    
    save_path = os.path.join(save_dir, 'edan_v3_main.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
