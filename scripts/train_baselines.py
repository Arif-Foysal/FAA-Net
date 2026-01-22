
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import numpy as np
from core.config import RANDOM_STATE
from core.data_loader import load_and_preprocess_data
from core.utils import set_all_seeds, evaluate_model, print_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import xgboost as xgb
import lightgbm as lgb
import joblib

def evaluate_sklearn_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates sklearn-API compatible models (XGBoost, LightGBM).
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > threshold).astype(int)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_test, y_probs),
        'Avg Precision': average_precision_score(y_test, y_probs)
    }
    return metrics

def main():
    print("="*60)
    print("FAIIA-IDS: Training Baselines (XGBoost & LightGBM)")
    print("="*60)

    # 1. Threading optimization for CPU
    # Avoids oversubscription issues if running alongside PyTorch
    os.environ["OMP_NUM_THREADS"] = "4" 

    set_all_seeds(RANDOM_STATE)

    # 2. Data
    data_dir = "/content" if os.path.exists("/content") else "."
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = load_and_preprocess_data(data_dir=data_dir)

    results = {}

    # --- XGBoost ---
    print("\n--- Training XGBoost ---")
    # Using parameters typically effective for imbalanced tabular data
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), # Handle imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    xgb_metrics = evaluate_sklearn_model(xgb_model, X_test_scaled, y_test)
    print_metrics(xgb_metrics, "XGBoost Results")
    results['XGBoost'] = xgb_metrics
    
    save_dir = "."
    if os.path.exists("/content/drive/MyDrive"):
        save_dir = "/content/drive/MyDrive/FAIIA_Models"
        os.makedirs(save_dir, exist_ok=True)

    joblib.dump(xgb_model, os.path.join(save_dir, 'xgboost_baseline.joblib'))

    # --- LightGBM ---
    print("\n--- Training LightGBM ---")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        class_weight='balanced', # Automatic imbalance handling
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    lgb_model.fit(X_train_scaled, y_train)
    lgb_metrics = evaluate_sklearn_model(lgb_model, X_test_scaled, y_test)
    print_metrics(lgb_metrics, "LightGBM Results")
    results['LightGBM'] = lgb_metrics
    joblib.dump(lgb_model, os.path.join(save_dir, 'lightgbm_baseline.joblib'))

    # Summary
    print("\n=== Baseline Models Summary ===")
    df = pd.DataFrame(results).T
    print(df)
    df.to_csv('baseline_summary.csv')

if __name__ == "__main__":
    main()
