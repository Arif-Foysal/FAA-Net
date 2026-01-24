
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib
import os
from .config import DROPPED_FEATURES, RANDOM_STATE

def get_data_paths(data_dir="/content"):
    """
    Detect the running environment and return appropriate data paths.
    Accepts data_dir kwarg for flexibility.
    """
    train_file = Path(data_dir) / "UNSW_NB15_training-set.csv"
    test_file = Path(data_dir) / "UNSW_NB15_testing-set.csv"

    if not train_file.exists() or not test_file.exists():
        # Fallback to local directory if not found in /content
        if Path("UNSW_NB15_training-set.csv").exists():
             return "UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"
        raise FileNotFoundError(
            f"Required dataset files not found in {data_dir} or current directory. "
            "Please ensure UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv are present."
        )

    return str(train_file), str(test_file)

def load_and_preprocess_data(data_dir="/content", cache_file="faiia_preprocessed_data.pkl"):
    """
    Loads, cleans, encodes, selects features, and scales the data.
    Returns X_train_scaled, X_test_scaled, y_train, y_test, y_train_cat, y_test_cat
    """
    if os.path.exists(cache_file):
        print(f"Loading cached preprocessed data from {cache_file}...")
        try:
            return joblib.load(cache_file)
        except:
            print("Failed to load cache, reprocessing...")

    print("Loading raw data...")
    train_path, test_path = get_data_paths(data_dir)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 1. Data Cleaning
    print("Cleaning data...")
    for df in [df_train, df_test]:
        if 'id' in df.columns:
            df.drop('id', axis=1, inplace=True)
        if 'service' in df.columns:
            df['service'] = df['service'].replace('-', 'none')
        
        # Handle infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Imputation: Compute medians ONLY on train, apply to both
    print("Imputing missing values...")
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    train_medians = df_train[numeric_cols].median()
    
    df_train[numeric_cols] = df_train[numeric_cols].fillna(train_medians)
    df_test[numeric_cols] = df_test[numeric_cols].fillna(train_medians) # Apply train medians to test

    # 2. Label Encoding (Safe for Unseen Labels)
    print("Encoding categorical features...")
    categorical_features = ['proto', 'service', 'state']
    target_categorical = 'attack_cat' 

    # Helper function for safe encoding
    def safe_transform(encoder, series, unknown_value=-1):
        # Determine known classes
        known_classes = set(encoder.classes_)
        # Map unknown to unknown_value (or handle strategy)
        # Here we map to the first class (or a specific 'unknown' if added)
        # For simplicity in this pipeline, we'll map to mode (most frequent) of train
        # OR just use a robust method:
        s_series = series.astype(str)
        # Replace unseen with mode of encoder training data? 
        # Better: fit encoder on train, if test has new val, map to 'other' or 0
        
        # Fast way: use pandas map
        mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
        return s_series.map(mapping).fillna(0).astype(int) # Default to 0 if unknown

    for col in categorical_features:
        le = LabelEncoder()
        # FIT ONLY ON TRAIN
        le.fit(df_train[col].astype(str))
        
        df_train[col] = le.transform(df_train[col].astype(str))
        # TRANSFORM TEST (Handle unseen)
        df_test[col] = safe_transform(le, df_test[col])

    # Encode attack_cat
    if target_categorical in df_train.columns:
        le_attack = LabelEncoder()
        le_attack.fit(df_train[target_categorical].astype(str))
        df_train['attack_cat_encoded'] = le_attack.transform(df_train[target_categorical].astype(str))
        # For test targets, we can just transform. If new attack appears in test, it's problematic for eval anyway.
        # But let's assume test labels are within train scope or strictly mapped.
        # We'll use safe transform to avoid crash.
        df_test['attack_cat_encoded'] = safe_transform(le_attack, df_test[target_categorical])

    # 3. Feature Selection (Dropping correlated)
    print("Removing high correlation features...")
    df_train.drop(columns=DROPPED_FEATURES, errors='ignore', inplace=True)
    df_test.drop(columns=DROPPED_FEATURES, errors='ignore', inplace=True)

    # 4. Prepare X and y
    drop_cols = ['label', 'attack_cat', 'attack_cat_encoded']
    X_train = df_train.drop(columns=drop_cols, errors='ignore')
    y_train = df_train['label']
    y_train_cat = df_train['attack_cat_encoded'] if 'attack_cat_encoded' in df_train.columns else None

    X_test = df_test.drop(columns=drop_cols, errors='ignore')
    y_test = df_test['label']
    y_test_cat = df_test['attack_cat_encoded'] if 'attack_cat_encoded' in df_test.columns else None

    # 5. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    # FIT ONLY ON TRAIN
    X_train_scaled = scaler.fit_transform(X_train)
    # TRANSFORM TEST
    X_test_scaled = scaler.transform(X_test)

    data = (X_train_scaled, X_test_scaled, y_train, y_test, y_train_cat, y_test_cat)
    
    print(f"Saving preprocessed data to {cache_file}...")
    joblib.dump(data, cache_file)
    
    return data

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=256, val_split=0.1):
    """
    Creates Train, Validation, and Test DataLoaders.
    Splits X_train into Train/Val.
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

    # Split train into train/val
    # Note: original notebook used test_loader as valid_loader in some parts, 
    # but best practice is a separate val set.
    # We will create a validation set from training data.
    
    dataset_full = TensorDataset(X_train_tensor, y_train_tensor)
    train_size = int((1 - val_split) * len(dataset_full))
    val_size = len(dataset_full) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset_full, [train_size, val_size], 
        generator=torch.Generator().manual_seed(RANDOM_STATE)
    )
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_test_tensor
