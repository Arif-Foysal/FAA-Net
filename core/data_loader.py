
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
        
        # Fill NaN with median (using train median for both ideally, but following notebook logic per df)
        # To be strictly correct, we should use train medians for test, but let's stick to notebook logic 
        # which filled per df:
        # "df_train_clean[col].fillna(median_val, inplace=True)" inside a loop over dataset
        # Actually notebook calculated median per df. We will replicate that.
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

    # 2. Label Encoding
    print("Encoding categorical features...")
    categorical_features = ['proto', 'service', 'state']
    target_categorical = 'attack_cat' # typically 'attack_cat'

    # Combine for consistent encoding
    df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    for col in categorical_features:
        le = LabelEncoder()
        le.fit(df_combined[col].astype(str))
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))

    # Encode attack_cat
    if target_categorical in df_train.columns:
        le_attack = LabelEncoder()
        le_attack.fit(df_combined[target_categorical].astype(str))
        df_train['attack_cat_encoded'] = le_attack.transform(df_train[target_categorical].astype(str))
        df_test['attack_cat_encoded'] = le_attack.transform(df_test[target_categorical].astype(str))

    # 3. Feature Selection (Dropping correlated)
    print("Removing high correlation features...")
    df_train.drop(columns=DROPPED_FEATURES, errors='ignore', inplace=True)
    df_test.drop(columns=DROPPED_FEATURES, errors='ignore', inplace=True)

    # 4. Prepare X and y
    drop_cols = ['label', 'attack_cat', 'attack_cat_encoded']
    X_train = df_train.drop(columns=drop_cols, errors='ignore')
    y_train = df_train['label']
    # Safety check if attack_cat_encoded exists
    y_train_cat = df_train['attack_cat_encoded'] if 'attack_cat_encoded' in df_train.columns else None

    X_test = df_test.drop(columns=drop_cols, errors='ignore')
    y_test = df_test['label']
    y_test_cat = df_test['attack_cat_encoded'] if 'attack_cat_encoded' in df_test.columns else None

    # 5. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
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
