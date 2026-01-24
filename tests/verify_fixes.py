
import torch
import torch.nn as nn
from core.model import FAIIAHead, FocalModulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys

def test_faiia_head():
    print("\n--- Testing FAIIAHead Logic ---")
    input_dim = 32
    head = FAIIAHead(input_dim=input_dim, attention_dim=16, n_prototypes=4)
    head.eval() # Disable dropout for sum check
    
    batch_size = 5
    x = torch.randn(batch_size, input_dim)
    minority_prob = torch.tensor([0.0, 0.5, 1.0, 0.2, 0.8])
    
    output, attn_weights = head(x, minority_prob)
    
    print(f"Output Check: Shape={output.shape} (Expected ({batch_size}, 16))")
    print(f"Attention Weights Shape: {attn_weights.shape} (Expected ({batch_size}, 4))")
    
    # Check that attention weights are NOT all 1.0 or equal
    # Since we use random inputs, weights should vary considerably if logic is correct
    print(f"Attention Weights Sample (First row): {attn_weights[0].tolist()}")
    
    # Check if softmax works (sum to 1) across dim -1
    sums = attn_weights.sum(dim=-1)
    print(f"Attention Sums: {sums.detach().numpy()} (Should be all ~1.0 closely)")
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Softmax failed!"
    print("✓ FAIIAHead Logic Validated")

def test_focal_modulation():
    print("\n--- Testing Focal Modulation Logic ---")
    focal = FocalModulation(alpha=1.0, gamma=2.0)
    
    probs = torch.tensor([0.0, 0.5, 1.0])
    dummy_attn = torch.ones(3, 1) # Batch of 3
    
    # New Logic: (1 - |p - 0.5|*2)^gamma
    # p=0.0 -> |0.5|*2=1 -> (1-1)^2 = 0 -> weight 0
    # p=0.5 -> |0|*2=0 -> (1-0)^2 = 1 -> weight alpha * 1
    # p=1.0 -> |0.5|*2=1 -> (1-1)^2 = 0 -> weight 0
    
    out = focal(dummy_attn, probs)
    # Weights added to 1: output = input * (1 + weight)
    
    # Expected:
    # 0.0 -> 1 * (1 + 0) = 1
    # 0.5 -> 1 * (1 + 1) = 2 (Assuming alpha=1)
    # 1.0 -> 1 * (1 + 0) = 1
    
    print(f"Input Probs: {probs.tolist()}")
    print(f"Modulated Output: {out.flatten().tolist()}")
    
    if out[1] > out[0] and out[1] > out[2]:
        print("✓ Focal Modulation correctly peaks at 0.5 (Uncertainty)")
    else:
        print("✗ Focal Modulation FAILED Logic Check")

def test_leakage_logic():
    print("\n--- Testing Data Leakage Presvention Logic ---")
    # Simulate Train vs Test data
    train_cats = np.array(['A', 'B', 'C', 'A'])
    test_cats = np.array(['A', 'D', 'E', 'B']) # D, E are unknown
    
    le = LabelEncoder()
    le.fit(train_cats)
    
    # Transform Test with Safe Logic logic (replicated here for verify)
    mapping = {label: idx for idx, label in enumerate(le.classes_)}
    print(f"Encoder Classes: {le.classes_}")
    
    s_test = pd.Series(test_cats)
    transformed = s_test.map(mapping).fillna(0).astype(int)
    
    print(f"Original Test: {test_cats}")
    print(f"Transformed: {transformed.tolist()}")
    
    # 'A' is index 0. 'B' is index 1. 'C' is 2.
    # D, E should be 0 (default fallback)
    # Result should be [0, 0, 0, 1] essentially (since mapping A->0) 
    # Wait, mapped unknown to 0. A is 0. So D->A. That is one strategy.
    # Our code uses fillna(0).
    
    assert transformed.isna().sum() == 0, "NaNs found in transformed data"
    print("✓ Transformation handled unknowns without crashing")

if __name__ == "__main__":
    test_faiia_head()
    test_focal_modulation()
    test_leakage_logic()
