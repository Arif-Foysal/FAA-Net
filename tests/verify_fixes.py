"""
Verification tests for EDA-Net (Entropy-Dynamic Attention Network).

Tests cover:
    1. EntropyDynamicTemperature — all three modes
    2. EDTAttention — forward pass, entropy & tau shapes
    3. EDTAttentionHead — prototype attention head
    4. MultiHeadEDT — multi-head wrapper with aggregation
    5. EDANet — full model forward pass + EDT info
    6. Ablation variants — VanillaDNN, FixedTemp, Heuristic, Full
    7. Loss functions — Focal, EntropyRegularization, EDANetLoss
    8. Data leakage prevention logic
    9. EDT behaviour validation (entropy→temperature relationship)
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from core.edt_attention import EntropyDynamicTemperature, EDTAttention
from core.model import EDTAttentionHead, MultiHeadEDT, EDANet
from core.ablation import (
    VanillaDNN_Ablation,
    FixedTempNet_Ablation,
    HeuristicEDTNet_Ablation,
    EDANet_Ablation,
)
from core.loss import (
    ImbalanceAwareFocalLoss,
    ImbalanceAwareFocalLoss_Logits,
    EntropyRegularization,
    EDANetLoss,
)


BATCH_SIZE = 16
INPUT_DIM = 30
ATTENTION_DIM = 16
N_PROTOTYPES = 4
NUM_HEADS = 2
passed = 0
failed = 0


def check(condition, msg):
    """Simple assertion helper with pass/fail tracking."""
    global passed, failed
    if condition:
        print(f"  ✓ {msg}")
        passed += 1
    else:
        print(f"  ✗ {msg}")
        failed += 1


# ─────────────────────────────────────────────────────────────────────────────
#  1. EntropyDynamicTemperature
# ─────────────────────────────────────────────────────────────────────────────

def test_edt_module():
    print("\n━━━ Test 1: EntropyDynamicTemperature ━━━")
    entropy_input = torch.rand(BATCH_SIZE, 1)  # normalised entropy ∈ [0,1]

    for mode in ['learned', 'heuristic', 'fixed']:
        edt = EntropyDynamicTemperature(tau_min=0.1, tau_max=5.0, hidden_dim=16, mode=mode)
        edt.eval()
        tau = edt(entropy_input)

        check(tau.shape == (BATCH_SIZE, 1), f"[{mode}] Output shape correct: {tau.shape}")
        check(tau.min().item() >= 0.1 - 1e-5, f"[{mode}] τ ≥ τ_min (min={tau.min().item():.4f})")
        check(tau.max().item() <= 5.0 + 1e-5, f"[{mode}] τ ≤ τ_max (max={tau.max().item():.4f})")

    # Learned mode should produce varied τ for varied entropy
    edt_learned = EntropyDynamicTemperature(tau_min=0.1, tau_max=5.0, hidden_dim=16, mode='learned')
    edt_learned.eval()
    low_entropy = torch.zeros(8, 1)
    high_entropy = torch.ones(8, 1)
    tau_low = edt_learned(low_entropy)
    tau_high = edt_learned(high_entropy)
    check(
        not torch.allclose(tau_low.mean(), tau_high.mean(), atol=1e-3),
        f"[learned] Different entropy → different τ (low={tau_low.mean():.3f}, high={tau_high.mean():.3f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  2. EDTAttention
# ─────────────────────────────────────────────────────────────────────────────

def test_edt_attention():
    print("\n━━━ Test 2: EDTAttention ━━━")
    edt_attn = EDTAttention(
        d_k=ATTENTION_DIM, n_keys=N_PROTOTYPES,
        tau_min=0.1, tau_max=5.0, tau_hidden_dim=16,
        edt_mode='learned', dropout=0.0, normalize_entropy=True
    )
    edt_attn.eval()

    q = torch.randn(BATCH_SIZE, ATTENTION_DIM)
    keys = torch.randn(N_PROTOTYPES, ATTENTION_DIM)
    values = torch.randn(N_PROTOTYPES, ATTENTION_DIM)

    output, attn_weights, edt_info = edt_attn(q, keys, values)

    check(output.shape == (BATCH_SIZE, ATTENTION_DIM), f"Output shape: {output.shape}")
    check(attn_weights.shape == (BATCH_SIZE, N_PROTOTYPES), f"Attention weights shape: {attn_weights.shape}")

    # Attention weights should sum to 1
    sums = attn_weights.sum(dim=-1)
    check(torch.allclose(sums, torch.ones_like(sums), atol=1e-4), f"Attention sums ≈ 1.0 (range: {sums.min():.4f}–{sums.max():.4f})")

    # EDT info
    check('entropy' in edt_info, "edt_info contains 'entropy'")
    check('tau' in edt_info, "edt_info contains 'tau'")
    check('raw_logits' in edt_info, "edt_info contains 'raw_logits'")
    check(edt_info['entropy'].shape == (BATCH_SIZE, 1), f"Entropy shape: {edt_info['entropy'].shape}")
    check(edt_info['tau'].shape == (BATCH_SIZE, 1), f"Tau shape: {edt_info['tau'].shape}")

    # Normalised entropy should be in [0, 1]
    check(edt_info['entropy'].min().item() >= -1e-5, "Entropy ≥ 0")
    check(edt_info['entropy'].max().item() <= 1.0 + 1e-5, "Entropy ≤ 1")


# ─────────────────────────────────────────────────────────────────────────────
#  3. EDTAttentionHead
# ─────────────────────────────────────────────────────────────────────────────

def test_edt_attention_head():
    print("\n━━━ Test 3: EDTAttentionHead ━━━")
    head = EDTAttentionHead(
        input_dim=INPUT_DIM, attention_dim=ATTENTION_DIM, n_prototypes=N_PROTOTYPES,
        tau_min=0.1, tau_max=5.0, tau_hidden_dim=16, edt_mode='learned',
        dropout=0.0, normalize_entropy=True
    )
    head.eval()

    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    output, attn_weights, edt_info = head(x)

    check(output.shape == (BATCH_SIZE, ATTENTION_DIM), f"Head output shape: {output.shape}")
    check(attn_weights.shape == (BATCH_SIZE, N_PROTOTYPES), f"Head attn weights shape: {attn_weights.shape}")

    # Test prototype initialisation
    dummy_protos = torch.randn(N_PROTOTYPES, INPUT_DIM)
    head.initialize_prototypes(dummy_protos)
    output2, _, _ = head(x)
    check(output2.shape == (BATCH_SIZE, ATTENTION_DIM), "Post-init forward pass works")


# ─────────────────────────────────────────────────────────────────────────────
#  4. MultiHeadEDT
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_head_edt():
    print("\n━━━ Test 4: MultiHeadEDT ━━━")
    mh = MultiHeadEDT(
        input_dim=INPUT_DIM, num_heads=NUM_HEADS, attention_dim=ATTENTION_DIM,
        n_prototypes=N_PROTOTYPES, tau_min=0.1, tau_max=5.0, tau_hidden_dim=16,
        edt_mode='learned', dropout=0.0, normalize_entropy=True
    )
    mh.eval()

    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    output, edt_info = mh(x)

    check(output.shape == (BATCH_SIZE, INPUT_DIM), f"Multi-head output shape: {output.shape}")
    check('mean_entropy' in edt_info, "Aggregated mean_entropy present")
    check('mean_tau' in edt_info, "Aggregated mean_tau present")
    check('head_weights' in edt_info, "Head weights present")
    check(edt_info['mean_entropy'].shape == (BATCH_SIZE, 1), f"Aggregated entropy shape: {edt_info['mean_entropy'].shape}")
    check(edt_info['mean_tau'].shape == (BATCH_SIZE, 1), f"Aggregated tau shape: {edt_info['mean_tau'].shape}")
    check(len(edt_info['per_head_tau']) == NUM_HEADS, f"Per-head tau count: {len(edt_info['per_head_tau'])}")

    # Test prototype initialisation via wrapper
    protos = np.random.randn(N_PROTOTYPES, INPUT_DIM).astype(np.float32)
    mh.initialize_all_prototypes(protos, device='cpu')
    output2, _ = mh(x)
    check(output2.shape == (BATCH_SIZE, INPUT_DIM), "Post-init multi-head forward works")


# ─────────────────────────────────────────────────────────────────────────────
#  5. EDANet full model
# ─────────────────────────────────────────────────────────────────────────────

def test_edanet():
    print("\n━━━ Test 5: EDANet Full Model ━━━")
    model = EDANet(
        input_dim=INPUT_DIM, num_heads=NUM_HEADS, attention_dim=ATTENTION_DIM,
        n_prototypes=N_PROTOTYPES, hidden_units=[64, 32],
        dropout_rate=0.1, attention_dropout=0.0,
        tau_min=0.1, tau_max=5.0, tau_hidden_dim=16,
        edt_mode='learned', normalize_entropy=True, output_logits=False
    )
    model.eval()

    x = torch.randn(BATCH_SIZE, INPUT_DIM)

    # Standard forward (returns probabilities)
    out = model(x)
    check(out.shape == (BATCH_SIZE, 1), f"EDANet output shape: {out.shape}")
    check(out.min().item() >= 0.0, "Probabilities ≥ 0")
    check(out.max().item() <= 1.0, "Probabilities ≤ 1")

    # Forward with EDT info
    out2, edt_info = model(x, return_edt_info=True)
    check(out2.shape == (BATCH_SIZE, 1), "EDANet w/ info output shape correct")
    check('edt_attention' in edt_info, "EDT attention info present")
    check('se_weights' in edt_info, "SE weights present")

    # last_edt_info is cached
    check(model.last_edt_info is not None, "last_edt_info cached after forward pass")

    # Logits mode
    model_logits = EDANet(
        input_dim=INPUT_DIM, num_heads=NUM_HEADS, attention_dim=ATTENTION_DIM,
        n_prototypes=N_PROTOTYPES, hidden_units=[64, 32],
        output_logits=True
    )
    model_logits.eval()
    logits = model_logits(x)
    check(logits.min().item() < 0 or logits.max().item() > 1,
          "Logits mode outputs raw values (not clamped to [0,1])")

    # Parameter counting
    n_params = model.count_parameters()
    check(n_params > 0, f"Parameter count: {n_params:,}")

    print(f"  → Total parameters: {n_params:,}")


# ─────────────────────────────────────────────────────────────────────────────
#  6. Ablation variants
# ─────────────────────────────────────────────────────────────────────────────

def test_ablation_variants():
    print("\n━━━ Test 6: Ablation Variants ━━━")
    x = torch.randn(BATCH_SIZE, INPUT_DIM)

    # VanillaDNN — no attention
    vanilla = VanillaDNN_Ablation(input_dim=INPUT_DIM, hidden_units=[64, 32], dropout_rate=0.1)
    vanilla.eval()
    out_v = vanilla(x)
    check(out_v.shape == (BATCH_SIZE, 1), f"VanillaDNN output shape: {out_v.shape}")
    check(hasattr(vanilla, 'output_logits') and vanilla.output_logits,
          "VanillaDNN outputs raw logits")

    # FixedTempNet — fixed τ (no EDT learning)
    fixed = FixedTempNet_Ablation(
        input_dim=INPUT_DIM, num_heads=NUM_HEADS, attention_dim=ATTENTION_DIM,
        n_prototypes=N_PROTOTYPES, hidden_units=[64, 32]
    )
    fixed.eval()
    out_f = fixed(x)
    check(out_f.shape == (BATCH_SIZE, 1), f"FixedTempNet output shape: {out_f.shape}")
    check(fixed.edt_mode == 'fixed', f"FixedTempNet mode: {fixed.edt_mode}")

    # HeuristicEDT — analytic τ mapping
    heuristic = HeuristicEDTNet_Ablation(
        input_dim=INPUT_DIM, num_heads=NUM_HEADS, attention_dim=ATTENTION_DIM,
        n_prototypes=N_PROTOTYPES, hidden_units=[64, 32]
    )
    heuristic.eval()
    out_h = heuristic(x)
    check(out_h.shape == (BATCH_SIZE, 1), f"HeuristicEDT output shape: {out_h.shape}")
    check(heuristic.edt_mode == 'heuristic', f"HeuristicEDT mode: {heuristic.edt_mode}")

    # Full EDANet ablation variant
    full = EDANet_Ablation(
        input_dim=INPUT_DIM, num_heads=NUM_HEADS, attention_dim=ATTENTION_DIM,
        n_prototypes=N_PROTOTYPES, hidden_units=[64, 32]
    )
    full.eval()
    out_e = full(x)
    check(out_e.shape == (BATCH_SIZE, 1), f"EDANet_Ablation output shape: {out_e.shape}")
    check(full.edt_mode == 'learned', f"EDANet_Ablation mode: {full.edt_mode}")


# ─────────────────────────────────────────────────────────────────────────────
#  7. Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_functions():
    print("\n━━━ Test 7: Loss Functions ━━━")

    targets = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
    probs = torch.sigmoid(torch.randn(BATCH_SIZE, 1))
    logits = torch.randn(BATCH_SIZE, 1)

    # Focal loss (probability input)
    focal = ImbalanceAwareFocalLoss(gamma=2.0, class_counts=[60, 40])
    loss_p = focal(probs, targets)
    check(loss_p.dim() == 0, f"FocalLoss(prob) returns scalar: {loss_p.item():.4f}")
    check(loss_p.item() >= 0, "FocalLoss(prob) is non-negative")

    # Focal loss (logit input)
    focal_logits = ImbalanceAwareFocalLoss_Logits(gamma=2.0, class_counts=[60, 40])
    loss_l = focal_logits(logits, targets)
    check(loss_l.dim() == 0, f"FocalLoss(logits) returns scalar: {loss_l.item():.4f}")
    check(loss_l.item() >= 0, "FocalLoss(logits) is non-negative")

    # Entropy regularisation
    entropy_reg = EntropyRegularization(weight=0.01)

    # Simulate EDT info from model
    mock_edt_info = {
        'edt_attention': {
            'mean_tau': torch.rand(BATCH_SIZE, 1) * 4 + 0.5  # varied τ
        }
    }
    reg_loss = entropy_reg(mock_edt_info)
    check(reg_loss.dim() == 0, f"EntropyReg returns scalar: {reg_loss.item():.6f}")
    check(reg_loss.item() <= 0, "EntropyReg is ≤ 0 (negative variance penalty)")

    # Collapsed τ should give weaker penalty
    mock_collapsed = {
        'edt_attention': {
            'mean_tau': torch.ones(BATCH_SIZE, 1) * 2.0  # constant τ
        }
    }
    reg_collapsed = entropy_reg(mock_collapsed)
    check(
        abs(reg_collapsed.item()) < abs(reg_loss.item()) + 1e-6,
        f"Collapsed τ has smaller penalty: {reg_collapsed.item():.6f} vs {reg_loss.item():.6f}"
    )

    # None input returns 0
    reg_none = entropy_reg(None)
    check(reg_none.item() == 0.0, "EntropyReg(None) = 0.0")

    # EDANetLoss — combined
    model = EDANet(
        input_dim=INPUT_DIM, num_heads=NUM_HEADS, attention_dim=ATTENTION_DIM,
        n_prototypes=N_PROTOTYPES, hidden_units=[64, 32],
        output_logits=False
    )
    model.eval()
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    preds = model(x)

    eda_loss = EDANetLoss(gamma=2.0, class_counts=[60, 40], entropy_reg_weight=0.01)
    total, components = eda_loss(preds, targets, model.last_edt_info)
    check(total.dim() == 0, f"EDANetLoss total is scalar: {total.item():.4f}")
    check('focal' in components, "Components contain 'focal'")
    check('entropy_reg' in components, "Components contain 'entropy_reg'")
    check('total' in components, "Components contain 'total'")


# ─────────────────────────────────────────────────────────────────────────────
#  8. Data leakage prevention
# ─────────────────────────────────────────────────────────────────────────────

def test_leakage_logic():
    print("\n━━━ Test 8: Data Leakage Prevention ━━━")
    train_cats = np.array(['A', 'B', 'C', 'A'])
    test_cats = np.array(['A', 'D', 'E', 'B'])  # D, E are unseen

    le = LabelEncoder()
    le.fit(train_cats)

    # Safe transform: unseen categories → 0
    mapping = {label: idx for idx, label in enumerate(le.classes_)}
    s_test = pd.Series(test_cats)
    transformed = s_test.map(mapping).fillna(0).astype(int)

    check(transformed.isna().sum() == 0, "No NaNs after safe transform")
    check(len(transformed) == len(test_cats), "Output length matches input")
    check(transformed.iloc[0] == mapping['A'], f"Known category 'A' mapped correctly → {mapping['A']}")
    check(transformed.iloc[3] == mapping['B'], f"Known category 'B' mapped correctly → {mapping['B']}")
    check(transformed.iloc[1] == 0, "Unknown 'D' falls back to 0")
    check(transformed.iloc[2] == 0, "Unknown 'E' falls back to 0")


# ─────────────────────────────────────────────────────────────────────────────
#  9. EDT Behaviour: Entropy → Temperature Relationship
# ─────────────────────────────────────────────────────────────────────────────

def test_edt_behaviour():
    print("\n━━━ Test 9: EDT Behaviour Validation ━━━")
    edt_attn = EDTAttention(
        d_k=ATTENTION_DIM, n_keys=N_PROTOTYPES,
        tau_min=0.1, tau_max=5.0, tau_hidden_dim=32,
        edt_mode='heuristic', dropout=0.0, normalize_entropy=True
    )
    edt_attn.eval()

    keys = torch.randn(N_PROTOTYPES, ATTENTION_DIM)
    values = torch.randn(N_PROTOTYPES, ATTENTION_DIM)

    # Uniform queries → high entropy → should give LOW tau (heuristic mode)
    q_uniform = torch.zeros(8, ATTENTION_DIM)  # all same → softmax uniform
    _, _, info_uniform = edt_attn(q_uniform, keys, values)

    # Peaked queries → low entropy → should give HIGH tau (heuristic mode)
    q_peaked = torch.zeros(8, ATTENTION_DIM)
    q_peaked[:, 0] = 10.0  # strongly peaked in one direction
    _, _, info_peaked = edt_attn(q_peaked, keys, values)

    # Heuristic: τ = τ_max·(1−H̃) + τ_min
    # High entropy H̃≈1 → τ ≈ τ_min
    # Low entropy  H̃≈0 → τ ≈ τ_max
    check(
        info_uniform['tau'].mean().item() < info_peaked['tau'].mean().item(),
        f"Heuristic: high entropy → lower τ ({info_uniform['tau'].mean():.3f}) < "
        f"low entropy → higher τ ({info_peaked['tau'].mean():.3f})"
    )
    check(
        info_uniform['entropy'].mean().item() > info_peaked['entropy'].mean().item(),
        f"Uniform queries have higher entropy ({info_uniform['entropy'].mean():.3f}) > "
        f"peaked queries ({info_peaked['entropy'].mean():.3f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EDA-Net Verification Tests")
    print("=" * 60)

    test_edt_module()
    test_edt_attention()
    test_edt_attention_head()
    test_multi_head_edt()
    test_edanet()
    test_ablation_variants()
    test_loss_functions()
    test_leakage_logic()
    test_edt_behaviour()

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("  All tests passed! ✓")
        sys.exit(0)
