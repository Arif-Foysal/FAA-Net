# EDA-Net: Entropy-Dynamic Attention Network for Network Intrusion Detection

A deep learning intrusion detection system that dynamically adapts attention sharpness per sample using **Entropy-Dynamic Temperature (EDT)** — a novel attention mechanism that derives softmax temperature from the information entropy of attention logits.

## Core Innovation: EDT Attention

Standard attention uses a fixed temperature scaling ($1/\sqrt{d_k}$). EDT replaces this with a **per-sample dynamic temperature** derived from the entropy of the raw attention distribution:

1. **Feature Entropy**: $H(x) = -\sum p(x) \log p(x)$ where $p(x) = \text{softmax}(QK^T / \sqrt{d_k})$
2. **Dynamic Temperature**: $\tau(x) = \tau_{\min} + (\tau_{\max} - \tau_{\min}) \cdot \sigma(\text{MLP}(\tilde{H}(x)))$
3. **EDT-Modulated Attention**: $\text{Attn} = \text{softmax}(QK^T / (\tau(x) \cdot \sqrt{d_k})) \cdot V$

**Key behaviour**:
- Ambiguous samples (high entropy) → low τ → **sharp** attention (focus on most relevant prototypes)
- Confident samples (low entropy) → high τ → **smooth** attention (spread across prototypes)

## Architecture

```
Input → BatchNorm → Multi-Head EDT Attention (with learned prototypes)
      → Squeeze-and-Excitation → Residual MLP Blocks → Classifier Head
```

## Directory Structure

```
core/                       # Core modules
├── edt_attention.py        # EDT attention mechanism (core contribution)
├── model.py                # EDANet, MultiHeadEDT, EDTAttentionHead
├── config.py               # EDA_CONFIG + ABLATION_CONFIGS
├── loss.py                 # Focal loss + entropy regularisation
├── trainer.py              # Training loop with EDT metric logging
├── ablation.py             # Ablation model variants
├── data_loader.py          # UNSW-NB15 data loading & preprocessing
└── utils.py                # Evaluation, EDT analysis collection, I/O

scripts/                    # Executable training scripts
├── train_main.py           # Train the full EDA-Net model
├── run_ablation.py         # Run the 8-experiment ablation study
└── train_baselines.py      # Train XGBoost & LightGBM baselines

notebooks/                  # Paper artifacts
└── paper_artifacts.ipynb   # Generate all figures & tables for the paper

tests/                      # Verification tests
└── verify_fixes.py         # 9 test suites covering EDT, model, loss, ablations

paper_draft/                # LaTeX paper sources
```

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare dataset:**

   Place `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` in the project root (or `/content` for Google Colab).

## Usage

### Train the main EDA-Net model

```bash
python scripts/train_main.py
```

Outputs saved to `EDANet_Models/`:
- `edanet_main.pt` — trained model weights
- `edanet_metrics.csv` — F1, precision, recall, AUC
- `edanet_history.csv` — per-epoch training history (including τ stats)
- `edanet_edt_analysis.csv` — per-sample entropy, τ, and predictions

### Run the ablation study

```bash
python scripts/run_ablation.py
```

Runs 8 experiments:
| # | Variant | Purpose |
|---|---------|---------|
| 1 | Vanilla DNN + BCE | No-attention, no-focal baseline |
| 2 | Vanilla DNN + Focal | Focal loss impact without attention |
| 3 | Fixed-Temp Attention + Focal | Attention without EDT |
| 4 | Heuristic EDT + Focal | Analytic τ mapping (no MLP) |
| 5 | Full EDA-Net | Complete system (learned EDT + focal) |
| 6 | Narrow τ range [0.5, 2.0] | Reduced adaptation capacity |
| 7 | Wide τ range [0.01, 10.0] | Increased adaptation capacity |
| 8 | No entropy normalisation | Unnormalised entropy input |

### Train ML baselines

```bash
python scripts/train_baselines.py
```

Trains XGBoost and LightGBM classifiers for comparison.

### Generate paper figures & tables

Open `notebooks/paper_artifacts.ipynb` and run all cells. Requires completed training runs.

### Run tests

```bash
python tests/verify_fixes.py
```

Validates EDT attention mechanics, model forward passes, loss functions, ablation variants, and data leakage prevention.

## Google Colab

```python
!git clone https://github.com/your-repo/eda-net.git
%cd eda-net
!pip install -r requirements.txt
# Upload UNSW_NB15 dataset files to /content
!python scripts/train_main.py
```

## EDT Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `learned` | MLP maps entropy → temperature | Full EDA-Net (default) |
| `heuristic` | Analytic: τ = τ_max·(1−H̃) + τ_min | Ablation baseline |
| `fixed` | Learnable but entropy-independent scalar | Attention w/o EDT |

## Dataset

**UNSW-NB15** — binary classification (Normal vs Attack).  
9 high-correlation features (>0.95) are dropped during preprocessing.

## Dependencies

- PyTorch
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- xgboost, lightgbm
- joblib