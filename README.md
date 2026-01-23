# FAIIA-IDS Refactored

This repository contains the refactored code for the FAIIA-IDS (Focal-Aware Imbalance-Integrated Attention) Intrusion Detection System.

## Directory Structure

- `core/`: Core modules for configuration, data loading, models, losses, training, and evaluation.
  - `config.py`: Configuration parameters.
  - `data_loader.py`: Data loading and preprocessing logic.
  - `model.py`: EDAN v3 and FAIIA component definitions.
  - `ablation.py`: Ablation study specific models (VanillaDNN, EDANv3_Ablation).
  - `loss.py`: Imbalance-Aware Focal Loss implementations.
  - `trainer.py`: Generic training engine.
  - `utils.py`: Evaluation metrics and random seeding.
- `scripts/`: Executable scripts.
  - `train_main.py`: Train the main EDAN v3 model.
  - `run_ablation.py`: Run the 4-experiment ablation study.
- `notebooks/`: Jupyter notebooks (if applicable).

## Setup

1. **Clone the repository.**
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   (Note: `requirements.txt` should contain torch, pandas, numpy, scikit-learn, joblib)

3. **Prepare Dataset:**
   Ensure `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` are in the project root or `/content` (for Colab).

## Usage

### Training Main Model

```bash
python scripts/train_main.py
```

### Running Ablation Study

```bash
python scripts/run_ablation.py
```

## Google Colab

To run this in Google Colab:

1. Clone this repo:
   ```python
   !git clone https://github.com/your-repo/faiia-ids.git
   %cd faiia-ids
   ```
2. Upload the `UNSW_NB15` dataset files to `/content`.
3. Run the scripts as shown above or import modules in a notebook.

```python
from core.model import EDANv3
# ...
```
# FAA-Net
# FAA-Net
Table 1: Overall Metrics (F1, Recall, Precision, Accuracy, AUC) — your main ablation table.
Table 2: Per-attack metrics, split by minority (< 5000 samples) vs majority (≥ 5000 samples).
Figure 1: F1/Recall vs Epoch for Vanilla DNN and FAIIA models.
Figure 2: PR curves per model.
Figure 3: ROC curves per model.
Figure 4: Minority detection comparison (bar plot of Recall per rare attack).
Figure 5: Majority detection comparison (bar plot per common attack).
Figure 6: Convergence plot (Loss vs Epoch).
Figure 7: Ablation comparison (FAIIA w/o prototypes, w/o attention, w/ BCE, w/ Focal).