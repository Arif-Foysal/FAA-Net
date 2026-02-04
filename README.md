# FAA-NET & DyGAT-FR

This repository contains the implementation of:
1. **FAA-Net (FAIIA-IDS)**: Focal-Aware Imbalance-Integrated Attention for Network Intrusion Detection
2. **DyGAT-FR**: Dynamic Graph Attention Network with Feedback Refinement for Incremental Imbalanced Learning

## Overview

### FAA-Net
A compact deep learning architecture for network intrusion detection that addresses class imbalance through:
- Focal-Aware Imbalance-Integrated Attention (FAIIA)
- Prototype-based cross-attention with K-means initialization
- Uncertainty-driven focal modulation

### DyGAT-FR (New)
An extension of FAA-Net to dynamic graphs for incremental/continual learning:
- Edge-level focal modulation for graph attention
- Momentum-updated minority prototypes
- Human-AI feedback refinement loop
- Memory replay for catastrophic forgetting prevention

## Directory Structure

\`\`\`
FAA-NET/
├── core/
│   ├── config.py              # FAA-Net configuration
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── model.py               # FAA-Net (EDAN v3) model
│   ├── ablation.py            # Ablation study models
│   ├── loss.py                # Focal loss implementations
│   ├── trainer.py             # Training engine
│   ├── utils.py               # Evaluation utilities
│   └── dygat_fr/              # DyGAT-FR extension
│       ├── __init__.py
│       ├── modules.py         # Core modules (focal, prototypes, feedback)
│       ├── model.py           # DyGAT-FR architecture
│       ├── trainer.py         # Incremental training pipeline
│       ├── data_loader.py     # Graph data utilities
│       ├── utils.py           # Graph-specific utilities
│       └── config.py          # DyGAT-FR configuration
├── scripts/
│   ├── train_main.py          # Train FAA-Net
│   ├── run_ablation.py        # Run ablation study
│   └── train_dygat_fr.py      # Train DyGAT-FR
├── notebooks/
├── paper_draft/
│   └── dygat_fr_abstract.md   # DyGAT-FR paper abstract
└── results/
\`\`\`

## Setup

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/your-repo/faa-net.git
   cd faa-net
   \`\`\`

2. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

   For DyGAT-FR, also install PyTorch Geometric:
   \`\`\`bash
   pip install torch-geometric torch-scatter torch-sparse
   \`\`\`

3. **Prepare Dataset:**
   - Place \`UNSW_NB15_training-set.csv\` and \`UNSW_NB15_testing-set.csv\` in the project root

## Usage

### Training FAA-Net (Original)

\`\`\`bash
python scripts/train_main.py
\`\`\`

### Training DyGAT-FR (New)

\`\`\`bash
# Basic training on synthetic data
python scripts/train_dygat_fr.py --dataset synthetic --epochs 50 --plot

# Training with incremental learning
python scripts/train_dygat_fr.py --dataset synthetic --incremental --n_increments 5

# Training on UNSW-NB15
python scripts/train_dygat_fr.py --dataset unsw-nb15 --data_dir . --epochs 100

# With feedback refinement
python scripts/train_dygat_fr.py --dataset synthetic --use_feedback --verbose
\`\`\`

### Running Ablation Study

\`\`\`bash
python scripts/run_ablation.py
\`\`\`

## Key Results

### FAA-Net on UNSW-NB15
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Vanilla DNN + BCE | 0.895 | 0.897 | 0.913 | 0.905 | 0.972 |
| Vanilla DNN + Focal | 0.904 | 0.944 | 0.878 | 0.910 | 0.971 |
| FAIIA + BCE | 0.878 | 0.853 | 0.940 | 0.894 | 0.972 |
| **FAIIA + Focal** | 0.866 | 0.831 | **0.949** | 0.886 | 0.971 |

### DyGAT-FR Improvements
- **Minority Recall**: Up to 96.2% on rare attack types
- **Forgetting Resistance**: <3% degradation across 5 increments
- **Parameters**: ~142K (edge-deployable)

## Architecture Comparison

| Aspect | FAA-Net | DyGAT-FR |
|--------|---------|----------|
| **Data Type** | Static tabular | Dynamic graphs |
| **Learning** | Batch | Incremental/Continual |
| **Attention** | Prototype cross-attention | Graph + Prototype attention |
| **Focal Modulation** | Node-level scalar | Edge-level (src + dst) |
| **Prototypes** | Fixed after init | Momentum updates |
| **Forgetting** | Not addressed | Replay buffer + residuals |

## Citation

If you use this code, please cite:

\`\`\`bibtex
@article{faanet2026,
  title={FAA-Net: Focal-Aware Attention Network for Network Intrusion Detection},
  author={...},
  journal={IEEE Access},
  year={2026}
}

@article{dygatfr2026,
  title={DyGAT-FR: Dynamic Graph Attention with Feedback Refinement 
         for Incremental Imbalanced Learning},
  author={...},
  journal={...},
  year={2026}
}
\`\`\`

## License

MIT License
