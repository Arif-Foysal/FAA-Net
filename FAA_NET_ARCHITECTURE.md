# FAA-Net: Deep Architectural Analysis

FAA-Net (Focal-Aware Attention Network) is a multi-stage deep learning architecture specifically engineered for tabular Network Intrusion Detection (NIDS). It addresses the "majority-class bias" inherent in intrusion datasets through an integrated focal-aware mechanism.

## 1. Probabilistic Preliminary Path ($p_{init}$)
FAA-Net utilizes a dual-path input strategy. While the main network processes features, an **Auxiliary Estimator** (MLP) produces a prior probability $p_{init} = \sigma(W_{aux}x + b_{aux})$.

*   **Mathematical Rationale**: In massive datasets like UNSW-NB15, most samples are trivially normal. By estimating $p_{init}$ early, the model assigns a "difficulty weight" to each sample.
*   **Decoupled Intelligence**: This allows the main FAIIA module to treat "easy" and "hard" samples differently by looking at the distance from the decision boundary: $1 - |p_{init} - 0.5| \times 2$.

---

## 2. FAIIA: Mathematical Formulation
The core contribution is the **Focal-Aware Imbalance-Integrated Attention** module.

### A. Prototype-Based Cross-Attention
Unlike self-attention, which scales as $O(L^2)$ or $O(d^2)$, FAIIA uses **learnable minority prototypes** $\{\mu_k\}_{k=1}^K$ (where $K=8$).
*   **Initialization**: Prototypes are initialized via K-Means on a subset of known minority class samples during pre-training.
*   **Comparison**: $d(x, \mu_k) = \frac{Q(x) \cdot K(\mu_k)^T}{\sqrt{d_{head}}}$. This creates a similarity map between the current flow and "gold standard" attack patterns.

### B. Focal Modulation Factor ($\mathcal{F}$)
The attention scores $s$ are modulated by the estimator's uncertainty:
$$\mathcal{F}(p_{init}) = \alpha \cdot (1 - |p_{init} - 0.5| \cdot 2 + \epsilon)^\gamma$$
*   **Alpha ($\alpha=0.6$)**: The boost magnitude.
*   **Gamma ($\gamma=2.0$)**: The focus parameter.
*   **Result**: If $p_{init} \approx 0.5$ (maximum uncertainty), $\mathcal{F}$ is maximized, "shouting" to the network that this sample needs deep scrutiny.

### C. Class-Conditional Gating (CCG)
The output of the attention heads is passed through a **Class-Conditional Gate**:
$$Gate(x, p) = \sigma(W_{gate} \cdot [x, \text{Diff}(p)])$$
This allows the model to "bypass" the complex attention output for samples it is already very confident about, preserving the integrity of the original feature distribution and reducing false positives on normal traffic.

---

## 3. High-Depth Feature Refinement
### Squeeze-and-Excitation (SE) Interplay
Following FAIIA, an **SE Block** recalibrates the 33-dimensional input based on the new attention-weighted context. By using a reduction ratio of 4, it finds dependencies between features that standard MLP layers often miss.

### Stacked Residual Hierarchy [256, 128, 64]
The deep extraction path uses a funnel-like structure to distill high-dimensional network flow patterns:
1.  **Block 256**: High-capacity feature expansion.
2.  **Block 128**: Latent representation refinement.
3.  **Block 64**: Final feature abstraction for classification.
Each block uses **GELU (Gaussian Error Linear Unit)** which, unlike ReLU, has a non-zero gradient for small negative values, preventing "neuron death" in the sparse feature spaces common in tabular IDS data.

---

## 4. Implementation & Efficiency Specs
*   **Input Space**: 33 selected features (reduced from 42).
*   **Parameter Count**: ~142,500 parameters (Standard variant).
*   **Inference Latency**: Optimized for CPU-only edge deployment (~2-5ms per batch).
*   **Training Objective**: Integrated Focal Loss $\mathcal{L} = -\alpha(1-\hat{y})^\gamma \log(\hat{y})$ to mirror the architectural focus.

### 💡 Design Rationale: Why Cross-Attention?
Standard self-attention in tabular data is often noisy because features (like `duration` vs `protocol`) have no inherent spatial relationship. By using **Prototype Cross-Attention**, we fix the "anchor points" to known attack categories, making the attention mechanism much more stable and interpretable.
