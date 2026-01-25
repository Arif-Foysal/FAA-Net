"""
Generate FAA-Net Architecture Diagram
Creates a publication-quality diagram for IEEE Access paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up the figure with wide aspect ratio
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

# Define colors
novel_color = '#FFA500'  # Orange
input_color = '#ADD8E6'  # Light blue
standard_color = '#DCDCDC'  # Light gray

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, color, fontsize=10, bold=False):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         edgecolor='black', facecolor=color,
                         linewidth=2, zorder=2)
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center', fontsize=fontsize,
           weight=weight, zorder=3)
    return box

# Helper function for arrows
def create_arrow(ax, x1, y1, x2, y2, style='solid', width=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=width, linestyle=style,
                           color='black', zorder=1)
    ax.add_patch(arrow)

# 1. Input
create_box(ax, 0.5, 2.5, 1.2, 1, 'Network Flow\nFeatures\n(d=33)', input_color, 9)

# 2. Batch Norm
create_box(ax, 2.2, 2.7, 0.8, 0.6, 'Batch\nNorm', standard_color, 8)
create_arrow(ax, 1.7, 3, 2.2, 3)

# 3. Probability Estimator (top)
create_box(ax, 2.5, 4.2, 1.3, 0.8, 'Probability\nEstimator\n2-layer MLP', standard_color, 7)
ax.text(3.1, 3.9, r'$p_{init}$', ha='center', fontsize=8, style='italic')
create_arrow(ax, 2.6, 3.3, 2.8, 4.2)
create_arrow(ax, 3.8, 4.6, 4.5, 4.6, style='dotted', width=1.5)

# 4. FAIIA Module (main component)
faiia_box = FancyBboxPatch((4, 1.2), 5, 4,
                          boxstyle="round,pad=0.1",
                          edgecolor=novel_color, facecolor=novel_color,
                          linewidth=3, alpha=0.3, zorder=0)
ax.add_patch(faiia_box)

# FAIIA title
ax.text(6.5, 5, 'FAIIA: Focal-Aware Imbalance-Integrated Attention',
       ha='center', fontsize=11, weight='bold')
ax.text(6.5, 4.7, '(Novel Contribution)', ha='center', fontsize=8, style='italic', color='gray')

# Multi-head attention
create_box(ax, 4.3, 4.1, 4.4, 0.5, 'Multi-Head Attention (H=4)', 'white', 9)

# Attention mechanism
attn_box = FancyBboxPatch((4.3, 2.3), 4.4, 1.5,
                         boxstyle="round,pad=0.05",
                         edgecolor='gray', facecolor='white',
                         linewidth=1.5, zorder=2)
ax.add_patch(attn_box)

ax.text(5, 3.7, 'Query\n(Q)', ha='center', fontsize=8)
ax.text(7.8, 3.7, 'Minority\nPrototypes\n(K, V)', ha='center', fontsize=8)
ax.text(7.8, 3.3, 'K=8, K-means init', ha='center', fontsize=6, color='gray')
ax.text(6.5, 2.9, r'Attention = softmax$((1 + \alpha \cdot u^\gamma) \cdot Q \cdot K^T) \cdot V$',
       ha='center', fontsize=7)
ax.text(6.5, 2.5, r'$u = 1-2|p_{init}-0.5|$', ha='center', fontsize=7, color='gray')

# Concat & Project
create_box(ax, 4.3, 1.5, 4.4, 0.5, 'Concat & Project', 'white', 9)
ax.text(6.5, 1.2, r'$\alpha$ per head: 0.60-0.78', ha='center', fontsize=6, color='gray')

create_arrow(ax, 3, 3, 4, 3)

# 5. Post-processing
create_box(ax, 9.5, 2.5, 1.5, 1.2, 'Gating &\nRecalibration\nGate + SE\n+ Residual',
          standard_color, 8)
create_arrow(ax, 9, 3.1, 9.5, 3.1)

# 6. Feature Extraction
create_box(ax, 11.5, 2.5, 1.5, 1.2, 'Deep Residual\nNetwork\n3 Blocks:\n[256,128,64]',
          standard_color, 8)
create_arrow(ax, 11, 3.1, 11.5, 3.1)

# 7. Classification
create_box(ax, 13.5, 2.5, 1.5, 1.2, 'Classification\nMLP\nAttack\nProbability',
          input_color, 8)
create_arrow(ax, 13, 3.1, 13.5, 3.1)

# 8. Training (bottom)
training_box = FancyBboxPatch((5, 0.2), 4, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='gray', facecolor='white',
                             linewidth=1.5, linestyle='dashed', zorder=2)
ax.add_patch(training_box)
ax.text(7, 0.5, 'Training: Focal Loss + Label Smoothing',
       ha='center', fontsize=9)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=novel_color, edgecolor='black', label='Novel Component'),
    mpatches.Patch(facecolor=standard_color, edgecolor='black', label='Standard Component')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8, frameon=True)

# Save the figure
plt.tight_layout()
plt.savefig('faanet_architecture.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('faanet_architecture.png', format='png', bbox_inches='tight', dpi=300)
print("✓ Generated: faanet_architecture.pdf")
print("✓ Generated: faanet_architecture.png")
print("\nTo use in your IEEE Access paper, add:")
print("\\begin{figure*}[!t]")
print("\\centering")
print("\\includegraphics[width=0.95\\textwidth]{faanet_architecture.pdf}")
print("\\caption{FAA-Net architecture overview...}")
print("\\label{fig:architecture}")
print("\\end{figure*}")
