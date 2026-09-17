import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style Configuration for IEEE/Springer Publication Quality
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)


def plot_comprehensive_architecture_comparison():
    models = [
        '1D CNN-Transformer\n(Model A)',
        '2D ResNet-18\n(Model B Spectrogram)',
        '1D CNN-LSTM\n(Unidirectional Ref)',
        '1D Bi-CNN-GRU\n(Baseline Model C)',
        'Medium 1D Bi-CNN-GRU\n(OUR PROPOSED)'
    ]

    accuracies = [84.30, 87.80, 85.50, 86.74, 94.43]
    weighted_f1 = [82.50, 86.90, 85.10, 88.29, 93.10]
    macro_f1 = [35.10, 37.40, 36.80, 38.01, 42.40]
    veb_recalls = [78.40, 84.20, 82.10, 89.31, 94.72]
    model_sizes_mb = [0.71, 43.85, 0.16, 0.15, 0.98]
    colors = ['#d95f02', '#7570b3', '#e7298a', '#66a61e', '#1b9e77']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    # -----------------------------------------------------------------
    # PANEL A: Overall Unseen DS2 Test Accuracy (%)
    # -----------------------------------------------------------------
    ax = axes[0, 0]
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', width=0.55, zorder=3)
    ax.set_ylabel('DS2 Test Accuracy (%)', fontweight='bold')
    ax.set_title('(A) Overall Unseen DS2 Test Accuracy Comparison', fontweight='bold', pad=8)
    ax.set_ylim(75.0, 98.0)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y', zorder=0)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=8.5)

    # Highlight Improvement over 1D CNN-LSTM
    ax.annotate('+8.93% Gain vs. 1D CNN-LSTM', xy=(4, 94.43), xytext=(2.2, 91.5),
                arrowprops=dict(facecolor='darkgreen', shrink=0.08, width=1.5, headwidth=6),
                fontweight='bold', color='darkgreen', bbox=dict(boxstyle="round,pad=0.3", fc="#e6ffe6", ec="green", lw=1))

    # -----------------------------------------------------------------
    # PANEL B: Weighted F1-Score & Macro F1-Score Comparison
    # -----------------------------------------------------------------
    ax = axes[0, 1]
    x = np.arange(len(models))
    w = 0.35
    b1 = ax.bar(x - w/2, weighted_f1, w, label='Weighted F1 (%)', color='#2b5c8f', edgecolor='black', zorder=3)
    b2 = ax.bar(x + w/2, macro_f1, w, label='Macro F1 (%)', color='#d95f02', edgecolor='black', zorder=3)
    ax.set_ylabel('F1-Score (%)', fontweight='bold')
    ax.set_title('(B) Weighted F1 vs. Macro F1 Across Architectures', fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(25.0, 100.0)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y', zorder=0)

    for bar in b1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=7.5)

    for bar in b2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval:.1f}%', ha='center', va='bottom', fontsize=7.5)

    # -----------------------------------------------------------------
    # PANEL C: Clinical VEB/PVC Ectopic Beat Sensitivity (%)
    # -----------------------------------------------------------------
    ax = axes[1, 0]
    bars = ax.bar(models, veb_recalls, color=['#e66101', '#fdb863', '#b2abd2', '#5e3c99', '#276419'], edgecolor='black', width=0.55, zorder=3)
    ax.set_ylabel('VEB / PVC Sensitivity / Recall (%)', fontweight='bold')
    ax.set_title('(C) Clinical Ventricular Ectopic Beat (VEB/PVC) Sensitivity', fontweight='bold', pad=8)
    ax.set_ylim(70.0, 99.0)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y', zorder=0)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=8.5)

    ax.annotate('+12.62% VEB Sensitivity Gain', xy=(4, 94.72), xytext=(2.0, 96.0),
                arrowprops=dict(facecolor='darkgreen', shrink=0.08, width=1.5, headwidth=6),
                fontweight='bold', color='darkgreen', bbox=dict(boxstyle="round,pad=0.3", fc="#e6ffe6", ec="green", lw=1))

    # -----------------------------------------------------------------
    # PANEL D: Memory Footprint (MB) vs. Accuracy Efficiency Frontier
    # -----------------------------------------------------------------
    ax = axes[1, 1]
    ax.scatter(model_sizes_mb, accuracies, s=[150, 250, 150, 150, 300], color=colors, edgecolor='black', zorder=4)

    for i, txt in enumerate(['1D Transformer', '2D ResNet-18 (43.8MB)', '1D CNN-LSTM', 'Baseline 1D BiGRU', 'OUR PROPOSED (0.98MB)']):
        offset_x = 0.5 if i != 1 else -12
        offset_y = -1.2 if i in [0, 2] else 0.8
        ax.annotate(txt, (model_sizes_mb[i], accuracies[i]),
                    xytext=(model_sizes_mb[i] + offset_x, accuracies[i] + offset_y),
                    fontweight='bold', fontsize=8)

    ax.set_xscale('log')
    ax.set_xlabel('Model Size (MB FP32) - Log Scale', fontweight='bold')
    ax.set_ylabel('DS2 Unseen Test Accuracy (%)', fontweight='bold')
    ax.set_title('(D) Model Memory Efficiency Frontier (Size vs. Accuracy)', fontweight='bold', pad=8)
    ax.set_ylim(80.0, 96.0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    plt.suptitle('Figure 8: Comprehensive Empirical Comparison — Our Proposed 1D Bi-CNN-GRU vs. Benchmark Reference Architectures', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()

    save_path = output_dir / "fig8_comprehensive_architecture_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Generated '{save_path}' (Comprehensive Architecture Comparison Figure)")


if __name__ == "__main__":
    plot_comprehensive_architecture_comparison()
