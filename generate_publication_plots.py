import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add current working directory
sys.path.append(os.getcwd())

# Style Configuration for IEEE/Springer Publication Quality
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

class_names = ["Normal (N)", "SVEB/A", "VEB/PVC", "Fusion (F)", "Unknown (Q)"]

# Raw Confusion Matrix for Medium Model B (257,541 Params)
cm_raw = np.array([
    [43657,   277,   187,    87,     0],
    [ 1576,   167,    86,     8,     0],
    [   83,    81,  3049,     6,     0],
    [  316,     1,    50,    21,     0],
    [    2,     0,     4,     1,     0]
], dtype=np.float64)

cm_row_norm = cm_raw / (cm_raw.sum(axis=1)[:, np.newaxis] + 1e-12)
cm_col_norm = cm_raw / (cm_raw.sum(axis=0)[np.newaxis, :] + 1e-12)


# =====================================================================
# FIGURE 2A: Row-Normalized Confusion Matrix (Sensitivity / Recall - Row Sum = 100%)
# =====================================================================
def plot_fig2a_row_norm():
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm_row_norm * 100.0, annot=True, fmt=".2f", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Recall / Sensitivity (%) [Row Sum = 100%]'},
                annot_kws={"size": 10, "weight": "bold"})

    plt.title('Figure 2A: Row-Normalized Confusion Matrix (Sensitivity / Recall)\nMedium Model C (257,541 Params | 94.43% Acc)', pad=12, fontweight='bold')
    plt.xlabel('Predicted AAMI Class', fontweight='bold', labelpad=8)
    plt.ylabel('True AAMI Class (Row Sum = 100%)', fontweight='bold', labelpad=8)
    plt.tight_layout()
    save_path = output_dir / "fig2a_row_normalized_confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Generated '{save_path}' (Row Normalized - Recall)")


# =====================================================================
# FIGURE 2B: Column-Normalized Confusion Matrix (Precision / PPV - Col Sum = 100%)
# =====================================================================
def plot_fig2b_col_norm():
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm_col_norm * 100.0, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Precision / PPV (%) [Col Sum = 100%]'},
                annot_kws={"size": 10, "weight": "bold"})

    plt.title('Figure 2B: Column-Normalized Confusion Matrix (Precision / PPV)\nMedium Model C (257,541 Params | 94.43% Acc)', pad=12, fontweight='bold')
    plt.xlabel('Predicted AAMI Class (Column Sum = 100%)', fontweight='bold', labelpad=8)
    plt.ylabel('True AAMI Class', fontweight='bold', labelpad=8)
    plt.tight_layout()
    save_path = output_dir / "fig2b_col_normalized_confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Generated '{save_path}' (Column Normalized - Precision)")


# =====================================================================
# FIGURE 2C: Raw Count Confusion Matrix
# =====================================================================
def plot_fig2c_raw_counts():
    plt.figure(figsize=(8.5, 6.5))
    sns.heatmap(cm_raw, annot=True, fmt=",.0f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Heartbeat Sample Count'},
                annot_kws={"size": 9.5, "weight": "bold"})

    plt.title('Figure 2C: Raw Heartbeat Count Confusion Matrix (DS2 Unseen Test Set: 49,659 Beats)', pad=12, fontweight='bold')
    plt.xlabel('Predicted AAMI Class', fontweight='bold', labelpad=8)
    plt.ylabel('True AAMI Class', fontweight='bold', labelpad=8)
    plt.tight_layout()
    save_path = output_dir / "fig2c_raw_count_confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Generated '{save_path}' (Raw Sample Counts)")


if __name__ == "__main__":
    plot_fig2a_row_norm()
    plot_fig2b_col_norm()
    plot_fig2c_raw_counts()
    print("\n[✓] ALL CONFUSION MATRIX VARIATIONS (ROW-NORM, COL-NORM, RAW COUNTS) GENERATED!")
