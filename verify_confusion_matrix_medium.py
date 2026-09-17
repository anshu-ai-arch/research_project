import torch
import numpy as np
from pathlib import Path
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU
from experiments.evaluate_patient_wise import evaluate_model_patient_wise, get_m1_device

def audit_medium_confusion_matrix():
    device = get_m1_device()
    checkpoint_path = Path("checkpoints/model_c_medium_augmented_best.pth")
    
    if not checkpoint_path.exists():
        print(f"[!] File {checkpoint_path} not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    summary, patient_breakdown = evaluate_model_patient_wise(model)
    
    cm_raw = summary["confusion_matrix"]
    cm_row_norm = cm_raw.astype('float') / (cm_raw.sum(axis=1)[:, np.newaxis] + 1e-12)
    cm_col_norm = cm_raw.astype('float') / (cm_raw.sum(axis=0)[np.newaxis, :] + 1e-12)

    class_names = ["Normal (N)", "SVEB/A", "VEB/PVC", "Fusion (F)", "Unknown (Q)"]

    print("=================================================================")
    print(" 1. RAW UN-NORMALIZED CONFUSION MATRIX (COUNTS)")
    print("=================================================================")
    print(cm_raw)

    print("\n=================================================================")
    print(" 2. ROW-NORMALIZED CONFUSION MATRIX (RECALL / SENSITIVITY - ROW SUM = 100%)")
    print("=================================================================")
    for i, row in enumerate(cm_row_norm):
        print(f"{class_names[i]:<12}: " + "  ".join([f"{v*100.0:6.2f}%" for v in row]) + f"  | Sum = {row.sum()*100.0:.2f}%")

    print("\n=================================================================")
    print(" 3. COLUMN-NORMALIZED CONFUSION MATRIX (PRECISION / PPV - COL SUM = 100%)")
    print("=================================================================")
    for i, row in enumerate(cm_col_norm):
        print(f"{class_names[i]:<12}: " + "  ".join([f"{v*100.0:6.2f}%" for v in row]))
    print("-" * 65)
    col_sums = cm_col_norm.sum(axis=0) * 100.0
    print(f"{'Col Sum':<12}: " + "  ".join([f"{v:6.2f}%" for v in col_sums]))

if __name__ == "__main__":
    audit_medium_confusion_matrix()
