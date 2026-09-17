import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from src.data.dataset import get_dataloaders
from src.models.factory import ModelFactory
from src.models.hybrid_1d_bi_cnn_gru import Hybrid1DBiCNNGRU
from optuna_tune_model_c import DynamicHybrid1DBiCNNGRU


# =====================================================================
# 1. HARDWARE SELECTION (Apple Mac M1 Acceleration)
# =====================================================================
def get_m1_device() -> torch.device:
    """Selects Apple Silicon MPS acceleration if available, fallback to CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[*] Target Hardware Device: {device}")
    return device


# =====================================================================
# 2. CONFUSION MATRIX TEXT & GRAPHICAL RENDERING
# =====================================================================
def print_text_confusion_matrix(cm_normalized: np.ndarray, class_names: list):
    """Prints a clean, formatted text confusion matrix to stdout."""
    print("\n=================================================================")
    print("        NORMALIZED CONFUSION MATRIX (DS2 UNSEEN TEST SET)        ")
    print("=================================================================")
    header = f"{'True \\ Pred':<12} | " + " | ".join([f"{name[:7]:>7}" for name in class_names])
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(cm_normalized):
        row_str = " | ".join([f"{val * 100.0:6.2f}%" for val in row])
        print(f"{class_names[idx][:12]:<12} | {row_str}")
    print("=================================================================\n")


def plot_and_save_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """Generates and saves a high-resolution confusion matrix heatmap."""
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title("Model C (Hybrid1DBiCNNGRU) - DS2 Unseen Test Patient Confusion Matrix", fontsize=12, pad=15)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True AAMI Class", fontsize=11)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Saved confusion matrix plot to '{save_path}'")


# =====================================================================
# 3. EVALUATION ENGINE
# =====================================================================
@torch.no_grad()
def evaluate_model_c(checkpoint_path: str = "checkpoints/model_c_optuna_best.pth"):
    """
    Loads Model C checkpoint and evaluates on unseen DS2 test set (49,659 heartbeats).
    """
    device = get_m1_device()
    ckpt_file = Path(checkpoint_path)

    if not ckpt_file.exists():
        fallback_path = Path("checkpoints/hybrid1dbicnngru_best.pth")
        if fallback_path.exists():
            print(f"[!] Checkpoint '{checkpoint_path}' not found. Loading fallback: '{fallback_path}'")
            ckpt_file = fallback_path
        else:
            raise FileNotFoundError(f"No valid checkpoint found at '{checkpoint_path}' or '{fallback_path}'")

    print(f"[*] Loading Model C checkpoint from '{ckpt_file}'...")
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    raw_state_dict = checkpoint["model_state_dict"]

    if any("cnn_1d" in k for k in raw_state_dict.keys()):
        state_dict = {}
        for k, v in raw_state_dict.items():
            state_dict[k.replace("cnn_1d.", "conv_block.")] = v

        gru_hidden = state_dict["gru.weight_ih_l0"].shape[0] // 3
        model = Hybrid1DBiCNNGRU(
            in_channels=1,
            num_classes=5,
            cnn_filters_1d=[32, 64, 64],
            kernel_sizes_1d=[7, 5, 3],
            gru_hidden_size=gru_hidden,
            bidirectional=True
        ).to(device)

        if "classifier.3.weight" in state_dict:
            model.classifier = nn.Sequential(
                nn.Linear(state_dict["classifier.0.weight"].shape[1], state_dict["classifier.0.weight"].shape[0]),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(state_dict["classifier.3.weight"].shape[1], state_dict["classifier.3.weight"].shape[0])
            ).to(device)

    elif "conv1.0.weight" in raw_state_dict:
        state_dict = raw_state_dict
        conv1_out = state_dict["conv1.0.weight"].shape[0]
        conv_kernel = state_dict["conv1.0.weight"].shape[2]
        gru_hidden = state_dict["gru.weight_ih_l0"].shape[0] // 3
        gru_layers = 2 if "gru.weight_ih_l1" in state_dict else 1

        model = DynamicHybrid1DBiCNNGRU(
            in_channels=1,
            conv_out_channels=conv1_out,
            conv_kernel_size=conv_kernel,
            gru_hidden_size=gru_hidden,
            gru_num_layers=gru_layers,
            dropout=0.3,
            num_classes=5
        ).to(device)
    else:
        state_dict = raw_state_dict
        model_cfg = checkpoint.get("config", {}).get("model", {})
        model_name = model_cfg.get("name", "hybrid_1d_bi_cnn_gru")
        model = ModelFactory.create(model_name, **model_cfg).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # Load Data
    _, _, test_loader, cfg = get_dataloaders("config.yaml")
    class_names = [
        "Normal (N)",
        "SVEB/A",
        "VEB/PVC",
        "Fusion (F)",
        "Unknown (Q)"
    ]

    all_preds = []
    all_targets = []

    print(f"[*] Running inference on {len(test_loader.dataset):,} unseen DS2 test heartbeats...")
    for batch in test_loader:
        if len(batch) == 4:
            batch_2d, batch_1d, batch_rr, targets = batch
        else:
            batch_2d, batch_1d, targets = batch

        input_tensor = batch_1d if batch_1d.dim() == 3 else batch_2d
        input_tensor = input_tensor.to(device)

        outputs = model(input_tensor)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Compute Metrics
    accuracy = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") * 100.0
    weighted_f1 = f1_score(y_true, y_pred, average="weighted") * 100.0

    print("\n=================================================================")
    print("      MODEL C (Hybrid1DBiCNNGRU) UNSEEN DS2 TEST PERFORMANCE     ")
    print("=================================================================")
    print(f" Total Test Heartbeats Evaluated: {len(y_true):,}")
    print(f" Overall Raw Accuracy:            {accuracy:.2f}%")
    print(f" Macro F1-Score:                  {macro_f1:.2f}%")
    print(f" Weighted F1-Score:               {weighted_f1:.2f}%")
    print("-----------------------------------------------------------------")

    # Detailed Classification Report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print("\nDetailed Per-Class Performance Report:\n")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    print_text_confusion_matrix(cm_norm, class_names)
    plot_and_save_confusion_matrix(cm, class_names, "results/confusion_matrix_model_c_optuna_eval.png")

    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    evaluate_model_c("checkpoints/model_c_optuna_best.pth")
