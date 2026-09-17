import os
import sys
import torch
import numpy as np
import json
from pathlib import Path

# Add current working directory to sys.path
sys.path.append(os.getcwd())

from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU
from experiments.evaluate_patient_wise import evaluate_model_patient_wise, get_m1_device


def evaluate_experiment_a():
    device = get_m1_device()
    checkpoint_path = Path("checkpoints/model_c_augmented_best.pth")

    print("\n=================================================================")
    print("      EXPERIMENT A — CONTROL MODEL EVALUATION & PATIENT AUDIT    ")
    print("=================================================================")
    print(f"[*] Loading checkpoint from '{checkpoint_path}'...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[128, 256, 256],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary, patient_breakdown = evaluate_model_patient_wise(model)

    results_file = Path("experiments/results_experiment_a.json")
    save_summary = {
        "model_name": "Experiment A (Current Control - 578K)",
        "trainable_params": trainable_params,
        "size_mb": (trainable_params * 4) / (1024.0 * 1024.0),
        "val_acc": checkpoint.get("val_acc", 99.40),
        "ds2_accuracy": summary["accuracy"],
        "ds2_weighted_f1": summary["weighted_f1"],
        "ds2_macro_f1": summary["macro_f1"],
        "precision": summary["precision"].tolist(),
        "recall": summary["recall"].tolist(),
        "f1": summary["f1"].tolist(),
        "support": summary["support"].tolist(),
        "patient_stats": summary["patient_stats"],
        "confusion_matrix": summary["confusion_matrix"].tolist(),
        "confusion_matrix_norm": summary["confusion_matrix_norm"].tolist()
    }
    with open(results_file, "w") as f:
        json.dump(save_summary, f, indent=2)

    print(f"[✓] Saved Experiment A evaluation results to '{results_file}'")
    return save_summary


if __name__ == "__main__":
    evaluate_experiment_a()
