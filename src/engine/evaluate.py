import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Any

from src.models.factory import ModelFactory
from src.utils.metrics import compute_metrics, print_metrics_report
from src.engine.trainer import get_device

# Ensure model modules are registered in ModelFactory
import src.models.cnn_model
import src.models.lstm_model
import src.models.gru_model
import src.models.hybrid_cnn_lstm_gru
import src.models.hybrid_1d_cnn_lstm_gru


def evaluate_checkpoint(checkpoint_path: str, test_loader: DataLoader) -> Dict[str, Any]:
    """Loads saved checkpoint, runs test set evaluation, prints paper metrics report and saves confusion matrix plot."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found!")

    device = get_device("auto")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]
    model_name = config["model"]["name"]

    model = ModelFactory.create(model_name, **config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_2d, batch_1d, targets in test_loader:
            batch_2d = batch_2d.to(device)
            outputs = model(batch_2d)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    num_classes = config["data"]["num_classes"]
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)

    class_names = {int(k): v for k, v in config["data"]["classes"].items()}
    print_metrics_report(metrics, class_names)

    # Save Confusion Matrix Heatmap
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    cm_path = results_dir / f"confusion_matrix_{model_name}.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[class_names[i] for i in range(num_classes)],
        yticklabels=[class_names[i] for i in range(num_classes)]
    )
    plt.title(f"Confusion Matrix - {model.get_model_name()}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"[*] Saved confusion matrix plot to '{cm_path}'")
    return metrics
