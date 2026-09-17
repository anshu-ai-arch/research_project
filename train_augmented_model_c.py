import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import get_dataloaders
from optuna_tune_accuracy_model_c import DynamicHybrid1DBiCNNGRU


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
# 2. ONLINE ECG DATA AUGMENTATION DATASET WRAPPER
# =====================================================================
class AugmentedECGDataset(Dataset):
    """
    Real-time online data augmentation dataset wrapper for 1D ECG signals.
    Applies amplitude scaling, noise injection, and temporal shifting on DS1 train set.
    """

    def __init__(self, base_dataset: Dataset, is_train: bool = True):
        self.base_dataset = base_dataset
        self.is_train = is_train

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        batch_item = self.base_dataset[idx]

        if len(batch_item) == 4:
            batch_2d, batch_1d, batch_rr, target = batch_item
        else:
            batch_2d, batch_1d, target = batch_item
            batch_rr = None

        if torch.is_tensor(batch_1d):
            x_1d = batch_1d.clone()
        else:
            x_1d = torch.tensor(batch_1d, dtype=torch.float32)

        if self.is_train:
            # 1. Random Amplitude Scaling: Simulates electrode impedance variation [0.85, 1.15]
            scale_factor = torch.FloatTensor(1).uniform_(0.85, 1.15).item()
            x_1d = x_1d * scale_factor

            # 2. Gaussian Noise / Baseline Wander: Simulates muscle artifacts & noise N(0, sigma)
            sigma = torch.FloatTensor(1).uniform_(0.005, 0.02).item()
            noise = torch.randn_like(x_1d) * sigma
            x_1d = x_1d + noise

            # 3. Random Time Shifting: [-5, +5] samples alignment tolerance
            shift = torch.randint(-5, 6, (1,)).item()
            if shift != 0:
                x_1d = torch.roll(x_1d, shifts=shift, dims=-1)

        if batch_rr is not None:
            return batch_2d, x_1d, batch_rr, target
        return batch_2d, x_1d, target


# =====================================================================
# 3. TRAINING & EVALUATION PIPELINE
# =====================================================================
def train_and_evaluate_augmented_model_c(num_epochs: int = 20):
    device = get_m1_device()
    print("\n=================================================================")
    print("   MODEL C (Hybrid1DBiCNNGRU) ONLINE DATA AUGMENTATION TRAINING   ")
    print("=================================================================")
    print(f"[*] Target Acceleration Device: {device}")
    print(f"[*] Baseline Hyperparameters: Trial #7 (conv_out=128, gru_hidden=64, layers=2)")
    print(f"[*] Online Augmentations: Random Scaling [0.85, 1.15], Noise N(0, 0.02), Time Shift [-5, +5]\n")

    # Load Base Dataloaders
    base_train_loader, val_loader, test_loader, config = get_dataloaders("config.yaml")

    # Wrap Train Set with Online Augmentation
    augmented_train_dataset = AugmentedECGDataset(base_train_loader.dataset, is_train=True)
    train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=base_train_loader.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Instantiate Model C with Optuna Trial #7 Best Hyperparameters
    model = DynamicHybrid1DBiCNNGRU(
        in_channels=1,
        conv_out_channels=128,
        conv_kernel_size=5,
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002018, weight_decay=2.35e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0
    checkpoint_path = Path("checkpoints/model_c_augmented_best.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("[*] Starting Training on Augmented DS1 Training Set...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if len(batch) == 4:
                batch_2d, batch_1d, batch_rr, targets = batch
            else:
                batch_2d, batch_1d, targets = batch

            batch_1d = batch_1d.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_1d)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Validation Step on Un-augmented Validation Set
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    batch_2d, batch_1d, batch_rr, targets = batch
                else:
                    batch_2d, batch_1d, targets = batch

                batch_1d = batch_1d.to(device)
                targets = targets.to(device)

                outputs = model(batch_1d)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = (val_correct / val_total) * 100.0
        scheduler.step(val_acc)

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "hyperparameters": {
                        "conv_out_channels": 128,
                        "conv_kernel_size": 5,
                        "gru_hidden_size": 64,
                        "gru_num_layers": 2,
                        "dropout": 0.2076
                    }
                },
                checkpoint_path
            )

    print(f"\n[✓] Augmented Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[*] Checkpoint saved to '{checkpoint_path}'")

    # Evaluate Best Augmented Model on Unseen DS2 Test Set
    evaluate_augmented_model(checkpoint_path)


# =====================================================================
# 4. UNSEEN DS2 TEST SET EVALUATION ENGINE
# =====================================================================
@torch.no_grad()
def evaluate_augmented_model(checkpoint_path: Path):
    device = get_m1_device()
    print(f"\n[*] Evaluating Augmented Model C on Unseen DS2 Test Patients...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hyperparams = checkpoint.get("hyperparameters", {})

    model = DynamicHybrid1DBiCNNGRU(
        in_channels=1,
        conv_out_channels=hyperparams.get("conv_out_channels", 128),
        conv_kernel_size=hyperparams.get("conv_kernel_size", 5),
        gru_hidden_size=hyperparams.get("gru_hidden_size", 64),
        gru_num_layers=hyperparams.get("gru_num_layers", 2),
        dropout=hyperparams.get("dropout", 0.2076),
        num_classes=5
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader, config = get_dataloaders("config.yaml")
    class_names = ["Normal (N)", "SVEB/A", "VEB/PVC", "Fusion (F)", "Unknown (Q)"]

    all_preds = []
    all_targets = []

    print(f"[*] Running inference on {len(test_loader.dataset):,} unseen DS2 test heartbeats...")
    for batch in test_loader:
        if len(batch) == 4:
            batch_2d, batch_1d, batch_rr, targets = batch
        else:
            batch_2d, batch_1d, targets = batch

        batch_1d = batch_1d.to(device)
        outputs = model(batch_1d)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    accuracy = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") * 100.0
    weighted_f1 = f1_score(y_true, y_pred, average="weighted") * 100.0

    print("\n=================================================================")
    print("   AUGMENTED MODEL C (Hybrid1DBiCNNGRU) UNSEEN DS2 TEST RESULTS   ")
    print("=================================================================")
    print(f" Total Test Heartbeats Evaluated: {len(y_true):,}")
    print(f" Overall Raw Accuracy:            {accuracy:.2f}%")
    print(f" Weighted F1-Score:               {weighted_f1:.2f}%")
    print(f" Macro F1-Score:                  {macro_f1:.2f}%")
    print("-----------------------------------------------------------------")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print("\nDetailed Per-Class Performance Report:\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    print("\n=================================================================")
    print("        NORMALIZED CONFUSION MATRIX (DS2 UNSEEN TEST SET)        ")
    print("=================================================================")
    header = f"{'True \\ Pred':<12} | " + " | ".join([f"{name[:7]:>7}" for name in class_names])
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(cm_norm):
        row_str = " | ".join([f"{val * 100.0:6.2f}%" for val in row])
        print(f"{class_names[idx][:12]:<12} | {row_str}")
    print("=================================================================\n")

    # Plot & Save Confusion Matrix
    save_fig_path = "results/confusion_matrix_model_c_augmented_eval.png"
    Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Purples", xticklabels=class_names, yticklabels=class_names, square=True)
    plt.title("Model C (Augmented Training) - DS2 Unseen Test Patient Confusion Matrix", fontsize=12, pad=15)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True AAMI Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    plt.close()
    print(f"[*] Saved confusion matrix plot to '{save_fig_path}'")

    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    train_and_evaluate_augmented_model_c(num_epochs=20)
