import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import get_dataloaders


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
# 2. LOSS FUNCTIONS (Standard CE, Weighted CE, Focal Loss)
# =====================================================================
class PyTorchFocalLoss(nn.Module):
    """
    PyTorch Focal Loss implementation for severe class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# =====================================================================
# 3. DYNAMIC MODEL C SPECIFICATION (Hybrid1DBiCNNGRU)
# =====================================================================
class DynamicHybrid1DBiCNNGRU(nn.Module):
    """
    Flexible Model C (Hybrid1DBiCNNGRU) supporting dynamic architectural parameters
    for Optuna hyperparameter optimization targeting maximum raw overall accuracy.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_out_channels: int = 64,
        conv_kernel_size: int = 5,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        dropout: float = 0.2,
        num_classes: int = 5
    ):
        super().__init__()

        padding = conv_kernel_size // 2

        # 3-Stage 1D CNN Feature Extractor
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv_out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_out_channels, conv_out_channels * 2, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(conv_out_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(conv_out_channels * 2, conv_out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_out_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        gru_input_dim = conv_out_channels * 2

        # Bidirectional GRU Recurrent Layer
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # 1D CNN Feature Extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Permute to (Batch, Time, Channels) for GRU
        x = x.permute(0, 2, 1)

        # Bi-GRU temporal sequence processing
        gru_out, _ = self.gru(x)

        # Global Average Temporal Pooling
        pooled = torch.mean(gru_out, dim=1)

        # Dense classification
        logits = self.classifier(pooled)
        return logits


# =====================================================================
# 4. OPTUNA OBJECTIVE FUNCTION (PIVOTED TO OVERALL RAW ACCURACY)
# =====================================================================
best_global_accuracy = 0.0

def objective(trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> float:
    global best_global_accuracy

    # -----------------------------------------------------------------
    # A. Broadened Search Space
    # -----------------------------------------------------------------
    conv_out_channels = trial.suggest_categorical('conv_out_channels', [32, 64, 128])
    conv_kernel_size = trial.suggest_categorical('conv_kernel_size', [3, 5, 7, 9])
    gru_hidden_size = trial.suggest_categorical('gru_hidden_size', [64, 128, 256])
    gru_num_layers = trial.suggest_int('gru_num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    loss_type = trial.suggest_categorical('loss_type', ['standard_ce', 'weighted_ce', 'focal'])

    # -----------------------------------------------------------------
    # B. Model Instantiation & Device Placement
    # -----------------------------------------------------------------
    model = DynamicHybrid1DBiCNNGRU(
        in_channels=1,
        conv_out_channels=conv_out_channels,
        conv_kernel_size=conv_kernel_size,
        gru_hidden_size=gru_hidden_size,
        gru_num_layers=gru_num_layers,
        dropout=dropout,
        num_classes=5
    ).to(device)

    # -----------------------------------------------------------------
    # C. Loss Function Setup
    # -----------------------------------------------------------------
    alpha_weights = torch.tensor([1.0, 10.0, 3.0, 10.0, 50.0], dtype=torch.float32).to(device)

    if loss_type == 'standard_ce':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=alpha_weights)
    else:
        focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
        criterion = PyTorchFocalLoss(alpha=alpha_weights, gamma=focal_gamma)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    num_epochs = 15
    best_val_acc = 0.0

    # -----------------------------------------------------------------
    # D. Epoch Training & Validation Loop (Targeting Raw Accuracy)
    # -----------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        # 1. Train Step
        model.train()
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

        # 2. Validation Step (Overall Raw Accuracy Metric)
        model.eval()
        correct = 0
        total = 0

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

                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_acc = (correct / total) * 100.0
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # 3. Checkpoint Saving if Global Best Accuracy Achieved
        if val_acc > best_global_accuracy:
            best_global_accuracy = val_acc
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / "model_c_accuracy_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "hyperparameters": trial.params
                },
                ckpt_path
            )

        # 4. Report to Optuna MedianPruner
        trial.report(val_acc, epoch)

        if trial.should_prune():
            if device.type == "mps":
                torch.mps.empty_cache()
            raise optuna.exceptions.TrialPruned()

    # Memory Cleanup
    del model, optimizer, scheduler, criterion
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return best_val_acc


# =====================================================================
# 5. EVALUATION ON UNSEEN DS2 TEST SET
# =====================================================================
@torch.no_grad()
def evaluate_best_accuracy_model(checkpoint_path: str = "checkpoints/model_c_accuracy_best.pth"):
    device = get_m1_device()
    ckpt_file = Path(checkpoint_path)

    if not ckpt_file.exists():
        print(f"[!] Checkpoint '{checkpoint_path}' not found. Cannot evaluate.")
        return

    print(f"\n[*] Evaluating New Best Accuracy Model from '{ckpt_file}'...")
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    hyperparams = checkpoint.get("hyperparameters", {})

    conv_out = hyperparams.get("conv_out_channels", 64)
    conv_kernel = hyperparams.get("conv_kernel_size", 5)
    gru_hidden = hyperparams.get("gru_hidden_size", 128)
    gru_layers = hyperparams.get("gru_num_layers", 1)
    dropout = hyperparams.get("dropout", 0.2)

    model = DynamicHybrid1DBiCNNGRU(
        in_channels=1,
        conv_out_channels=conv_out,
        conv_kernel_size=conv_kernel,
        gru_hidden_size=gru_hidden,
        gru_num_layers=gru_layers,
        dropout=dropout,
        num_classes=5
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_loader, val_loader, test_loader, config = get_dataloaders("config.yaml")
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
    print("   MODEL C (RAW ACCURACY OPTIMIZED) UNSEEN DS2 TEST PERFORMANCE  ")
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

    # Plot Heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", xticklabels=class_names, yticklabels=class_names, square=True)
    plt.title("Model C (Raw Accuracy Optimized) - DS2 Unseen Test Patient Confusion Matrix", fontsize=12, pad=15)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True AAMI Class", fontsize=11)
    plt.tight_layout()
    
    save_fig_path = "results/confusion_matrix_model_c_accuracy_best.png"
    Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path, dpi=300)
    plt.close()
    print(f"[*] Saved confusion matrix plot to '{save_fig_path}'")


# =====================================================================
# 6. MAIN PIPELINE EXECUTION
# =====================================================================
def run_accuracy_optimization(n_trials: int = 10):
    device = get_m1_device()
    print(f"\n=================================================================")
    print(f"   OPTUNA HYPERPARAMETER TUNING FOR MODEL C (OVERALL RAW ACCURACY)")
    print(f"=================================================================")
    print(f"[*] Target Hardware Device: {device}")
    print(f"[*] Optimization Target Metric: OVERALL RAW ACCURACY (MAXIMIZE)")
    print(f"[*] Target Accuracy Goal: 95.0%+\n")

    train_loader, val_loader, test_loader, config = get_dataloaders("config.yaml")

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        study_name="model_c_raw_accuracy_optimization",
        direction="maximize",
        pruner=pruner
    )

    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\n[✓] Optuna Accuracy Optimization Study Complete!")
    print(f"[*] Best Trial Number: #{study.best_trial.number}")
    print(f"[*] Best Validation Overall Raw Accuracy: {study.best_value:.2f}%")
    print(f"\n[*] Best Hyperparameters Found:")
    for key, val in study.best_trial.params.items():
        print(f"    - {key}: {val}")

    # Evaluate best saved checkpoint on DS2 test set
    evaluate_best_accuracy_model("checkpoints/model_c_accuracy_best.pth")


if __name__ == "__main__":
    run_accuracy_optimization(n_trials=10)
