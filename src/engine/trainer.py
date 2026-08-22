import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader

from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory
from src.utils.metrics import compute_metrics, print_metrics_report


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance.
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


def get_device(device_str: str = "auto") -> torch.device:
    """Selects target computing device (MPS for Apple Silicon, CUDA, or CPU)."""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"[*] Using PyTorch target device: {device}")
    return device


class Trainer:
    """
    Unified Training Engine for training any registered BaseECGModel architecture.
    Handles device placement, forward/backward passes, metric calculation, and best checkpoint saving.
    """

    def __init__(
        self,
        model: BaseECGModel,
        config: Dict[str, Any],
        device: torch.device = None
    ):
        self.model = model
        self.config = config
        self.device = device if device else get_device(config["training"].get("device", "auto"))
        self.model.to(self.device)

        self.epochs = config["training"]["epochs"]
        self.lr = float(config["training"]["learning_rate"])
        self.weight_decay = float(config["training"].get("weight_decay", 1.0e-4))

        self.checkpoint_dir = Path(config["training"].get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        use_focal = config["training"].get("use_focal_loss", True)
        if config["training"].get("use_class_weights", True):
            weights = torch.tensor([1.0, 10.0, 3.0, 10.0, 50.0], dtype=torch.float32).to(self.device)
            if use_focal:
                gamma = float(config["training"].get("focal_gamma", 2.0))
                self.criterion = FocalLoss(alpha=weights, gamma=gamma)
                print(f"[*] Applied Focal Loss (gamma={gamma}) with Alpha Weights: {weights.cpu().tolist()}")
            else:
                self.criterion = nn.CrossEntropyLoss(weight=weights)
                print(f"[*] Applied Inverse Class Weights to CrossEntropyLoss: {weights.cpu().tolist()}")
        else:
            if use_focal:
                gamma = float(config["training"].get("focal_gamma", 2.0))
                self.criterion = FocalLoss(gamma=gamma)
                print(f"[*] Applied Focal Loss (gamma={gamma}) without Alpha Weights.")
            else:
                self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            if len(batch) == 4:
                batch_2d, batch_1d, batch_rr, targets = batch
                batch_2d = batch_2d.to(self.device)
                batch_1d = batch_1d.to(self.device)
                batch_rr = batch_rr.to(self.device)
                targets = targets.to(self.device)

                if "fusion" in self.model.get_model_name().lower():
                    outputs = self.model(batch_1d, batch_rr)
                else:
                    outputs = self.model(batch_2d)
            else:
                batch_2d, batch_1d, targets = batch
                batch_2d = batch_2d.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(batch_2d)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, Dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in data_loader:
            if len(batch) == 4:
                batch_2d, batch_1d, batch_rr, targets = batch
                batch_2d = batch_2d.to(self.device)
                batch_1d = batch_1d.to(self.device)
                batch_rr = batch_rr.to(self.device)
                targets = targets.to(self.device)

                if "fusion" in self.model.get_model_name().lower():
                    outputs = self.model(batch_1d, batch_rr)
                else:
                    outputs = self.model(batch_2d)
            else:
                batch_2d, batch_1d, targets = batch
                batch_2d = batch_2d.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(batch_2d)

            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        total = len(all_targets)
        eval_loss = total_loss / total

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        metrics = compute_metrics(y_true, y_pred, num_classes=self.config["data"]["num_classes"])
        return eval_loss, metrics["accuracy"], metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> str:
        model_name = self.model.get_model_name()
        print(f"\n[*] Starting training for model '{model_name}' ({self.epochs} epochs)...")
        print(f"[*] Total Trainable Parameters: {self.model.get_num_parameters():,}\n")

        best_val_acc = 0.0
        best_ckpt_path = self.checkpoint_dir / f"{model_name.lower()}_best.pth"

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_metrics = self.evaluate(val_loader)

            self.scheduler.step(val_loss)

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "config": self.config
                    },
                    best_ckpt_path
                )
                print(f" -> Best checkpoint saved to '{best_ckpt_path}' (Val Acc: {val_acc:.2f}%)")

        print(f"\n[✓] Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
        return str(best_ckpt_path)
