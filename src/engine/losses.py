import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for addressing severe class imbalance in ECG arrhythmia classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (torch.Tensor, optional): Per-class weighting factors. Shape (num_classes,).
        gamma (float): Focusing parameter for modulating factor (1 - p_t)^gamma. Default: 2.0.
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'.
    """

    def __init__(
        self,
        alpha: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Focal Loss.

        Args:
            inputs (torch.Tensor): Logits of shape (N, C) where C is the number of classes.
            targets (torch.Tensor): Ground truth labels of shape (N,) with values in [0, C-1].

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # pt is the estimated probability for the ground-truth class
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def create_loss_function(
    loss_type: str = "focal",
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory function to dynamically instantiate Focal Loss or Weighted Cross-Entropy Loss.

    Args:
        loss_type (str): 'focal' or 'weighted_ce' / 'cross_entropy'.
        gamma (float): Focusing parameter if loss_type is 'focal'.
        class_weights (torch.Tensor, optional): Class weights tensor for imbalance handling.
        device (torch.device, optional): Target compute device.

    Returns:
        nn.Module: PyTorch loss criterion.
    """
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)

    loss_type_lower = loss_type.lower()
    if "focal" in loss_type_lower:
        criterion = FocalLoss(alpha=class_weights, gamma=gamma)
    elif "weighted" in loss_type_lower or "cross_entropy" in loss_type_lower or "ce" in loss_type_lower:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Expected 'focal' or 'weighted_ce'.")

    return criterion
