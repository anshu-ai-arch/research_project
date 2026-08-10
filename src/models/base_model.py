import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseECGModel(nn.Module, ABC):
    """
    Abstract Base Class for all ECG Arrhythmia Classification Models.
    Enforces a strict input/output contract to ensure plug-and-play modularity.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ECG signal inputs.
        Input:
            x: Tensor of shape (batch_size, 1, 40, 32) [2D matrix] or (batch_size, 1, 1280) [1D sequence]
        Output:
            logits: Tensor of shape (batch_size, num_classes=5)
        """
        pass

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_name(self) -> str:
        """Returns the class name of the model."""
        return self.__class__.__name__
