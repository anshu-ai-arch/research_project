import torch
import torch.nn as nn
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("cnn")
class BaselineCNN(BaseECGModel):
    """
    Baseline 2D Convolutional Neural Network for 40x32 ECG input matrix.
    Extracts spatial features using Conv2D, ReLU, MaxPool2D, and FC layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        matrix_rows: int = 40,
        matrix_cols: int = 32,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (32, 20, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (64, 10, 8)
        )

        flattened_dim = 64 * (matrix_rows // 4) * (matrix_cols // 4)  # 64 * 10 * 8 = 5120

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is 1D sequence (B, 1, 1280), reshape to 2D (B, 1, 40, 32)
        if x.dim() == 3:
            x = x.view(x.size(0), 1, 40, 32)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)
