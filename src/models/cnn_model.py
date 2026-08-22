import torch
import torch.nn as nn
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("cnn")
class BaselineCNN(BaseECGModel):
    """
    Baseline 2D Convolutional Neural Network for ECG matrix inputs.
    Uses AdaptiveAvgPool2d to dynamically process 16x16 beat matrices or 40x32 segment matrices.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        matrix_rows: int = 16,
        matrix_cols: int = 16,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((4, 4))  # Output always (64, 4, 4)
        )

        flattened_dim = 64 * 4 * 4  # 1024

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B = x.size(0)
            side = int(x.size(2)**0.5)
            x = x.view(B, 1, side, side)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)
