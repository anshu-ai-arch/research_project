import torch
import torch.nn as nn
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("hybrid_1d_cnn_gru")
@ModelFactory.register("cnn_gru_1d")
class Hybrid1DCNNGRU(BaseECGModel):
    """
    Simplified 1D Continuous Waveform Hybrid CNN-GRU Architecture.
    
    Operates directly on continuous 1D ECG sequences (1280 points) using 1D convolutions
    and routes temporal features directly into GRU (bypassing LSTM), optimizing speed and parameters.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        cnn_filters_1d: list = [32, 64, 64],
        kernel_sizes_1d: list = [7, 5, 3],
        gru_hidden_size: int = 32,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()

        # 1. 1D Continuous Waveform Feature Extractor
        self.cnn_1d = nn.Sequential(
            nn.Conv1d(in_channels, cnn_filters_1d[0], kernel_size=kernel_sizes_1d[0], padding=kernel_sizes_1d[0] // 2),
            nn.BatchNorm1d(cnn_filters_1d[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # -> (32, 640)

            nn.Conv1d(cnn_filters_1d[0], cnn_filters_1d[1], kernel_size=kernel_sizes_1d[1], padding=kernel_sizes_1d[1] // 2),
            nn.BatchNorm1d(cnn_filters_1d[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # -> (64, 320)

            nn.Conv1d(cnn_filters_1d[1], cnn_filters_1d[2], kernel_size=kernel_sizes_1d[2], padding=kernel_sizes_1d[2] // 2),
            nn.BatchNorm1d(cnn_filters_1d[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # -> (64, 160)
        )

        feature_dim = cnn_filters_1d[2]  # 64

        # 2. Direct GRU Layer (bypassing LSTM)
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 3. Dense Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # 1D Continuous Convolutions -> (B, 64, 160)
        features_1d = self.cnn_1d(x)

        # Format for GRU: (B, Time=160, Channels=64)
        seq_input = features_1d.permute(0, 2, 1)

        # Direct GRU layer -> (B, 160, 32)
        gru_out, _ = self.gru(seq_input)

        # Global average pooling
        pooled_out = torch.mean(gru_out, dim=1)

        # Classification
        logits = self.classifier(pooled_out)
        return logits
