import torch
import torch.nn as nn
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("hybrid_cnn_gru")
@ModelFactory.register("cnn_gru")
class HybridCNNGRU(BaseECGModel):
    """
    Simplified 2D Hybrid CNN-GRU Model Architecture.
    
    Eliminates the intermediate LSTM layer from the standard paper model (CNN -> GRU -> FC),
    reducing recurrent parameters and FLOPs by ~57% while maintaining temporal gating capacity.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        matrix_rows: int = 40,
        matrix_cols: int = 32,
        cnn_filters: list = [32, 64],
        gru_hidden_size: int = 32,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.matrix_rows = matrix_rows
        self.matrix_cols = matrix_cols

        # 1. 2D CNN Spatial Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> (32, 20, 32)

            nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> (64, 10, 32)
        )

        self.seq_len = matrix_rows // 4  # 10 sequence steps
        self.feature_dim = cnn_filters[1] * matrix_cols  # 64 * 32 = 2048

        # 2. Direct GRU Layer (bypassing LSTM)
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 3. Fully-Connected Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 2D CNN spatial extraction
        cnn_out = self.cnn(x)  # (B, 64, 10, 32)

        # Reshape to sequence: (B, 10, 2048)
        B, C, H, W = cnn_out.shape
        seq_input = cnn_out.permute(0, 2, 1, 3).contiguous().view(B, H, C * W)

        # Direct GRU layer
        gru_out, _ = self.gru(seq_input)  # (B, 10, gru_hidden_size)

        # Global average pooling across temporal sequence
        pooled_out = torch.mean(gru_out, dim=1)  # (B, gru_hidden_size)

        # Classification
        logits = self.classifier(pooled_out)
        return logits
