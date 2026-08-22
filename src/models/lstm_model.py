import torch
import torch.nn as nn
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("lstm")
class BaselineLSTM(BaseECGModel):
    """
    Baseline LSTM architecture for ECG sequence classification.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        matrix_cols: int = 16,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.matrix_cols = matrix_cols
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.proj = nn.Linear(matrix_cols, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, H, W)
        elif x.dim() == 3 and x.size(1) == 1:
            B, _, L = x.shape
            side = int(L**0.5)
            x = x.view(B, side, side)
        elif x.dim() == 2:
            B, L = x.shape
            side = int(L**0.5)
            x = x.view(B, side, side)

        in_dim = x.size(-1)
        if self.proj.in_features != in_dim:
            self.proj = nn.Linear(in_dim, self.hidden_size).to(x.device)

        seq = self.proj(x)
        out, _ = self.lstm(seq)
        out = torch.max(out, dim=1)[0]
        return self.classifier(out)
