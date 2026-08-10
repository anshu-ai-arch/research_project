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
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=32,  # Treat matrix rows as sequence length (40) and cols as feature dim (32)
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
        # Input shape: (B, 1, 40, 32) -> squeeze to (B, 40, 32)
        if x.dim() == 4:
            x = x.squeeze(1)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.view(x.size(0), 40, 32)

        out, (h_n, c_n) = self.lstm(x)
        # Global max pooling over sequence length
        out = torch.max(out, dim=1)[0]
        return self.classifier(out)
