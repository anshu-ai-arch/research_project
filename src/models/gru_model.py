import torch
import torch.nn as nn
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("gru")
class BaselineGRU(BaseECGModel):
    """
    Baseline GRU architecture for ECG sequence classification.
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
        self.gru = nn.GRU(
            input_size=32,  # 40 steps of dimension 32
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
            x = x.squeeze(1)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.view(x.size(0), 40, 32)

        out, h_n = self.gru(x)
        out = torch.max(out, dim=1)[0]
        return self.classifier(out)
