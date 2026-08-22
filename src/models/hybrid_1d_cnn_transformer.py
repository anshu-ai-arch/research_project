import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("hybrid_1d_cnn_transformer")
@ModelFactory.register("cnn_transformer_1d")
class Hybrid1DCNNTransformer(BaseECGModel):
    """
    Lightweight 1D CNN-Transformer Encoder Hybrid Architecture.
    
    Combines 1D convolutions for spatial feature extraction with a 1-layer Multi-Head
    Self-Attention Transformer Encoder to model temporal dependencies across ECG sequence timesteps,
    strictly maintaining total trainable parameter count < 50,000.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        cnn_filters: list = [32, 48, 48],
        d_model: int = 48,
        nhead: int = 4,
        dim_feedforward: int = 48,
        dropout: float = 0.2,
        max_len: int = 256,
        **kwargs
    ):
        super().__init__()

        # A. 1D CNN Feature Extractor
        self.cnn_1d = nn.Sequential(
            nn.Conv1d(in_channels, cnn_filters[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_filters[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # B. Learnable Positional Encoding Buffer/Parameter
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # C. Transformer Encoder Block (Single-Layer Multi-Head Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # D. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # A. 1D Convolutions -> (B, 48, L_seq)
        features_1d = self.cnn_1d(x)

        # B. Sequence Permutation -> (B, L_seq, 48)
        seq_input = features_1d.permute(0, 2, 1)
        B, L, D = seq_input.shape

        # Dynamic positional slicing up to current sequence length L
        pos = self.pos_embed[:, :L, :]
        seq_input = seq_input + pos

        # C. Transformer Encoder Processing
        trans_out = self.transformer_encoder(seq_input)

        # D. Global Average Temporal Pooling across timesteps
        pooled_out = torch.mean(trans_out, dim=1)

        # Output Logits
        logits = self.classifier(pooled_out)
        return logits
