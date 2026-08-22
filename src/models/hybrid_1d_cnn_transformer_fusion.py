import torch
import torch.nn as nn
from typing import Union, Tuple
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("hybrid_1d_cnn_transformer_fusion")
@ModelFactory.register("cnn_transformer_fusion_1d")
class Hybrid1DCNNTransformerFusion(BaseECGModel):
    """
    Lightweight 1D CNN-Transformer Encoder with Late RR-Interval Fusion.
    
    Extracts deep spatial-temporal representation (64 dims) from raw 1D ECG waveform
    via 1D Convolutions + Multi-Head Self-Attention Transformer Encoder,
    and fuses it with normalized pre_RR and post_RR interval features (2 dims)
    for late classification, strictly maintaining total trainable parameter count < 70,000.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        cnn_filters: list = [32, 64, 64],
        seq_len: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 64,
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

        # B. Learnable Positional Encoding Buffer
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

        # D. Classification Head with Late RR-Interval Fusion (64 signal dims + 2 RR dims = 66 dims)
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], x_rr: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass supporting single input tensor, tuple input (x_signal, x_rr), or dual parameters.
        """
        if isinstance(x, (tuple, list)):
            x_signal, x_rr = x[0], x[1]
        elif x_rr is None:
            x_signal = x
            # Default zero RR features if not passed directly
            B = x_signal.size(0)
            x_rr = torch.zeros(B, 2, device=x_signal.device, dtype=torch.float32)
        else:
            x_signal = x

        if x_signal.dim() == 4:
            B, C, H, W = x_signal.shape
            x_signal = x_signal.view(B, C, H * W)
        elif x_signal.dim() == 2:
            x_signal = x_signal.unsqueeze(1)

        # A. 1D Convolutions -> (B, 64, L_seq)
        features_1d = self.cnn_1d(x_signal)

        # B. Sequence Permutation -> (B, L_seq, 64)
        seq_input = features_1d.permute(0, 2, 1)
        B, L, D = seq_input.shape

        # Positional slicing
        pos = self.pos_embed[:, :L, :]
        seq_input = seq_input + pos

        # C. Transformer Encoder Processing
        trans_out = self.transformer_encoder(seq_input)

        # D. Global Average Temporal Pooling -> (B, 64)
        pooled_out = torch.mean(trans_out, dim=1)

        # E. Late Fusion with RR-Interval Vector (B, 2) -> (B, 66)
        if x_rr.dim() == 1:
            x_rr = x_rr.unsqueeze(1)
        fused_features = torch.cat([pooled_out, x_rr], dim=1)

        # Output Logits
        logits = self.classifier(fused_features)
        return logits
