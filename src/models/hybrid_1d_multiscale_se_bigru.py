import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


class SqueezeAndExcitation1D(nn.Module):
    """
    1D Squeeze-and-Excitation (SE) Channel Attention Module.
    AdaptiveAvgPool1d -> Linear(channels, reduction) -> ReLU -> Linear(reduction, channels) -> Sigmoid
    """
    def __init__(self, channels: int = 48, reduction_ratio: int = 8):
        super().__init__()
        reduced_channels = max(channels // reduction_ratio, 6)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


@ModelFactory.register("hybrid_1d_multiscale_se_bigru")
@ModelFactory.register("multiscale_se_bigru_1d")
class Hybrid1DMultiScaleSEBiGRU(BaseECGModel):
    """
    Final Architectural Innovation: Multi-Scale Inception 1D Convolutions + SE Channel Attention + Bi-GRU.
    
    1. Multi-Scale 1D Inception Feature Extractor:
       - Kernel 3 (Fine-grained QRS details)
       - Kernel 7 (Medium QRS/P-wave dynamics)
       - Kernel 15 (Wide ST-T wave repolarization trends)
    2. Squeeze-and-Excitation (SE) Channel Attention Block: Focuses on critical ECG channels.
    3. Secondary 1D Convolution: Refines feature maps.
    4. Bidirectional GRU Layer (hidden_size=20, 2 directions = 40 features).
    5. Dense Classifier: Outputs logits over 5 AAMI classes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        gru_hidden_size: int = 20,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()

        # A. Multi-Scale Inception 1D Convolutions
        self.branch1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.branch3 = nn.Conv1d(in_channels, 16, kernel_size=15, padding=7)

        self.bn_inception = nn.BatchNorm1d(48)
        self.pool_inception = nn.MaxPool1d(kernel_size=2, stride=2)

        # B. Squeeze-and-Excitation Channel Attention
        self.se_block = SqueezeAndExcitation1D(channels=48, reduction_ratio=8)

        # C. Secondary 1D Convolution
        self.conv2 = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # D. Bidirectional GRU Recurrent Layer (20 hidden units * 2 directions = 40 features)
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # E. Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # A. Multi-Scale Inception Branch Concatenation -> (B, 48, L)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        inception_out = torch.cat([x1, x2, x3], dim=1)
        
        inception_out = self.pool_inception(F.relu(self.bn_inception(inception_out)))

        # B. Squeeze-and-Excitation Channel Attention
        se_out = self.se_block(inception_out)

        # C. Secondary 1D Convolution -> (B, 64, L/4)
        conv2_out = self.conv2(se_out)

        # D. Format for Bidirectional GRU: (B, Time, Channels)
        seq_input = conv2_out.permute(0, 2, 1)
        gru_out, _ = self.gru(seq_input)

        # Global Average Temporal Pooling across time sequence
        pooled_out = torch.mean(gru_out, dim=1)

        # E. Dense Classifier
        logits = self.classifier(pooled_out)
        return logits
