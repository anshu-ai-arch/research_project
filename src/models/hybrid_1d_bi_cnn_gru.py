import torch
import torch.nn as nn
from typing import Optional, Union, List
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("hybrid_1d_bi_cnn_gru")
@ModelFactory.register("bi_cnn_gru_1d")
@ModelFactory.register("model_c")
class Hybrid1DBiCNNGRU(BaseECGModel):
    """
    Model C: 1D Continuous Waveform Hybrid CNN with Bidirectional GRU (Bi-GRU).
    
    Backbone:
      1. 1D Convolutional layer with BatchNorm, ReLU, and MaxPool1d.
      2. Bidirectional GRU (Bi-GRU) capturing forward and backward temporal dependencies.
      3. Global Average Temporal Pooling across the GRU sequence.
      4. Fully connected classification head with Dropout.
    
    Supports dynamic architectural search:
      - conv_out_channels: [16, 32, 64]
      - conv_kernel_size: [3, 5, 7]
      - gru_hidden_size: [32, 64, 128]
      - gru_num_layers: [1, 2]
      - dropout: [0.2, 0.5]
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        conv_out_channels: int = 32,
        conv_kernel_size: int = 5,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 1,
        dropout: float = 0.3,
        cnn_filters_1d: Optional[List[int]] = None,
        kernel_sizes_1d: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.dropout_rate = dropout

        # Backward compatibility for multi-layer list configs if passed
        if cnn_filters_1d is not None and kernel_sizes_1d is not None:
            layers = []
            prev_ch = in_channels
            for out_ch, k_size in zip(cnn_filters_1d, kernel_sizes_1d):
                layers.extend([
                    nn.Conv1d(prev_ch, out_ch, kernel_size=k_size, padding=k_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                ])
                prev_ch = out_ch
            self.conv_block = nn.Sequential(*layers)
            conv_feature_dim = cnn_filters_1d[-1]
        else:
            self.conv_out_channels = conv_out_channels
            self.conv_kernel_size = conv_kernel_size
            self.conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2
                ),
                nn.BatchNorm1d(conv_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            conv_feature_dim = conv_out_channels

        # 2. Bidirectional GRU Layer
        gru_dropout = dropout if gru_num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=conv_feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout
        )

        # 3. Fully Connected Classifier Head
        bi_gru_dim = gru_hidden_size * 2  # 2 directions (Forward + Backward)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(bi_gru_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Model C.
        
        Args:
            x (torch.Tensor): ECG input tensor. Can be (B, 1, L), (B, L), or (B, 1, H, W).
        
        Returns:
            torch.Tensor: Class logits of shape (B, num_classes).
        """
        # Shape adjustment for 1D convolution
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        # 1. 1D Convolutional feature extraction
        conv_features = self.conv_block(x)  # Shape: (B, Channels, Time)

        # 2. Format for GRU: (B, Time, Channels)
        seq_input = conv_features.permute(0, 2, 1)

        # 3. Bidirectional GRU processing
        gru_out, _ = self.gru(seq_input)  # Shape: (B, Time, 2 * hidden_size)

        # 4. Global Average Temporal Pooling over both forward and backward representations
        pooled_out = torch.mean(gru_out, dim=1)  # Shape: (B, 2 * hidden_size)

        # 5. Classification head
        logits = self.classifier(pooled_out)  # Shape: (B, num_classes)
        return logits
