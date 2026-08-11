import torch
import torch.nn as nn
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("hybrid_1d_cnn_lstm_gru")
@ModelFactory.register("hybrid_1d")
class Hybrid1DCNNLSTMGRU(BaseECGModel):
    """
    Novel 1D Continuous Waveform Hybrid Architecture for ECG Arrhythmia Classification.
    
    Processes 1D ECG sequences (1280 samples) directly using 1D convolutions (Conv1d),
    preserving continuous temporal cardiac wave morphology without row-reshaping artifacts.
    
    Pipeline:
    1. 1D CNN Layers: Multi-scale temporal feature extraction over raw 1280-point waveform.
    2. Sequence Permutation: Format (B, Channels, Time) -> (B, Time, Channels) for recurrent processing.
    3. LSTM Layer: Captures long-range temporal dependencies across downsampled feature steps.
    4. GRU Layer: Refines temporal dynamics with parameter efficiency.
    5. Fully-Connected Classifier: Outputs probabilities for 5 arrhythmia classes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        cnn_filters_1d: list = [32, 64, 64],
        kernel_sizes_1d: list = [7, 5, 3],
        lstm_hidden_size: int = 64,
        gru_hidden_size: int = 32,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()

        # 1. 1D Continuous Feature Extractor
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

        # 2. Recurrent LSTM Layer
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 3. Recurrent GRU Layer
        self.gru = nn.GRU(
            input_size=lstm_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 4. Dense Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape handling: Ensure 1D sequence tensor (B, 1, 1280)
        if x.dim() == 4:
            # Flatten 2D (B, 1, 40, 32) into continuous 1D (B, 1, 1280)
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # Step 1: 1D Continuous Waveform Convolutions -> (B, 64, 160)
        features_1d = self.cnn_1d(x)

        # Step 2: Format for Recurrent Processing (B, Channels, Time) -> (B, Time, Channels)
        seq_input = features_1d.permute(0, 2, 1)  # (B, 160, 64)

        # Step 3: LSTM Layer
        lstm_out, _ = self.lstm(seq_input)  # (B, 160, 64)

        # Step 4: GRU Layer
        gru_out, _ = self.gru(lstm_out)  # (B, 160, 32)

        # Global temporal average pooling
        pooled_out = torch.mean(gru_out, dim=1)  # (B, 32)

        # Step 5: Classification
        logits = self.classifier(pooled_out)
        return logits
