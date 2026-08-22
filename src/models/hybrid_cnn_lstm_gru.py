import torch
import torch.nn as nn
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("hybrid_cnn_lstm_gru")
@ModelFactory.register("hybrid_2d")
class HybridCNNLSTMGRU(BaseECGModel):
    """
    Paper Replication: Hybrid 2D CNN-LSTM-GRU Architecture.
    
    1. 2D CNN layers: Extract spatial/wavefront features from (matrix_rows x matrix_cols) ECG matrix.
    2. Sequence formatting: Reshape feature map into sequence representation (H, C * W).
    3. Sequential LSTM layer: Models long-range temporal dependencies.
    4. Sequential GRU layer: Memory refinement and gating.
    5. Dense Classifier: Fully connected output over 5 arrhythmia classes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        matrix_rows: int = 16,
        matrix_cols: int = 16,
        cnn_filters: list = [32, 64],
        lstm_hidden_size: int = 64,
        gru_hidden_size: int = 32,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.matrix_rows = matrix_rows
        self.matrix_cols = matrix_cols
        self.lstm_hidden_size = lstm_hidden_size
        self.gru_hidden_size = gru_hidden_size

        # 1. Spatial Feature Extractor (2D CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        initial_feature_dim = cnn_filters[1] * matrix_cols  # 64 * 16 = 1024

        # 2. Sequential LSTM Layer
        self.lstm = nn.LSTM(
            input_size=initial_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 3. Sequential GRU Layer
        self.gru = nn.GRU(
            input_size=lstm_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 4. Fully-Connected Softmax Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B = x.size(0)
            side = int(x.size(2)**0.5)
            x = x.view(B, 1, side, side)
        elif x.dim() == 2:
            B = x.size(0)
            side = int(x.size(1)**0.5)
            x = x.view(B, 1, side, side)

        # Step 1: 2D CNN spatial extraction -> (B, 64, H_feat, W_feat)
        cnn_out = self.cnn(x)
        B, C, H, W = cnn_out.shape

        # Step 2: Reshape feature map into sequence (B, H_feat, C * W_feat)
        seq_input = cnn_out.permute(0, 2, 1, 3).contiguous().view(B, H, C * W)

        # Dynamic LSTM input size adaptation if matrix dimensions change
        if self.lstm.input_size != C * W:
            self.lstm = nn.LSTM(
                input_size=C * W,
                hidden_size=self.lstm_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            ).to(x.device)

        # Step 3: LSTM layer processing
        lstm_out, _ = self.lstm(seq_input)

        # Step 4: GRU layer processing
        gru_out, _ = self.gru(lstm_out)

        # Global average pooling across temporal sequence
        pooled_out = torch.mean(gru_out, dim=1)

        # Step 5: Dense FC classification
        logits = self.classifier(pooled_out)
        return logits
