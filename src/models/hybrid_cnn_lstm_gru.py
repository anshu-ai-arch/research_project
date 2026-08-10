import torch
import torch.nn as nn
from src.models.base_model import BaseECGModel
from src.models.factory import ModelFactory


@ModelFactory.register("hybrid_cnn_lstm_gru")
class HybridCNNLSTMGRU(BaseECGModel):
    """
    Proposed Hybrid CNN-LSTM-GRU model architecture for ECG Arrhythmia Classification
    as specified in Section 3.3 of the research paper.
    
    Pipeline:
    1. 2D CNN layers: Extract spatial features from (40 x 32) ECG matrix.
    2. Sequence formatting: Reshape feature map into sequence representation.
    3. LSTM layer: Captures long-term temporal relationships.
    4. GRU layer: Enhances temporal dynamics modeling efficiently.
    5. Fully-Connected Softmax Classifier: Outputs probabilities for 5 arrhythmia classes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        matrix_rows: int = 40,
        matrix_cols: int = 32,
        cnn_filters: list = [32, 64],
        lstm_hidden_size: int = 64,
        gru_hidden_size: int = 32,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.matrix_rows = matrix_rows
        self.matrix_cols = matrix_cols

        # 1. Spatial Feature Extractor (2D CNN)
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

        # Sequence dimension after CNN feature extraction
        self.seq_len = matrix_rows // 4  # 10 sequence steps
        self.feature_dim = cnn_filters[1] * matrix_cols  # 64 * 32 = 2048

        # 2. Long Short-Term Memory (LSTM) layer
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 3. Gated Recurrent Unit (GRU) layer
        self.gru = nn.GRU(
            input_size=lstm_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 4. Fully-Connected Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape handling: (B, 1, 40, 32)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Step 1: 2D CNN spatial extraction
        cnn_out = self.cnn(x)  # (B, 64, 10, 32)

        # Step 2: Reshape feature map for sequential recurrent processing
        # Permute to (B, 10 [seq_len], 64, 32) -> Flatten channels & cols -> (B, 10, 2048)
        B, C, H, W = cnn_out.shape
        seq_input = cnn_out.permute(0, 2, 1, 3).contiguous().view(B, H, C * W)

        # Step 3: LSTM layer processing
        lstm_out, _ = self.lstm(seq_input)  # (B, 10, lstm_hidden_size)

        # Step 4: GRU layer processing
        gru_out, _ = self.gru(lstm_out)  # (B, 10, gru_hidden_size)

        # Global average pooling across temporal sequence
        pooled_out = torch.mean(gru_out, dim=1)  # (B, gru_hidden_size)

        # Step 5: Fully-connected output classification
        logits = self.classifier(pooled_out)
        return logits
