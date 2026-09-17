import torch
import torch.nn as nn
from pathlib import Path


class ECGTransferModel(nn.Module):
    """
    Transfer Learning PyTorch Architecture for ECG Arrhythmia Classification.
    Loads a pre-trained 1D CNN feature backbone, supports optional backbone gradient freezing,
    and attaches a customizable classification head for target datasets/tasks.
    """

    def __init__(
        self,
        backbone_path: str = "checkpoints/backbone_1d_cnn_pretrained.pth",
        num_target_classes: int = 5,
        freeze_backbone: bool = True,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 2,
        dropout: float = 0.2076
    ):
        super().__init__()

        self.freeze_backbone = freeze_backbone
        self.num_target_classes = num_target_classes

        # 1. Reconstruct 3-Stage 1D CNN Feature Backbone
        # Stage 1 (1 -> 64)
        conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Stage 2 (64 -> 128)
        conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Stage 3 (128 -> 128)
        conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.backbone = nn.Sequential(conv1, conv2, conv3)

        # Load pre-trained backbone weights if available
        backbone_file = Path(backbone_path)
        if backbone_file.exists():
            ckpt = torch.load(backbone_file, map_location="cpu", weights_only=False)
            backbone_state = ckpt.get('backbone_state_dict', ckpt)
            self.backbone.load_state_dict(backbone_state)
            print(f"[✓] ECGTransferModel: Successfully loaded pre-trained backbone from '{backbone_file}'!")
        else:
            print(f"[!] Warning: Backbone checkpoint '{backbone_file}' not found. Initializing randomly.")

        # Optionally freeze backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[✓] ECGTransferModel: Backbone parameters FROZEN (gradient updates locked).")
        else:
            print("[✓] ECGTransferModel: Backbone parameters TRAINABLE (end-to-end fine-tuning mode).")

        # 2. Recurrent BiGRU Temporal Sequence Encoder
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 3. New Customizable Target Task Classification Head
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_target_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 1, 256]
        features = self.backbone(x)  # [B, 128, 32]
        
        # Permute for GRU: [B, 32, 128]
        features_seq = features.permute(0, 2, 1)
        
        gru_out, _ = self.gru(features_seq)  # [B, 32, 128]
        
        # Global temporal average pooling
        pooled = torch.mean(gru_out, dim=1)  # [B, 128]
        
        logits = self.fc(pooled)  # [B, num_target_classes]
        return logits
