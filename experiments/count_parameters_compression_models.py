import torch
import torch.nn as nn
from pathlib import Path


class GenericHybrid1DBiCNNGRU(nn.Module):
    """
    Generalized Hybrid 1D Bi-CNN-GRU architecture supporting flexible
    CNN channel widths and BiGRU hidden sizes for exact parameter auditing.
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: list = [128, 256, 256],
        kernel_sizes: list = [5, 5, 3],
        gru_hidden_size: int = 64,
        gru_num_layers: int = 2,
        dropout: float = 0.2076,
        num_classes: int = 5
    ):
        super().__init__()

        # Stage 1
        p1 = kernel_sizes[0] // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels[0], kernel_size=kernel_sizes[0], padding=p1),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Stage 2
        p2 = kernel_sizes[1] // 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=kernel_sizes[1], padding=p2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Stage 3
        p3 = kernel_sizes[2] // 2
        self.conv3 = nn.Sequential(
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=kernel_sizes[2], padding=p3),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        gru_input_dim = cnn_channels[2]

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True
        )

        gru_out_dim = gru_hidden_size * 2

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        pooled = torch.mean(gru_out, dim=1)
        logits = self.classifier(pooled)
        return logits


def audit_compression_models():
    configs = {
        "Experiment A (Control - Current)": {
            "cnn_channels": [128, 256, 256],
            "kernel_sizes": [5, 5, 3],
            "gru_hidden_size": 64,
            "gru_num_layers": 2
        },
        "Experiment B (Medium CNN)": {
            "cnn_channels": [64, 128, 128],
            "kernel_sizes": [5, 5, 3],
            "gru_hidden_size": 64,
            "gru_num_layers": 2
        },
        "Experiment C (Small CNN)": {
            "cnn_channels": [32, 64, 64],
            "kernel_sizes": [5, 5, 3],
            "gru_hidden_size": 64,
            "gru_num_layers": 2
        },
        "Experiment D (Medium CNN + BiGRU32)": {
            "cnn_channels": [64, 128, 128],
            "kernel_sizes": [5, 5, 3],
            "gru_hidden_size": 32,
            "gru_num_layers": 2
        }
    }

    print("=========================================================================================")
    print("           MODEL C COMPRESSION EXPERIMENTS — EXACT PARAMETER AUDIT TABLE                 ")
    print("=========================================================================================")
    print(f"{'Experiment Model':<35} | {'CNN Channels':<18} | {'BiGRU Hidden':<12} | {'Trainable Params':>14} | {'Size MB':>8} | {'Red %':>6}")
    print("-" * 105)

    ref_params = None
    results = {}

    for name, cfg in configs.items():
        model = GenericHybrid1DBiCNNGRU(
            cnn_channels=cfg["cnn_channels"],
            kernel_sizes=cfg["kernel_sizes"],
            gru_hidden_size=cfg["gru_hidden_size"],
            gru_num_layers=cfg["gru_num_layers"]
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = (total_params * 4) / (1024.0 * 1024.0)

        if ref_params is None:
            ref_params = trainable_params
            red_pct = 0.0
        else:
            red_pct = ((ref_params - trainable_params) / ref_params) * 100.0

        cnn_str = "->".join(map(str, cfg["cnn_channels"]))
        gru_str = f"{cfg['gru_hidden_size']}x2 ({cfg['gru_num_layers']}L)"

        print(f"{name:<35} | {cnn_str:<18} | {gru_str:<12} | {trainable_params:>14,} | {size_mb:>8.4f} | {red_pct:>5.1f}%")

        results[name] = {
            "cnn_channels": cfg["cnn_channels"],
            "trainable_params": trainable_params,
            "total_params": total_params,
            "size_mb": size_mb,
            "red_pct": red_pct
        }

    print("=========================================================================================\n")
    return results


if __name__ == "__main__":
    audit_compression_models()
