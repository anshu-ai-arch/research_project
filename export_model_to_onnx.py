import torch
import torch.nn as nn
from pathlib import Path
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU


def export_medium_model_to_torchscript(
    checkpoint_path: str = "checkpoints/model_c_medium_augmented_best.pth",
    output_pt_path: str = "checkpoints/model_b_medium_torchscript.pt"
):
    print("=================================================================")
    print("      EXPORTING MEDIUM MODEL B TO TORCHSCRIPT FOR RASPBERRY PI    ")
    print("=================================================================")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[!] Checkpoint file '{ckpt_file}' not found.")
        return

    print(f"[*] Loading PyTorch checkpoint from '{ckpt_file}'...")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 1, 256, dtype=torch.float32)

    output_path = Path(output_pt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Tracing model to TorchScript format at '{output_path}'...")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(str(output_path))

    print(f"[✓] Successfully exported TorchScript model for Raspberry Pi!")
    print(f"    - TorchScript Location: {output_path}")
    print(f"    - File Size on Disk:   {output_path.stat().st_size / 1024.0:.2f} KB ({output_path.stat().st_size / (1024*1024):.4f} MB)")


if __name__ == "__main__":
    export_medium_model_to_torchscript()
