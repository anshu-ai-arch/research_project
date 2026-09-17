import torch
import torch.nn as nn
from pathlib import Path
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU

def export_pretrained_backbone(
    checkpoint_path: str = "checkpoints/model_c_medium_augmented_best.pth",
    output_backbone_path: str = "checkpoints/backbone_1d_cnn_pretrained.pth"
):
    print("=================================================================")
    print("   STEP 2: EXPORTING PRE-TRAINED 1D CNN FEATURE BACKBONE WEIGHTS ")
    print("=================================================================")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[!] Error: Source checkpoint '{ckpt_file}' not found.")
        return False

    print(f"[*] Loading full model from '{ckpt_file}'...")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    full_model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    )
    full_model.load_state_dict(state_dict)

    # Extract 1D CNN stages: conv1, conv2, conv3
    cnn_backbone = nn.Sequential(
        full_model.conv1,
        full_model.conv2,
        full_model.conv3
    )
    
    backbone_state_dict = cnn_backbone.state_dict()

    out_file = Path(output_backbone_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'backbone_state_dict': backbone_state_dict,
        'cnn_channels': [64, 128, 128],
        'kernel_sizes': [5, 5, 3],
        'output_feature_dim': 128,
        'source_checkpoint': str(ckpt_file)
    }, out_file)

    backbone_params = sum(p.numel() for p in cnn_backbone.parameters())
    backbone_size_kb = (backbone_params * 4) / 1024.0

    print("\n[✓] Pre-trained Feature Backbone Exported Successfully!")
    print(f"    - Output Location:     {out_file}")
    print(f"    - Backbone Parameters: {backbone_params:,}")
    print(f"    - Backbone Memory Size: {backbone_size_kb:.2f} KB ({backbone_size_kb/1024.0:.4f} MB)")
    print("=================================================================\n")
    return True

if __name__ == "__main__":
    export_pretrained_backbone()
