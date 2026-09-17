import torch
import torch.nn as nn
from pathlib import Path
from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU

def verify_and_audit_weights(
    checkpoint_path: str = "checkpoints/model_c_medium_augmented_best.pth"
):
    print("=================================================================")
    print("      STEP 1: AUDITING & VERIFYING TRAINED MODEL WEIGHTS         ")
    print("=================================================================")
    
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[!] Error: Checkpoint file '{ckpt_file}' not found.")
        return False

    print(f"[*] Loading state dict from '{ckpt_file}'...")
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

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # Parameter count and memory audit
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weight_size_bytes = total_params * 4  # FP32
    weight_size_mb = weight_size_bytes / (1024 * 1024)

    print("\n[✓] State Dictionary Verified Successfully!")
    print(f"    - Total Parameters:     {total_params:,}")
    print(f"    - Trainable Parameters: {trainable_params:,}")
    print(f"    - FP32 Weight Memory:   {weight_size_mb:.4f} MB ({weight_size_bytes / 1024:.2f} KB)")
    print(f"    - Target Checkpoint:    {ckpt_file}")
    
    # Run test forward pass
    dummy_input = torch.randn(2, 1, 256)
    with torch.no_grad():
        out = model(dummy_input)
    
    print(f"    - Dummy Forward Pass Shape: {out.shape} (Expected: [2, 5])")
    print("=================================================================\n")
    return True

if __name__ == "__main__":
    verify_and_audit_weights()
