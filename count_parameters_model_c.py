import torch
import torch.nn as nn
from pathlib import Path
from optuna_tune_accuracy_model_c import DynamicHybrid1DBiCNNGRU


def count_parameters_model_c(checkpoint_path: str = "checkpoints/model_c_augmented_best.pth"):
    print("=================================================================")
    print("      MODEL C (Hybrid1DBiCNNGRU) EXACT PARAMETER AUDIT           ")
    print("=================================================================")

    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        print(f"[*] Loading checkpoint configuration from '{ckpt_file}'...")
        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        hyperparams = checkpoint.get("hyperparameters", {})
        conv_out = hyperparams.get("conv_out_channels", 128)
        conv_kernel = hyperparams.get("conv_kernel_size", 5)
        gru_hidden = hyperparams.get("gru_hidden_size", 64)
        gru_layers = hyperparams.get("gru_num_layers", 2)
        dropout = hyperparams.get("dropout", 0.2076)
    else:
        print("[!] Checkpoint not found. Using default Trial #7 hyperparameters...")
        conv_out = 128
        conv_kernel = 5
        gru_hidden = 64
        gru_layers = 2
        dropout = 0.2076

    model = DynamicHybrid1DBiCNNGRU(
        in_channels=1,
        conv_out_channels=conv_out,
        conv_kernel_size=conv_kernel,
        gru_hidden_size=gru_hidden,
        gru_num_layers=gru_layers,
        dropout=dropout,
        num_classes=5
    )

    if ckpt_file.exists():
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # Formatted Header
    print(f"\n{'Layer (Module Name)':<35} | {'Param Tensor Shape':<22} | {'Param Count':>12}")
    print("-" * 75)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        param_count = param.numel()
        shape_str = str(list(param.shape))
        requires_grad = param.requires_grad

        total_params += param_count
        if requires_grad:
            trainable_params += param_count

        print(f"{name:<35} | {shape_str:<22} | {param_count:>12,}")

    print("=" * 75)
    
    # Calculate Memory Footprint (FP32 = 4 bytes per parameter)
    model_size_bytes = total_params * 4
    model_size_kb = model_size_bytes / 1024.0
    model_size_mb = model_size_kb / 1024.0

    print("\n=================================================================")
    print("                     GRAND SUMMARY TOTALS                        ")
    print("=================================================================")
    print(f" Total Network Parameters:     {total_params:,}")
    print(f" Total Trainable Parameters:   {trainable_params:,}")
    print(f" Total Non-Trainable Params:   {total_params - trainable_params:,}")
    print(f" Estimated Weights Size (FP32): {model_size_kb:.2f} KB ({model_size_mb:.4f} MB)")
    print("=================================================================\n")

    # Pass a dummy tensor through to verify forward pass output
    dummy_input = torch.randn(1, 1, 256)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"[*] Verified Single Sample Inference Forward Pass:")
    print(f"    - Input Waveform Shape:  {list(dummy_input.shape)}")
    print(f"    - Output Logits Shape:   {list(output.shape)}")


if __name__ == "__main__":
    count_parameters_model_c("checkpoints/model_c_augmented_best.pth")
