import torch
import wfdb
import numpy as np
import yaml
from pathlib import Path
from src.models.factory import ModelFactory
import src.models.hybrid_1d_cnn_gru
import src.models.hybrid_cnn_gru
import src.models.hybrid_1d_cnn_lstm_gru
import src.models.hybrid_cnn_lstm_gru
import src.models.cnn_model
import src.models.lstm_model
import src.models.gru_model
from src.data.beat_extractor import AAMI_MAPPING


def evaluate_raw_stream(record_id: str = "100", model_name: str = "hybrid_1d_cnn_gru"):
    """
    Evaluates a pretrained model on a 100% UNPROCESSED, UNFILTERED, UNNORMALIZED
    continuous 30-minute ECG raw stream without R-peak detection or segmentation.
    """
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["data_dir"])
    record_path = data_dir / record_id

    print(f"\n" + "=" * 80)
    print(f"   EVALUATING PRETRAINED MODEL '{model_name.upper()}' ON 100% UNPROCESSED RAW ECG STREAM")
    print(f"   Record: MIT-BIH {record_id} | No Filter | No Normalization | No R-Peak Detection")
    print("=" * 80)

    # 1. Load Raw ECG Signal directly from PhysioNet file
    record = wfdb.rdrecord(str(record_path))
    raw_ecg = record.p_signal[:, 0]  # Raw continuous mV signal (648,000 samples)
    total_samples = len(raw_ecg)
    print(f"[*] Loaded Raw ECG Stream: {total_samples:,} samples ({total_samples / 360 / 60:.1f} minutes)")

    # 2. Load Ground Truth Annotations from .atr file (for accuracy calculation)
    ann = wfdb.rdann(str(record_path), 'atr')
    annotated_r_peaks = set(ann.sample)
    annotated_labels = {idx: AAMI_MAPPING[sym] for idx, sym in zip(ann.sample, ann.symbol) if sym in AAMI_MAPPING}

    # 3. Instantiate Pretrained Model & Load Best Checkpoint
    model = ModelFactory.create(model_name, **config["model"])
    ckpt_path = f"checkpoints/{model_name.replace('_', '')}_best.pth"
    
    if not Path(ckpt_path).exists():
        print(f" [!] Checkpoint file '{ckpt_path}' not found! Please train model first.")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[✓] Loaded Pretrained Model Checkpoint: '{ckpt_path}'")

    # 4. Continuous Sliding Window Inference over RAW signal (Window Size: 256 samples, Step: 128 samples)
    window_size = 256
    step_size = 128

    correct_predictions = 0
    total_evaluated_beats = 0
    class_correct = {c: 0 for c in range(5)}
    class_total = {c: 0 for c in range(5)}

    print(f"[*] Running Continuous Sliding Window Inference over {total_samples:,} raw samples...")

    with torch.no_grad():
        for start_idx in range(0, total_samples - window_size + 1, step_size):
            end_idx = start_idx + window_size
            raw_window = raw_ecg[start_idx:end_idx]  # ZERO PREPROCESSING / ZERO NORMALIZATION

            # Check if an R-peak falls inside this window
            window_r_peaks = [r for r in annotated_r_peaks if start_idx <= r < end_idx]
            if not window_r_peaks:
                continue  # Skip background non-beat windows for fair evaluation

            target_r_peak = window_r_peaks[0]
            true_label = annotated_labels.get(target_r_peak, None)
            if true_label is None:
                continue

            # Feed RAW 1D window directly into PyTorch model
            inp_tensor = torch.tensor(raw_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            if model_name.startswith("hybrid_1d"):
                output = model(inp_tensor)
            else:
                side = int(window_size**0.5)
                matrix_inp = inp_tensor.view(1, 1, side, side)
                output = model(matrix_inp)

            pred_label = torch.argmax(output, dim=1).item()

            total_evaluated_beats += 1
            class_total[true_label] += 1

            if pred_label == true_label:
                correct_predictions += 1
                class_correct[true_label] += 1

    overall_acc = (correct_predictions / total_evaluated_beats * 100) if total_evaluated_beats > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"   RAW CONTINUOUS STREAM INFERENCE RESULTS (Record {record_id})")
    print("=" * 70)
    print(f" Total Evaluated Heartbeat Windows: {total_evaluated_beats:,}")
    print(f" Correct Predictions:               {correct_predictions:,}")
    print(f" Raw Stream Accuracy:               {overall_acc:.2f}%\n")
    print("-----------------------------------------------------------------")
    print("Class Name                          | Correct / Total | Acc (%)")
    print("-----------------------------------------------------------------")
    class_names = ["Normal (N)", "Supraventricular / A-Fib", "PVC (VEB)", "Fusion / VT", "Other / Paced"]
    for c_id in range(5):
        tot = class_total[c_id]
        corr = class_correct[c_id]
        acc = (corr / tot * 100) if tot > 0 else 0.0
        print(f"{class_names[c_id]:<35} | {corr:5d} / {tot:5d}   | {acc:6.2f}%")
    print("=================================================================\n")


if __name__ == "__main__":
    evaluate_raw_stream("100", "hybrid_1d_cnn_gru")
