import time
import torch
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data.preprocessor import ECGPreprocessor
from src.data.beat_extractor import ECGBeatExtractor


def run_ds2_benchmark_on_pi4(
    model_path: str = "checkpoints/model_b_medium_torchscript.pt",
    config_path: str = "config.yaml"
):
    print("=================================================================")
    print("     RASPBERRY PI 4 — UNSEEN DS2 TEST SET BENCHMARK ENGINE       ")
    print("=================================================================")
    device = torch.device("cpu")
    print(f"[*] Platform: Physical ARM CPU (Raspberry Pi 4)")
    
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[!] TorchScript model file '{model_file}' not found.")
        return

    print(f"[*] Loading TorchScript model '{model_file}' into Raspberry Pi RAM...")
    start_load = time.time()
    model = torch.jit.load(str(model_file), map_location=device)
    model.eval()
    load_time_ms = (time.time() - start_load) * 1000.0
    print(f"[✓] Model loaded in {load_time_ms:.2f} ms!")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["data_dir"])
    ds2_recs = config["split"]["ds2_test_recordings"]

    preprocessor = ECGPreprocessor(
        raw_fs=config["data"]["raw_sampling_rate"],
        target_fs=config["data"]["target_sampling_rate"],
        lowcut=config["data"]["lowcut"],
        highcut=config["data"]["highcut"],
        filter_order=config["data"]["filter_order"],
        segment_duration_sec=config["data"]["segment_duration_sec"],
        matrix_rows=config["data"]["matrix_rows"],
        matrix_cols=config["data"]["matrix_cols"],
        use_filtering=config["data"].get("use_filtering", True),
    )

    beat_extractor = ECGBeatExtractor(
        beat_window_size=config["data"].get("beat_window_size", 256),
        matrix_rows=config["data"].get("matrix_rows", 16),
        matrix_cols=config["data"].get("matrix_cols", 16)
    )

    all_preds = []
    all_targets = []
    latencies_ms = []

    print(f"[*] Running inference across all 22 unseen DS2 test patient recordings...")
    start_total_bench = time.time()

    for idx, rec_id in enumerate(ds2_recs, 1):
        record_path = data_dir / rec_id
        if not record_path.with_suffix(".dat").exists():
            continue

        import wfdb
        record = wfdb.rdrecord(str(record_path))
        ecg_signal = record.p_signal[:, 0]

        if config["data"].get("use_filtering", True):
            processed_signal = preprocessor.bandpass_filter(ecg_signal)
        else:
            processed_signal = ecg_signal

        resampled_signal = preprocessor.resample_signal(processed_signal)

        b_1d, b_2d, b_rr, b_y = beat_extractor.extract_beats_from_record(
            record_path, resampled_signal, raw_fs=config["data"]["raw_sampling_rate"], target_fs=config["data"]["target_sampling_rate"]
        )

        if len(b_y) == 0:
            continue

        # Min-Max normalize
        min_vals = b_1d.min(axis=1, keepdims=True)
        max_vals = b_1d.max(axis=1, keepdims=True)
        b_1d_norm = (b_1d - min_vals) / (max_vals - min_vals + 1e-8)

        inputs = torch.tensor(b_1d_norm, dtype=torch.float32).unsqueeze(1).to(device)
        targets = b_y

        rec_start_t = time.time()
        with torch.no_grad():
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
        rec_time_ms = (time.time() - rec_start_t) * 1000.0

        per_beat_latency = rec_time_ms / len(b_y)
        latencies_ms.append(per_beat_latency)

        all_preds.extend(preds)
        all_targets.extend(targets)

        rec_acc = accuracy_score(targets, preds) * 100.0
        print(f" Record [{idx:02d}/22] ID {rec_id} | Beats: {len(b_y):4d} | Acc: {rec_acc:6.2f}% | Latency/beat: {per_beat_latency:.2f} ms")

    total_bench_sec = time.time() - start_total_bench
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred) * 100.0
    wf1 = f1_score(y_true, y_pred, average="weighted") * 100.0
    mf1 = f1_score(y_true, y_pred, average="macro") * 100.0

    print("\n=================================================================")
    print("      RASPBERRY PI 4 PHYSICAL BENCHMARK SUMMARY TOTALS           ")
    print("=================================================================")
    print(f" Total DS2 Test Heartbeats Evaluated: {len(y_true):,}")
    print(f" Total Evaluation Time on Pi 4:      {total_bench_sec:.2f} seconds")
    print(f" Mean Inference Latency Per Beat:    {np.mean(latencies_ms):.2f} ms")
    print(f" Physical Pi 4 DS2 Unseen Accuracy:  {acc:.2f}%")
    print(f" Physical Pi 4 Weighted F1-Score:    {wf1:.2f}%")
    print(f" Physical Pi 4 Macro F1-Score:       {mf1:.2f}%")
    print("=================================================================\n")


if __name__ == "__main__":
    run_ds2_benchmark_on_pi4()
