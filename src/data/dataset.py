import os
import wfdb
import torch
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split

from src.data.preprocessor import ECGPreprocessor
from src.data.beat_extractor import ECGBeatExtractor


class ArrhythmiaDataset(Dataset):
    """
    PyTorch Dataset for ECG Arrhythmia Classification with RR-Interval Fusion support.
    Yields (matrix_2d, segment_1d, rr_features, label).
    """

    def __init__(self, matrices_2d: np.ndarray, segments_1d: np.ndarray, rr_features: np.ndarray, labels: np.ndarray):
        self.matrices_2d = torch.tensor(matrices_2d, dtype=torch.float32).unsqueeze(1)
        self.segments_1d = torch.tensor(segments_1d, dtype=torch.float32).unsqueeze(1)
        self.rr_features = torch.tensor(rr_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.matrices_2d[idx], self.segments_1d[idx], self.rr_features[idx], self.labels[idx]


def build_dataset_from_records(
    config_path: str = "config.yaml"
) -> Tuple[ArrhythmiaDataset, ArrhythmiaDataset, ArrhythmiaDataset, Dict]:
    """
    Loads WFDB recordings, processes signal through ECGPreprocessor and ECGBeatExtractor,
    and constructs Train, Validation, and Test datasets using AAMI Inter-Patient Partitioning (DS1 vs DS2).
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["data_dir"])
    dataset_mode = config["data"].get("dataset_mode", "beat")
    split_mode = config["split"].get("split_mode", "inter_patient")
    use_filtering = config["data"].get("use_filtering", True)

    preprocessor = ECGPreprocessor(
        raw_fs=config["data"]["raw_sampling_rate"],
        target_fs=config["data"]["target_sampling_rate"],
        lowcut=config["data"]["lowcut"],
        highcut=config["data"]["highcut"],
        filter_order=config["data"]["filter_order"],
        segment_duration_sec=config["data"]["segment_duration_sec"],
        matrix_rows=config["data"]["matrix_rows"],
        matrix_cols=config["data"]["matrix_cols"],
        use_filtering=use_filtering,
    )

    beat_extractor = ECGBeatExtractor(
        beat_window_size=config["data"].get("beat_window_size", 256),
        matrix_rows=config["data"].get("matrix_rows", 16),
        matrix_cols=config["data"].get("matrix_cols", 16)
    )

    ds1_recs = config["split"]["ds1_train_recordings"]
    ds2_recs = config["split"]["ds2_test_recordings"]

    print(f"[*] Processing MIT-BIH Records | Mode: '{dataset_mode.upper()}' | Filtering: {use_filtering} | Split: '{split_mode.upper()}'...")

    def load_records_data(rec_list: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_1d, all_2d, all_rr, all_y = [], [], [], []

        for rec_id in rec_list:
            record_path = data_dir / rec_id
            if not record_path.with_suffix(".dat").exists() or not record_path.with_suffix(".atr").exists():
                print(f" [!] Record {rec_id} missing .dat or .atr file at '{record_path}'. Skipping.")
                continue

            record = wfdb.rdrecord(str(record_path))
            ecg_signal = record.p_signal[:, 0]  # Channel 0 (MLII)

            if dataset_mode == "beat":
                # Step 1: Bandpass Filter (if enabled)
                if use_filtering:
                    processed_signal = preprocessor.bandpass_filter(ecg_signal)
                else:
                    processed_signal = ecg_signal

                # Step 2: Resample signal if target_fs != raw_fs
                resampled_signal = preprocessor.resample_signal(processed_signal)

                # Step 3: Extract R-peak beat windows, RR intervals & ground-truth AAMI labels
                b_1d, b_2d, b_rr, b_y = beat_extractor.extract_beats_from_record(
                    record_path, resampled_signal, raw_fs=config["data"]["raw_sampling_rate"], target_fs=config["data"]["target_sampling_rate"]
                )

                if len(b_y) > 0:
                    all_1d.append(b_1d)
                    all_2d.append(b_2d)
                    all_rr.append(b_rr)
                    all_y.append(b_y)
            else:
                # Segment mode
                s_1d, s_2d = preprocessor.process_raw_record(ecg_signal)
                if len(s_1d) > 0:
                    all_1d.append(s_1d)
                    all_2d.append(s_2d)
                    all_rr.append(np.zeros((len(s_1d), 2), dtype=np.float32))
                    all_y.append(np.zeros(len(s_1d), dtype=np.int64))

        if len(all_y) == 0:
            raise RuntimeError("No beats extracted! Ensure PhysioNet records (.dat & .atr) are downloaded.")

        X_1d = np.concatenate(all_1d, axis=0)
        X_2d = np.concatenate(all_2d, axis=0)
        X_rr = np.concatenate(all_rr, axis=0)
        y = np.concatenate(all_y, axis=0)

        return X_1d, X_2d, X_rr, y

    # Load DS1 (Train/Val Patients) and DS2 (Unseen Test Patients)
    print(" -> Processing DS1 Training Patient Recordings...")
    ds1_1d, ds1_2d, ds1_rr, ds1_y = load_records_data(ds1_recs)

    print(" -> Processing DS2 Unseen Test Patient Recordings...")
    ds2_1d, ds2_2d, ds2_rr, ds2_y = load_records_data(ds2_recs)

    # Split DS1 into Train (80%) and Validation (20%)
    X_train_2d, X_val_2d, X_train_1d, X_val_1d, X_train_rr, X_val_rr, y_train, y_val = train_test_split(
        ds1_2d, ds1_1d, ds1_rr, ds1_y, test_size=config["split"].get("val_ratio", 0.20), stratify=ds1_y, random_state=42
    )

    train_ds = ArrhythmiaDataset(X_train_2d, X_train_1d, X_train_rr, y_train)
    val_ds = ArrhythmiaDataset(X_val_2d, X_val_1d, X_val_rr, y_val)
    test_ds = ArrhythmiaDataset(ds2_2d, ds2_1d, ds2_rr, ds2_y)

    print(f"\n[✓] AAMI Dataset Built Successfully (with RR-Interval Features):")
    print(f" -> Train Set (DS1 Patients): {len(train_ds):,} heartbeats")
    print(f" -> Val Set   (DS1 Patients): {len(val_ds):,} heartbeats")
    print(f" -> Test Set  (DS2 Unseen Patients): {len(test_ds):,} heartbeats\n")

    return train_ds, val_ds, test_ds, config


from torch.utils.data import WeightedRandomSampler


def get_dataloaders(config_path: str = "config.yaml") -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    train_ds, val_ds, test_ds, config = build_dataset_from_records(config_path)
    batch_size = config["training"]["batch_size"]

    if config["training"].get("use_oversampling", True):
        # Calculate per-sample weights based on class weights [1.0, 10.0, 3.0, 10.0, 50.0]
        class_weights = np.array([1.0, 10.0, 3.0, 10.0, 50.0], dtype=np.float32)
        sample_weights = class_weights[train_ds.labels.numpy()]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        print("[*] WeightedRandomSampler Enabled for Training DataLoader.")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, config
