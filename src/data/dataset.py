import os
import wfdb
import torch
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
from src.data.preprocessor import ECGPreprocessor


class ArrhythmiaDataset(Dataset):
    """
    PyTorch Dataset for ECG Arrhythmia Classification.
    Supports both 2D Matrix format (40x32) and 1D Sequence format (1280,).
    """

    def __init__(self, matrices_2d: np.ndarray, segments_1d: np.ndarray, labels: np.ndarray):
        self.matrices_2d = torch.tensor(matrices_2d, dtype=torch.float32).unsqueeze(1)  # (N, 1, 40, 32)
        self.segments_1d = torch.tensor(segments_1d, dtype=torch.float32).unsqueeze(1)  # (N, 1, 1280)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (2d_matrix, 1d_segment, label)."""
        return self.matrices_2d[idx], self.segments_1d[idx], self.labels[idx]


def build_dataset_from_records(
    config_path: str = "config.yaml"
) -> Tuple[ArrhythmiaDataset, ArrhythmiaDataset, ArrhythmiaDataset, Dict]:
    """
    Loads raw WFDB recordings, processes them through ECGPreprocessor,
    assigns class labels, and splits them into Train, Validation, and Test Datasets.
    Supports both 'segment' (stratified random segment split) and 'recording' (patient-level split).
    """
    from sklearn.model_selection import train_test_split

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["data_dir"])
    preprocessor = ECGPreprocessor(
        raw_fs=config["data"]["raw_sampling_rate"],
        target_fs=config["data"]["target_sampling_rate"],
        lowcut=config["data"]["lowcut"],
        highcut=config["data"]["highcut"],
        filter_order=config["data"]["filter_order"],
        segment_duration_sec=config["data"]["segment_duration_sec"],
        matrix_rows=config["data"]["matrix_rows"],
        matrix_cols=config["data"]["matrix_cols"],
    )

    split_mode = config["split"].get("split_mode", "segment")

    # Map recordings to target class ID based on paper specifications
    rec_to_class = {}
    for class_id, (class_name, rec_list) in enumerate(config["data"]["recordings"].items()):
        for rec in rec_list:
            rec_to_class[rec] = class_id

    print(f"[*] Processing MIT-BIH recordings (Split Mode: '{split_mode}')...")

    if split_mode == "recording":
        train_recs = set(config["split"]["train_recordings"])
        val_recs = set(config["split"]["val_recordings"])
        test_recs = set(config["split"]["test_recordings"])

        data_splits = {
            "train": {"1d": [], "2d": [], "labels": []},
            "val": {"1d": [], "2d": [], "labels": []},
            "test": {"1d": [], "2d": [], "labels": []},
        }

        for rec_id, class_id in rec_to_class.items():
            record_path = data_dir / rec_id
            if not record_path.with_suffix(".dat").exists():
                print(f" [!] Warning: Record {rec_id} not found at '{record_path}'. Skipping.")
                continue

            record = wfdb.rdrecord(str(record_path))
            ecg_signal = record.p_signal[:, 0]

            segments_1d, matrices_2d = preprocessor.process_raw_record(ecg_signal)
            n_segs = len(segments_1d)
            labels = np.full(n_segs, class_id, dtype=np.int64)

            target_split = "train"
            if rec_id in val_recs:
                target_split = "val"
            elif rec_id in test_recs:
                target_split = "test"

            data_splits[target_split]["1d"].append(segments_1d)
            data_splits[target_split]["2d"].append(matrices_2d)
            data_splits[target_split]["labels"].append(labels)

        datasets = {}
        for split in ["train", "val", "test"]:
            all_1d = np.concatenate(data_splits[split]["1d"], axis=0)
            all_2d = np.concatenate(data_splits[split]["2d"], axis=0)
            all_labels = np.concatenate(data_splits[split]["labels"], axis=0)
            datasets[split] = ArrhythmiaDataset(all_2d, all_1d, all_labels)
            print(f"[*] Split '{split}': Total samples = {len(datasets[split])}")

        return datasets["train"], datasets["val"], datasets["test"], config

    else:
        # Segment-level stratified train/val/test split
        all_1d_list = []
        all_2d_list = []
        all_labels_list = []

        for rec_id, class_id in rec_to_class.items():
            record_path = data_dir / rec_id
            if not record_path.with_suffix(".dat").exists():
                print(f" [!] Warning: Record {rec_id} not found at '{record_path}'. Skipping.")
                continue

            record = wfdb.rdrecord(str(record_path))
            ecg_signal = record.p_signal[:, 0]

            segments_1d, matrices_2d = preprocessor.process_raw_record(ecg_signal)
            n_segs = len(segments_1d)
            labels = np.full(n_segs, class_id, dtype=np.int64)

            all_1d_list.append(segments_1d)
            all_2d_list.append(matrices_2d)
            all_labels_list.append(labels)

        X_1d = np.concatenate(all_1d_list, axis=0)
        X_2d = np.concatenate(all_2d_list, axis=0)
        y = np.concatenate(all_labels_list, axis=0)

        train_ratio = config["split"].get("train_ratio", 0.70)
        val_ratio = config["split"].get("val_ratio", 0.15)
        test_ratio = config["split"].get("test_ratio", 0.15)

        # First split train vs (val + test)
        X_train_2d, X_temp_2d, X_train_1d, X_temp_1d, y_train, y_temp = train_test_split(
            X_2d, X_1d, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=42
        )

        # Then split val vs test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        X_val_2d, X_test_2d, X_val_1d, X_test_1d, y_val, y_test = train_test_split(
            X_temp_2d, X_temp_1d, y_temp, test_size=relative_test_ratio, stratify=y_temp, random_state=42
        )

        train_ds = ArrhythmiaDataset(X_train_2d, X_train_1d, y_train)
        val_ds = ArrhythmiaDataset(X_val_2d, X_val_1d, y_val)
        test_ds = ArrhythmiaDataset(X_test_2d, X_test_1d, y_test)

        print(f"[*] Stratified Segment Split -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        return train_ds, val_ds, test_ds, config


def get_dataloaders(config_path: str = "config.yaml") -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Factory helper to build PyTorch DataLoaders for train, val, and test splits."""
    train_ds, val_ds, test_ds, config = build_dataset_from_records(config_path)
    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, config


if __name__ == "__main__":
    t_loader, v_loader, test_loader, cfg = get_dataloaders()
    for b_2d, b_1d, b_y in t_loader:
        print(f"Sample Batch -> 2D Matrix Shape: {b_2d.shape}, 1D Segment Shape: {b_1d.shape}, Labels Shape: {b_y.shape}")
        break
