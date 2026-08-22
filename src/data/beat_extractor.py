import numpy as np
import wfdb
from pathlib import Path
from typing import Tuple, List, Dict


# AAMI Standard 5-Class Symbol Mapping
AAMI_MAPPING = {
    # Class 0: Normal & Bundle Branch Block beats (N)
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    # Class 1: Supraventricular Ectopic Beats & A-Fib (SVEB / A)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    # Class 2: Ventricular Ectopic Beats / PVC (VEB / PVC)
    'V': 2, 'E': 2, 'r': 2,
    # Class 3: Fusion & Ventricular Tachycardia (F / VT)
    'F': 3, '[': 3, ']': 3, '!': 3,
    # Class 4: Unknown / Paced / Other (Q)
    '/': 4, 'f': 4, 'Q': 4, '?': 4
}


class ECGBeatExtractor:
    """
    Extracts individual R-peak centered heartbeat windows and normalized pre/post RR-intervals
    from PhysioNet ECG records, mapping ground-truth annotations to AAMI 5-class standard.
    """

    def __init__(self, beat_window_size: int = 256, matrix_rows: int = 16, matrix_cols: int = 16):
        self.window_size = beat_window_size
        self.half_window = beat_window_size // 2
        self.matrix_rows = matrix_rows
        self.matrix_cols = matrix_cols
        assert matrix_rows * matrix_cols == beat_window_size, "Matrix size must equal beat_window_size!"

    def extract_beats_from_record(
        self, record_path: Path, ecg_signal: np.ndarray, raw_fs: float = 360.0, target_fs: float = 128.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parses PhysioNet .atr annotation file, extracts R-peak centered windows,
        computes pre-RR and post-RR interval features (in seconds, normalized), and assigns ground-truth AAMI labels.
        
        Returns:
            - 1D beat segments: (N_beats, 256)
            - 2D beat matrices: (N_beats, 16, 16)
            - RR features (pre_RR, post_RR): (N_beats, 2)
            - AAMI labels: (N_beats,)
        """
        try:
            ann = wfdb.rdann(str(record_path), 'atr')
        except Exception as e:
            print(f" [!] Error reading annotation file for {record_path}: {e}")
            return (
                np.empty((0, self.window_size)),
                np.empty((0, self.matrix_rows, self.matrix_cols)),
                np.empty((0, 2)),
                np.empty((0,))
            )

        r_peaks = ann.sample
        symbols = ann.symbol

        valid_indices = []
        valid_symbols = []
        for idx, (r_peak, sym) in enumerate(zip(r_peaks, symbols)):
            if sym in AAMI_MAPPING:
                valid_indices.append(idx)
                valid_symbols.append(sym)

        beats_1d = []
        beats_2d = []
        rr_features = []
        labels = []

        sig_len = len(ecg_signal)
        ratio = target_fs / raw_fs

        # Calculate median RR interval for boundary fallback
        diffs = np.diff(r_peaks) / raw_fs
        median_rr = np.median(diffs) if len(diffs) > 0 else 0.8  # ~800ms default

        for i, idx in enumerate(valid_indices):
            r_peak = r_peaks[idx]
            sym = valid_symbols[i]
            class_id = AAMI_MAPPING[sym]

            # Compute pre_RR and post_RR in seconds
            if idx > 0:
                pre_rr = (r_peaks[idx] - r_peaks[idx - 1]) / raw_fs
            else:
                pre_rr = median_rr

            if idx < len(r_peaks) - 1:
                post_rr = (r_peaks[idx + 1] - r_peaks[idx]) / raw_fs
            else:
                post_rr = median_rr

            # Normalize RR intervals into [0, 1] range (clipped between 0.2s and 2.0s)
            pre_rr_norm = np.clip((pre_rr - 0.2) / 1.8, 0.0, 1.0)
            post_rr_norm = np.clip((post_rr - 0.2) / 1.8, 0.0, 1.0)

            # Convert R-peak sample index if signal is resampled
            r_idx = int(r_peak * ratio) if raw_fs != target_fs else r_peak

            start_idx = r_idx - self.half_window
            end_idx = r_idx + self.half_window

            if start_idx < 0 or end_idx > sig_len:
                continue

            beat_wave = ecg_signal[start_idx:end_idx]
            if len(beat_wave) != self.window_size:
                continue

            # Min-Max Scaling per heartbeat window [0, 1]
            min_v, max_v = np.min(beat_wave), np.max(beat_wave)
            if max_v - min_v > 0:
                beat_wave = (beat_wave - min_v) / (max_v - min_v)
            else:
                beat_wave = np.zeros_like(beat_wave)

            beat_2d = beat_wave.reshape((self.matrix_rows, self.matrix_cols))

            beats_1d.append(beat_wave)
            beats_2d.append(beat_2d)
            rr_features.append([pre_rr_norm, post_rr_norm])
            labels.append(class_id)

        return (
            np.array(beats_1d, dtype=np.float32),
            np.array(beats_2d, dtype=np.float32),
            np.array(rr_features, dtype=np.float32),
            np.array(labels, dtype=np.int64)
        )
