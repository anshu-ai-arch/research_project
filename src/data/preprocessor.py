import numpy as np
import scipy.signal as signal
from typing import Tuple, List


class ECGPreprocessor:
    """
    Implements paper-exact preprocessing for ECG signals:
    1. 4th-order zero-phase Butterworth Bandpass Filter (0.5 - 50 Hz).
    2. Resampling from raw frequency (360 Hz) to target frequency (128 Hz).
    3. 10-second Segmentation (1280 samples).
    4. Min-Max Normalization per segment [0, 1].
    5. 2D Matrix Reshaping (40 x 32).
    """

    def __init__(
        self,
        raw_fs: float = 360.0,
        target_fs: float = 128.0,
        lowcut: float = 0.5,
        highcut: float = 50.0,
        filter_order: int = 4,
        segment_duration_sec: float = 10.0,
        matrix_rows: int = 40,
        matrix_cols: int = 32,
        use_filtering: bool = True,
    ):
        self.raw_fs = raw_fs
        self.target_fs = target_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.segment_duration_sec = segment_duration_sec
        self.matrix_rows = matrix_rows
        self.matrix_cols = matrix_cols
        self.use_filtering = use_filtering

        self.samples_per_segment = int(self.target_fs * self.segment_duration_sec)  # 1280
        assert (
            self.matrix_rows * self.matrix_cols in (self.samples_per_segment, 256)
        ), f"Matrix size {self.matrix_rows}x{self.matrix_cols} != {self.samples_per_segment} or 256"

    def bandpass_filter(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Applies 4th order zero-phase Butterworth bandpass filter (0.5Hz - 50Hz)."""
        nyquist = 0.5 * self.raw_fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Ensure highcut is less than Nyquist
        if high >= 1.0:
            high = 0.99
            
        b, a = signal.butter(self.filter_order, [low, high], btype="bandpass")
        # Zero-phase digital filtering to avoid phase distortion
        filtered_signal = signal.filtfilt(b, a, ecg_signal)
        return filtered_signal

    def resample_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Resamples ECG signal to target sampling rate (128 Hz)."""
        if self.raw_fs == self.target_fs:
            return ecg_signal

        num_target_samples = int(len(ecg_signal) * (self.target_fs / self.raw_fs))
        resampled_signal = signal.resample(ecg_signal, num_target_samples)
        return resampled_signal

    def segment_signal(self, ecg_signal: np.ndarray) -> List[np.ndarray]:
        """Partitions continuous signal into 10-second (1280 samples) segments."""
        n_samples = len(ecg_signal)
        segments = []

        for i in range(0, n_samples - self.samples_per_segment + 1, self.samples_per_segment):
            seg = ecg_signal[i : i + self.samples_per_segment]
            segments.append(seg)

        return segments

    def normalize_segment(self, segment: np.ndarray) -> np.ndarray:
        """Applies Min-Max scaling to range [0, 1]."""
        min_val = np.min(segment)
        max_val = np.max(segment)

        if max_val - min_val == 0:
            return np.zeros_like(segment)

        normalized = (segment - min_val) / (max_val - min_val)
        return normalized

    def reshape_2d(self, segment: np.ndarray) -> np.ndarray:
        """Reshapes 1D segment of length 1280 into 2D matrix of shape (40, 32)."""
        return segment.reshape((self.matrix_rows, self.matrix_cols))

    def process_raw_record(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete processing pipeline for a single raw ECG record.
        Returns:
            - 1D segments array: (N_segments, 1280)
            - 2D matrices array: (N_segments, 40, 32)
        """
        # Step 1: Bandpass Filter (Optional based on self.use_filtering)
        if self.use_filtering:
            signal_to_process = self.bandpass_filter(ecg_signal)
        else:
            signal_to_process = ecg_signal

        # Step 2: Resample to 128 Hz
        resampled = self.resample_signal(signal_to_process)

        # Step 3: Segment into 10s windows (1280 samples)
        raw_segments = self.segment_signal(resampled)

        segments_1d = []
        matrices_2d = []

        # Step 4 & 5: Normalize and Reshape to 2D
        for seg in raw_segments:
            norm_seg = self.normalize_segment(seg)
            mat_2d = self.reshape_2d(norm_seg)

            segments_1d.append(norm_seg)
            matrices_2d.append(mat_2d)

        return np.array(segments_1d, dtype=np.float32), np.array(matrices_2d, dtype=np.float32)
