import sys
import time
import torch
import numpy as np
import scipy.signal as signal
from pathlib import Path


# =====================================================================
# 1. BANDPASS FILTERING (4th-order Butterworth 0.5 - 50.0 Hz)
# =====================================================================
def bandpass_filter_ecg(raw_signal: np.ndarray, fs: float = 360.0, lowcut: float = 0.5, highcut: float = 50.0) -> np.ndarray:
    """Filters baseline wander (<0.5 Hz) and muscle/powerline noise (>50 Hz)."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, raw_signal)
    return filtered


# =====================================================================
# 2. SIGNAL RESAMPLING & MIN-MAX NORMALIZATION
# =====================================================================
def preprocess_ecg_beat_window(beat_window_256: np.ndarray) -> torch.Tensor:
    """
    Min-max normalizes a 256-sample beat window to [0, 1]
    and converts to PyTorch Float Tensor [1, 1, 256] for ARM inference.
    """
    min_val = np.min(beat_window_256)
    max_val = np.max(beat_window_256)
    norm_window = (beat_window_256 - min_val) / (max_val - min_val + 1e-8)
    tensor_input = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor_input


# =====================================================================
# 3. RASPBERRY PI HARDWARE INFERENCE ENGINE
# =====================================================================
class RaspberryPiECGInferenceEngine:

    def __init__(self, model_path: str = "checkpoints/model_b_medium_torchscript.pt"):
        print("=================================================================")
        print("    RASPBERRY PI REAL-TIME ECG ARRHYTHMIA DIAGNOSTIC ENGINE     ")
        print("=================================================================")
        
        self.device = torch.device("cpu")
        print(f"[*] Target Hardware Platform: ARM CPU (Raspberry Pi)")
        
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"[!] TorchScript model not found at '{model_file}'")

        print(f"[*] Loading TorchScript compiled model: '{model_file}'...")
        start_t = time.time()
        self.model = torch.jit.load(str(model_file), map_location=self.device)
        self.model.eval()
        load_time_ms = (time.time() - start_t) * 1000.0
        print(f"[✓] Model successfully loaded into RAM in {load_time_ms:.2f} ms!")

        self.class_names = [
            "Normal Beat (N)",
            "Supraventricular Ectopic Beat (SVEB / A)",
            "Ventricular Ectopic Beat / PVC (VEB / PVC)",
            "Fusion Beat (F / VT)",
            "Unknown / Paced Beat (Q)"
        ]

    @torch.no_grad()
    def predict_beat(self, beat_window_256: np.ndarray) -> dict:
        """
        Runs real-time inference on a 256-sample R-peak centered beat window.
        Returns predicted class, confidence scores, and latency in milliseconds.
        """
        tensor_input = preprocess_ecg_beat_window(beat_window_256).to(self.device)

        start_time = time.time()
        output_logits = self.model(tensor_input)
        probabilities = torch.softmax(output_logits, dim=1).cpu().numpy()[0]
        inference_latency_ms = (time.time() - start_time) * 1000.0

        predicted_class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class_id]) * 100.0

        is_critical_arrhythmia = (predicted_class_id == 2)  # VEB/PVC alert

        return {
            "class_id": predicted_class_id,
            "class_name": self.class_names[predicted_class_id],
            "confidence_pct": confidence,
            "all_probabilities": probabilities.tolist(),
            "latency_ms": inference_latency_ms,
            "critical_alert": is_critical_arrhythmia
        }


# =====================================================================
# 4. DEMO RUNNER
# =====================================================================
if __name__ == "__main__":
    engine = RaspberryPiECGInferenceEngine()

    print("\n[*] Running simulated real-time heartbeat inference stream...")
    np.random.seed(42)

    for beat_num in range(1, 6):
        # Generate simulated 256-sample ECG beat window
        simulated_beat = np.sin(np.linspace(0, 4 * np.pi, 256)) + np.random.normal(0, 0.05, 256)
        if beat_num == 3:  # Simulate a PVC arrhythmia spike
            simulated_beat[110:140] += 2.5

        result = engine.predict_beat(simulated_beat)

        alert_tag = " [🚨 CRITICAL VEB/PVC ALERT!]" if result["critical_alert"] else ""
        print(f"Beat #{beat_num:02d} | Class: {result['class_name']:<40} | Conf: {result['confidence_pct']:6.2f}% | Latency: {result['latency_ms']:.2f} ms{alert_tag}")
        time.sleep(0.1)
