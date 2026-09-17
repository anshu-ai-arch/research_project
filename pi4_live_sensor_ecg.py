import time
import torch
import numpy as np
import scipy.signal as signal
from pathlib import Path

# Hardware I2C / ADC Imports
try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    HARDWARE_ADC_AVAILABLE = True
except ImportError:
    HARDWARE_ADC_AVAILABLE = False


class RealTimeAD8232Pi4Engine:
    """
    Real-time AD8232 Single-Lead ECG Hardware Diagnostic Engine for Raspberry Pi 4.
    Reads continuous analog signals at 128 Hz, filters noise, detects R-peaks,
    and runs TorchScript AI model inference in real-time.
    """

    def __init__(self, model_path: str = "checkpoints/model_b_medium_torchscript.pt"):
        print("=================================================================")
        print("    RASPBERRY PI 4 — LIVE AD8232 SENSOR HARDWARE DIAGNOSTIC    ")
        print("=================================================================")
        
        self.device = torch.device("cpu")
        self.fs = 128.0  # Target sampling rate: 128 Hz
        self.window_size = 256  # 2.0 seconds beat window
        
        print(f"[*] Loading TorchScript AI Model '{model_path}'...")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("[✓] Model loaded into Pi 4 RAM!")

        # Initialize Hardware ADC if available
        if HARDWARE_ADC_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                ads = ADS.ADS1115(i2c)
                self.chan = AnalogIn(ads, ADS.P0)
                print("[✓] ADS1115 Hardware ADC connected on I2C Channel A0!")
            except Exception as e:
                print(f"[!] ADC Initialization Warning: {e}. Defaulting to Stream Simulation Mode.")
                self.chan = None
        else:
            print("[!] Adafruit ADS1115 library not installed. Defaulting to Stream Simulation Mode.")
            self.chan = None

        self.class_names = [
            "Normal Beat (N)",
            "Supraventricular Ectopic (SVEB/S)",
            "Ventricular Ectopic / PVC (VEB/V)",
            "Fusion Beat (F)",
            "Unknown Beat (Q)"
        ]

    def read_raw_voltage(self) -> float:
        """Reads a single voltage sample from ADS1115 Channel A0 or simulates signal."""
        if self.chan is not None:
            return self.chan.voltage
        else:
            # Simulated 128 Hz ECG Lead-II waveform
            t = time.time()
            val = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
            if np.random.rand() > 0.95:  # Random QRS spike
                val += 3.0
            return float(val)

    def filter_signal_buffer(self, buffer: np.ndarray) -> np.ndarray:
        """Applies 4th-order Butterworth bandpass filter (0.5 - 50 Hz)."""
        nyquist = 0.5 * self.fs
        b, a = signal.butter(4, [0.5 / nyquist, 50.0 / nyquist], btype='band')
        return signal.filtfilt(b, a, buffer)

    @torch.no_grad()
    def classify_beat_window(self, window_256: np.ndarray) -> dict:
        """Min-Max normalizes 256-sample window and runs Model B inference."""
        min_v = np.min(window_256)
        max_v = np.max(window_256)
        norm = (window_256 - min_v) / (max_v - min_v + 1e-8)
        
        tensor_in = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        start_t = time.time()
        logits = self.model(tensor_in)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        latency_ms = (time.time() - start_t) * 1000.0
        
        class_id = int(np.argmax(probs))
        conf = float(probs[class_id]) * 100.0
        
        return {
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "confidence": conf,
            "latency_ms": latency_ms,
            "is_critical": (class_id == 2)
        }

    def start_live_monitoring(self, duration_sec: int = 15):
        print(f"\n[*] Starting Live ECG Monitoring for {duration_sec} seconds...")
        print("[*] Sampling at 128 Hz... Applying Butterworth filter (0.5-50 Hz)...")
        
        buffer = []
        start_time = time.time()
        sample_interval = 1.0 / self.fs  # 7.81 ms per sample

        beat_count = 0
        while (time.time() - start_time) < duration_sec:
            sample_start = time.time()
            voltage = self.read_raw_voltage()
            buffer.append(voltage)

            if len(buffer) >= self.window_size:
                raw_win = np.array(buffer[-self.window_size:])
                filtered_win = self.filter_signal_buffer(raw_win)
                
                # Check for R-peak spike threshold
                if np.max(filtered_win) > 1.2:
                    beat_count += 1
                    result = self.classify_beat_window(filtered_win)
                    
                    alert = " [🚨 ALERT: PVC DETECTED!]" if result["is_critical"] else ""
                    print(f"[{time.strftime('%H:%M:%S')}] Beat #{beat_count:03d} | {result['class_name']:<35} | Conf: {result['confidence']:6.2f}% | Latency: {result['latency_ms']:.2f} ms{alert}")
                    
                    # Clear buffer window to avoid duplicate peak triggers
                    buffer = buffer[-64:]

            elapsed = time.time() - sample_start
            sleep_t = sample_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        print("\n[✓] Live Monitoring Session Complete!")


if __name__ == "__main__":
    engine = RealTimeAD8232Pi4Engine()
    engine.start_live_monitoring(duration_sec=10)
