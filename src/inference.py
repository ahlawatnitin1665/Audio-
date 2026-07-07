import os
import sys
import time
import numpy as np
import torch
import sounddevice as sd
import librosa

sys.path.insert(0, os.path.dirname(__file__))
from model import DistressCalculator, NUM_CLASSES
from preprocessing import TARGET_SR, DURATION
from simple_denoiser import SimpleTwoStagePipeline


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIER_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
DENOISER_PATH = os.path.join(BASE_DIR, "checkpoints", "simple_denoiser.pth")
CHUNK_DURATION = 3.0
OVERLAP = 0.5
SAMPLE_RATE = TARGET_SR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RealTimeMonitor:
    def __init__(self, threshold=0.4):
        self.pipeline = SimpleTwoStagePipeline(denoiser_path=DENOISER_PATH, classifier_path=CLASSIFIER_PATH)
        self.distress_calc = DistressCalculator()
        self.threshold = threshold
        self.class_names = ["coughing", "crying", "groaning", "gasping", "normal", "noise"]
        self.history = []

    def predict_audio(self, audio):
        result = self.pipeline.predict(audio, SAMPLE_RATE)
        probs = torch.tensor([result['probabilities'][name] for name in self.class_names], dtype=torch.float32)
        return probs

    def monitor_chunk(self, audio):
        max_amp = np.max(np.abs(audio))
        if max_amp < 0.001:
            probs = torch.zeros(5, dtype=torch.float32).to(DEVICE)
            probs[4] = 1.0
            distress_score = 0.0
        else:
            probs = self.predict_audio(audio)
            distress_score = self.distress_calc.calculate(probs)
        level, priority = self.distress_calc.get_alert_level(distress_score)
        dominant, conf = self.distress_calc.get_dominant_class(probs)

        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "distress_score": distress_score,
            "alert_level": level,
            "priority": priority,
            "dominant_class": dominant,
            "confidence": conf,
            "probabilities": {name: probs[i].item() for i, name in enumerate(self.class_names)},
        }

        self.history.append(result)
        return result

    def print_result(self, result):
        print(f"\n[{result['timestamp']}]")
        print(f"  Distress Score: {result['distress_score']:.3f}")
        print(f"  Alert: {result['alert_level']} ({result['priority']})")
        print(f"  Dominant: {result['dominant_class']} ({result['confidence']:.1%})")
        probs = result['probabilities']
        print(f"  Probs: cough={probs['coughing']:.2f} cry={probs['crying']:.2f} "
              f"groan={probs['groaning']:.2f} pant={probs['gasping']:.2f}")

    def run_live(self, duration=60):
        print(f"Starting real-time monitoring for {duration} seconds...")
        print("Press Ctrl+C to stop early\n")

        chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
        step_samples = int((CHUNK_DURATION - OVERLAP) * SAMPLE_RATE)

        try:
            elapsed = 0
            while elapsed < duration:
                audio = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
                sd.wait()
                audio = audio.flatten()
                result = self.monitor_chunk(audio)
                self.print_result(result)
                elapsed += CHUNK_DURATION
                time.sleep(max(0, OVERLAP))
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

    def run_file(self, filepath):
        print(f"Processing file: {filepath}")
        y, sr = librosa.load(filepath, sr=TARGET_SR, duration=30)
        chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)

        results = []
        for start in range(0, len(y), chunk_samples):
            chunk = y[start:start + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            result = self.monitor_chunk(chunk)
            self.print_result(result)
            results.append(result)

        return results

    def get_summary(self):
        if not self.history:
            return "No data collected"
        scores = [r["distress_score"] for r in self.history]
        alerts = [r for r in self.history if r["alert_level"] != "Normal"]
        return {
            "total_chunks": len(self.history),
            "avg_distress": np.mean(scores),
            "max_distress": np.max(scores),
            "alert_count": len(alerts),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time patient audio monitor")
    parser.add_argument("--mode", choices=["live", "file"], default="live")
    parser.add_argument("--file", type=str, help="Audio file path for file mode")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    parser.add_argument("--threshold", type=float, default=0.4, help="Distress alert threshold")
    args = parser.parse_args()

    monitor = RealTimeMonitor(threshold=args.threshold)

    if args.mode == "live":
        monitor.run_live(duration=args.duration)
    elif args.mode == "file":
        if not args.file:
            print("Error: --file required for file mode")
            sys.exit(1)
        monitor.run_file(args.file)

    summary = monitor.get_summary()
    print(f"\nSummary: {summary}")
