"""Audio capture script for ICU patient monitoring.

Records audio from the microphone in 3-second chunks and sends each chunk
to the FastAPI backend for distress classification.

Usage:
    python audio_capture.py [--backend-url http://127.0.0.1:8000] [--duration 300]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import requests
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0
CHANNELS = 1
LOG_FILE = "audio_capture.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def send_audio(audio_chunk: np.ndarray, backend_url: str) -> dict | None:
    try:
        resp = requests.post(
            f"{backend_url}/audio",
            json={"chunk": audio_chunk.tolist(), "sample_rate": SAMPLE_RATE},
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        log(f"Backend error: {e}")
        return None


def run(backend_url: str, duration: float) -> None:
    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    log(f"Starting mic capture: {SAMPLE_RATE}Hz, {CHUNK_DURATION}s chunks")
    log(f"Backend: {backend_url}/audio")

    start = time.time()
    chunk_count = 0

    try:
        while True:
            audio = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
            sd.wait()
            audio = audio.flatten()
            chunk_count += 1

            max_amp = np.max(np.abs(audio))
            log(f"Chunk #{chunk_count}: {len(audio)} samples, max_amp={max_amp:.4f}")

            if max_amp < 0.001:
                log("  -> Silent, skipping")
            else:
                result = send_audio(audio, backend_url)
                if result and result.get("received"):
                    analysis = result.get("audio_analysis", {})
                    score = analysis.get("distress_score", 0)
                    level = analysis.get("alert_level", "?")
                    dominant = analysis.get("dominant_class", "?")
                    log(f"  -> {level} | score={score:.3f} | dominant={dominant}")
                else:
                    log(f"  -> Send failed: {result}")

            if duration > 0 and (time.time() - start) >= duration:
                log("Duration reached, stopping.")
                break
    except KeyboardInterrupt:
        log("Stopped by user.")
    except Exception as e:
        log(f"FATAL: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream mic audio to ICU backend")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, default=0, help="0 = unlimited")
    args = parser.parse_args()

    # Clear old log
    open(LOG_FILE, "w").close()

    run(args.backend_url, args.duration)


if __name__ == "__main__":
    main()
