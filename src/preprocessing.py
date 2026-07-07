import os
import numpy as np
import librosa
from pathlib import Path


TARGET_SR = 16000
DURATION = 3.0
N_SAMPLES = int(TARGET_SR * DURATION)
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024


def add_icu_noise(y, sr=TARGET_SR):
    """Add realistic ICU machine noise to raw audio"""
    noise = np.zeros_like(y)

    # Power line hum (50/60 Hz)
    t = np.linspace(0, len(y) / sr, len(y), endpoint=False)
    for freq in [50, 60]:
        amp = np.random.uniform(0.005, 0.02)
        noise += amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))

    # Equipment beeps
    if np.random.random() < 0.6:
        n_beeps = np.random.randint(1, 5)
        for _ in range(n_beeps):
            beep_len = int(np.random.uniform(0.03, 0.15) * sr)
            start = np.random.randint(0, max(1, len(y) - beep_len))
            freq = np.random.uniform(800, 2500)
            t_beep = np.linspace(0, beep_len / sr, beep_len, endpoint=False)
            amp = np.random.uniform(0.01, 0.05)
            beep = amp * np.sin(2 * np.pi * freq * t_beep)
            beep *= np.hanning(beep_len)
            noise[start:start + beep_len] += beep

    # Broadband machine noise (fans, ventilators)
    if np.random.random() < 0.5:
        noise += np.random.normal(0, np.random.uniform(0.003, 0.012), len(y))

    return np.clip(y + noise, -1.0, 1.0)


def load_audio(path, sr=TARGET_SR, duration=DURATION):
    y, orig_sr = librosa.load(path, sr=sr, duration=duration)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]
    return y


def normalize_audio(y):
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y


def extract_features(y, sr=TARGET_SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_delta = librosa.feature.delta(mel_db)
    mel_delta2 = librosa.feature.delta(mel_db, order=2)
    features = np.stack([mel_db, mel_delta, mel_delta2], axis=-1)
    return features


def process_directory(input_dir, output_dir, label_map):
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for class_name, label_id in label_map.items():
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping")
            continue

        audio_files = list(Path(class_dir).rglob("*.flac"))
        audio_files += list(Path(class_dir).rglob("*.wav"))
        print(f"Processing {class_name}: {len(audio_files)} files")

        for i, audio_path in enumerate(audio_files):
            try:
                y = load_audio(str(audio_path))
                y = normalize_audio(y)
                features = extract_features(y)

                out_path = os.path.join(
                    output_dir, f"{class_name}_{i:05d}.npy"
                )
                np.save(out_path, features)

                records.append({
                    "file": out_path,
                    "label": label_id,
                    "class": class_name,
                    "original": str(audio_path),
                })

                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(audio_files)} done")
            except Exception as e:
                print(f"  Error processing {audio_path}: {e}")

    return records


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "vocal_bursts")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

    label_map = {
        "Cough": 0,
        "Crying": 1,
        "Moan": 2,
        "Pant": 3,
        "Normal": 4,
    }

    records = process_directory(RAW_DIR, PROCESSED_DIR, label_map)
    print(f"\nTotal processed: {len(records)} samples")

    import json
    with open(os.path.join(PROCESSED_DIR, "metadata.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"Metadata saved to {PROCESSED_DIR}/metadata.json")
