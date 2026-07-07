import os, sys, librosa, numpy as np, soundfile as sf, random

BASE = r"C:\Users\Lenovo\OneDrive\Desktop\elc\patient_audio_monitor"
SR = 16000
DURATION = 9.0
N_SAMPLES = int(SR * DURATION)

# Pick a random gasping (Pant) sample
pant_dir = os.path.join(BASE, "data", "vocal_16k", "Pant")
pant_files = [f for f in os.listdir(pant_dir) if f.endswith(".wav")]
pant_path = os.path.join(pant_dir, random.choice(pant_files))
vocal, _ = librosa.load(pant_path, sr=SR, mono=True)
vocal = vocal.astype(np.float32)
peak = max(np.max(np.abs(vocal)), 1e-8)
vocal /= peak

# Repeat/pad to fill duration
if len(vocal) < N_SAMPLES:
    repeats = int(np.ceil(N_SAMPLES / len(vocal)))
    vocal = np.tile(vocal, repeats)[:N_SAMPLES]
else:
    vocal = vocal[:N_SAMPLES]

# Pick a random ICU noise
noise_dirs = [
    os.path.join(BASE, "archive", "Hospital noise original", "Hospital noise original"),
    os.path.join(BASE, "archive", "icu_machine_sounds", "sounddino_hospital"),
    os.path.join(BASE, "archive", "icu_machine_sounds", "sounddino_heart"),
    os.path.join(BASE, "archive", "icu_machine_sounds", "bigsoundbank"),
]
nd = random.choice(noise_dirs)
nf = random.choice([f for f in os.listdir(nd) if f.endswith((".wav", ".mp3", ".flac"))])
noise, _ = librosa.load(os.path.join(nd, nf), sr=SR, mono=True)
noise = noise.astype(np.float32)
peak_n = max(np.max(np.abs(noise)), 1e-8)
noise /= peak_n

# Trim/pad noise to match
if len(noise) < N_SAMPLES:
    repeats = int(np.ceil(N_SAMPLES / len(noise)))
    noise = np.tile(noise, repeats)[:N_SAMPLES]
else:
    noise = noise[:N_SAMPLES]

# Mix at 0dB SNR (equal energy)
sp = np.mean(vocal ** 2) + 1e-8
np_ = np.mean(noise ** 2) + 1e-8
noise_scaled = noise * np.sqrt(sp / np_)
mix = np.clip(vocal + noise_scaled, -1.0, 1.0)

out_path = os.path.join(BASE, "test_gasp_noise.wav")
sf.write(out_path, mix, SR)
print(f"Created: {out_path}")
print(f"  Source gasp: {os.path.basename(pant_path)}")
print(f"  Source noise: {nf}")
print(f"  Duration: {DURATION:.0f}s")
