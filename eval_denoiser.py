import os
import torch
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000
DURATION = 3.0
N_SAMPLES = int(TARGET_SR * DURATION)
N_FFT = 512
HOP_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from train_spec_denoiser import SpecUNet, stft, istft
denoiser = SpecUNet()
state = torch.load("checkpoints/spec_denoiser.pth", map_location="cpu")
denoiser.load_state_dict(state)
denoiser.to(device)
denoiser.eval()
print("Loaded spectrogram denoiser")

base = os.path.dirname(os.path.abspath(__file__))

noise_dirs = [
    "archive/Hospital noise original/Hospital noise original",
    "archive/icu_machine_sounds/sounddino_hospital",
    "archive/icu_machine_sounds/sounddino_heart",
    "archive/icu_machine_sounds/bigsoundbank",
]
noise_files = []
for nd in noise_dirs:
    p = os.path.join(base, nd)
    if os.path.isdir(p):
        noise_files.extend([os.path.join(p, f) for f in os.listdir(p)])

vocal_dir = os.path.join(base, "data", "vocal_16k")
vocal_files = []
for cls in ["Cough", "Crying", "Moan", "Pant", "Normal"]:
    d = os.path.join(vocal_dir, cls)
    if os.path.isdir(d):
        vocal_files.extend([(os.path.join(d, f), cls) for f in os.listdir(d) if f.endswith('.wav')])

out_dir = os.path.join(base, "eval_output")
os.makedirs(out_dir, exist_ok=True)

for test_idx in range(min(10, len(vocal_files))):
    clean_path, cls = vocal_files[test_idx * 500]
    noise_path = noise_files[np.random.randint(len(noise_files))]

    clean, sr = librosa.load(clean_path, sr=TARGET_SR, mono=True)
    if len(clean) > N_SAMPLES:
        clean = clean[:N_SAMPLES]
    else:
        clean = np.pad(clean, (0, N_SAMPLES - len(clean)))

    noise, _ = librosa.load(noise_path, sr=TARGET_SR, mono=True)
    if len(noise) > N_SAMPLES:
        start = np.random.randint(0, len(noise) - N_SAMPLES)
        noise = noise[start:start + N_SAMPLES]
    else:
        noise = np.pad(noise, (0, N_SAMPLES - len(noise)))

    clean = clean.astype(np.float32)
    peak = np.max(np.abs(clean))
    if peak > 0: clean /= peak

    noise = noise.astype(np.float32)
    peak = np.max(np.abs(noise))
    if peak > 0: noise /= peak

    snr_db = 3
    sp = np.mean(clean**2) + 1e-8
    np_ = np.mean(noise**2) + 1e-8
    noise_scaled = noise * np.sqrt(sp / (np_ * 10**(snr_db/10)))
    mixed = np.clip(clean + noise_scaled, -1.0, 1.0)

    mixed_spec = np.abs(stft(mixed)).astype(np.float32)
    mixed_spec_log = np.log1p(mixed_spec)
    mixed_t = torch.tensor(mixed_spec_log, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        clean_spec_log = denoiser(mixed_t).squeeze().cpu().numpy()

    clean_spec = np.expm1(np.maximum(clean_spec_log, 0))
    mixed_phase = np.angle(stft(mixed))
    denoised = istft(clean_spec, mixed_phase)
    denoised = np.clip(denoised, -1.0, 1.0)

    sf.write(os.path.join(out_dir, f"test{test_idx}_{cls}_clean.wav"), clean, TARGET_SR)
    sf.write(os.path.join(out_dir, f"test{test_idx}_{cls}_noisy.wav"), mixed, TARGET_SR)
    sf.write(os.path.join(out_dir, f"test{test_idx}_{cls}_denoised.wav"), denoised, TARGET_SR)

    print(f"Test {test_idx} [{cls}]: noisy (SNR={snr_db}dB) -> denoised | noise={os.path.basename(noise_path)[:30]}")

print(f"\nAll files saved to: {out_dir}")
print("Listen to noisy vs denoised to judge quality!")
