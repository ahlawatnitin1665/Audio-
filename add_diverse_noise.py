import os, sys, json, numpy as np, torch, sounddevice as sd
from glob import glob
sys.path.insert(0, "src")
from preprocessing import extract_features, normalize_audio, TARGET_SR, N_SAMPLES
from train_spec_denoiser import SpecUNet, stft, istft

BASE = r"C:\Users\Lenovo\OneDrive\Desktop\elc\patient_audio_monitor"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

denoiser = SpecUNet().to(DEVICE)
denoiser.load_state_dict(torch.load(os.path.join(BASE, "checkpoints", "spec_denoiser.pth"), map_location=DEVICE))
denoiser.eval()

def denoise_chunk(audio_chunk):
    spec = np.abs(stft(audio_chunk)).astype(np.float32)
    spec_log = np.log1p(spec)
    t = torch.tensor(spec_log, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = denoiser(t).squeeze().cpu().numpy()
    out_spec = np.expm1(np.maximum(out, 0))
    phase = np.angle(stft(audio_chunk))
    return np.clip(istft(out_spec, phase), -1.0, 1.0)

noise_meta_path = os.path.join(BASE, "data", "processed", "noise_metadata.json")
with open(noise_meta_path) as f:
    records = json.load(f)

out_dir = os.path.join(BASE, "data", "processed")
den_dir = os.path.join(BASE, "data", "processed_denoised")
start_idx = len(records)
count = 0

print("Recording 60s of quiet room tone (stay silent, just room ambience)...")
audio = sd.rec(60 * TARGET_SR, samplerate=TARGET_SR, channels=1, device=1, dtype="float32")
sd.wait()
audio = audio[:, 0].astype(np.float32)
peak = max(np.max(np.abs(audio)), 1e-8)
audio /= peak

for j in range(0, len(audio), N_SAMPLES // 2):
    chunk = audio[j:j + N_SAMPLES]
    if len(chunk) < TARGET_SR:
        continue
    if len(chunk) < N_SAMPLES:
        chunk = np.pad(chunk, (0, N_SAMPLES - len(chunk)))
    feat = extract_features(normalize_audio(chunk))
    name = f"roomtone_{start_idx + count:04d}.npy"
    np.save(os.path.join(out_dir, name), feat)
    den = denoise_chunk(chunk)
    feat_d = extract_features(normalize_audio(den))
    np.save(os.path.join(den_dir, name), feat_d)
    records.append({"file": os.path.join(out_dir, name), "label": 5, "class": "noise"})
    count += 1

print("Recording 60s of white noise (fan, static, hiss)...")
audio2 = sd.rec(60 * TARGET_SR, samplerate=TARGET_SR, channels=1, device=1, dtype="float32")
sd.wait()
audio2 = audio2[:, 0].astype(np.float32)
peak2 = max(np.max(np.abs(audio2)), 1e-8)
audio2 /= peak2

for j in range(0, len(audio2), N_SAMPLES // 2):
    chunk = audio2[j:j + N_SAMPLES]
    if len(chunk) < TARGET_SR:
        continue
    if len(chunk) < N_SAMPLES:
        chunk = np.pad(chunk, (0, N_SAMPLES - len(chunk)))
    feat = extract_features(normalize_audio(chunk))
    name = f"whitenoise_{start_idx + count:04d}.npy"
    np.save(os.path.join(out_dir, name), feat)
    den = denoise_chunk(chunk)
    feat_d = extract_features(normalize_audio(den))
    np.save(os.path.join(den_dir, name), feat_d)
    records.append({"file": os.path.join(out_dir, name), "label": 5, "class": "noise"})
    count += 1

with open(noise_meta_path, "w") as f:
    json.dump(records, f, indent=2)
print(f"Added {count} room tone + white noise chunks. Total noise: {len(records)}")
