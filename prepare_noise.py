import os, sys, json, librosa, numpy as np
sys.path.insert(0, "src")
from preprocessing import extract_features, normalize_audio, TARGET_SR, N_SAMPLES
from train_spec_denoiser import SpecUNet, stft, istft
import torch
from glob import glob

BASE = r"C:\Users\Lenovo\OneDrive\Desktop\elc\patient_audio_monitor"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

denoiser = SpecUNet().to(DEVICE)
denoiser.load_state_dict(torch.load(os.path.join(BASE, "checkpoints", "spec_denoiser.pth"), map_location=DEVICE))
denoiser.eval()

files = sorted(glob(os.path.join(BASE, "data", "noise_cache", "*.wav")))
print(f"Found {len(files)} noise WAVs")

out_dir = os.path.join(BASE, "data", "processed")
den_dir = os.path.join(BASE, "data", "processed_denoised")
os.makedirs(den_dir, exist_ok=True)

records = []
for i, fp in enumerate(files):
    try:
        y, _ = librosa.load(fp, sr=TARGET_SR, mono=True)
        y = y.astype(np.float32)
        peak = max(np.max(np.abs(y)), 1e-8)
        y /= peak
        for j in range(0, max(len(y), N_SAMPLES), N_SAMPLES // 2):
            chunk = y[j:j+N_SAMPLES]
            if len(chunk) < TARGET_SR: continue
            if len(chunk) < N_SAMPLES: chunk = np.pad(chunk, (0, N_SAMPLES-len(chunk)))
            feat = extract_features(normalize_audio(chunk))
            name = f"noise_{i:04d}_{j:04d}.npy"
            np.save(os.path.join(out_dir, name), feat)
            spec = np.abs(stft(chunk)).astype(np.float32)
            sl = np.log1p(spec)
            t = torch.tensor(sl).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad(): o = denoiser(t).squeeze().cpu().numpy()
            ospec = np.expm1(np.maximum(o, 0))
            phase = np.angle(stft(chunk))
            den = np.clip(istft(ospec, phase), -1.0, 1.0)
            feat_d = extract_features(normalize_audio(den))
            np.save(os.path.join(den_dir, name), feat_d)
            records.append({"file": os.path.join(out_dir, name), "label": 5, "class": "noise"})
    except Exception as e:
        print(f"  Skipping {os.path.basename(fp)}: {e}")
    if (i+1) % 100 == 0: print(f"  {i+1}/{len(files)}")

with open(os.path.join(BASE, "data", "processed", "noise_metadata.json"), "w") as f:
    json.dump(records, f, indent=2)
print(f"Done: {len(records)} noise samples")
