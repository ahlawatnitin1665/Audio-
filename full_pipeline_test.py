import os, sys, json
import torch
import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model import PatientAudioCNN, NUM_CLASSES, DistressCalculator
from preprocessing import extract_features, normalize_audio, TARGET_SR, DURATION, N_SAMPLES
from train_spec_denoiser import SpecUNet, stft, istft

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FFT, HOP_LENGTH = 512, 128

# Load models
denoiser = SpecUNet().to(DEVICE)
denoiser.load_state_dict(torch.load(os.path.join(BASE, "checkpoints", "spec_denoiser.pth"), map_location=DEVICE))
denoiser.eval()

classifier = PatientAudioCNN(NUM_CLASSES).to(DEVICE)
ckpt = torch.load(os.path.join(BASE, "checkpoints", "best_model.pth"), map_location=DEVICE)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    classifier.load_state_dict(ckpt["model_state_dict"])
else:
    classifier.load_state_dict(ckpt)
classifier.eval()

distress = DistressCalculator()
class_names = ["coughing", "crying", "groaning", "gasping", "normal"]

print("Models loaded. Processing test samples...")

# Collect noise files
noise_dirs = [
    "archive/Hospital noise original/Hospital noise original",
    "archive/icu_machine_sounds/sounddino_hospital",
    "archive/icu_machine_sounds/sounddino_heart",
    "archive/icu_machine_sounds/bigsoundbank",
]
noise_files = []
for nd in noise_dirs:
    p = os.path.join(BASE, nd)
    if os.path.isdir(p):
        noise_files.extend([os.path.join(p, f) for f in os.listdir(p)])

# Get clean vocal files (one per class)
vocal_dir = os.path.join(BASE, "data", "vocal_16k")
test_files = []
for cls_name, label in [("Cough", 0), ("Crying", 1), ("Moan", 2), ("Pant", 3), ("Normal", 4)]:
    d = os.path.join(vocal_dir, cls_name)
    if os.path.isdir(d):
        files = sorted([f for f in os.listdir(d) if f.endswith(".wav")])
        if files:
            test_files.append((os.path.join(d, files[0]), cls_name, label))

out_dir = os.path.join(BASE, "pipeline_output")
os.makedirs(out_dir, exist_ok=True)

results = []
for clean_path, cls_name, label in test_files:
    clean, sr = librosa.load(clean_path, sr=TARGET_SR, mono=True)
    if len(clean) < N_SAMPLES:
        clean = np.pad(clean, (0, N_SAMPLES - len(clean)))
    else:
        clean = clean[:N_SAMPLES]
    clean = clean.astype(np.float32)
    peak = np.max(np.abs(clean))
    if peak > 0: clean /= peak

    noise_path = noise_files[np.random.randint(len(noise_files))]
    noise, _ = librosa.load(noise_path, sr=TARGET_SR, mono=True)
    if len(noise) > N_SAMPLES:
        start = np.random.randint(0, len(noise) - N_SAMPLES)
        noise = noise[start:start + N_SAMPLES]
    else:
        noise = np.pad(noise, (0, N_SAMPLES - len(noise)))
    noise = noise.astype(np.float32)
    peak = np.max(np.abs(noise))
    if peak > 0: noise /= peak

    snr_db = 3
    sp = np.mean(clean**2) + 1e-8
    np_ = np.mean(noise**2) + 1e-8
    noise_scaled = noise * np.sqrt(sp / (np_ * 10**(snr_db/10)))
    mixed = np.clip(clean + noise_scaled, -1.0, 1.0)

    # Stage 1: Denoise
    mixed_spec = np.abs(stft(mixed)).astype(np.float32)
    mixed_spec_log = np.log1p(mixed_spec)
    mixed_t = torch.tensor(mixed_spec_log, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        clean_spec_log = denoiser(mixed_t).squeeze().cpu().numpy()
    clean_spec = np.expm1(np.maximum(clean_spec_log, 0))
    mixed_phase = np.angle(stft(mixed))
    denoised = np.clip(istft(clean_spec, mixed_phase), -1.0, 1.0)

    sf.write(os.path.join(out_dir, f"{cls_name}_clean.wav"), clean, TARGET_SR)
    sf.write(os.path.join(out_dir, f"{cls_name}_noisy.wav"), mixed, TARGET_SR)
    sf.write(os.path.join(out_dir, f"{cls_name}_denoised.wav"), denoised, TARGET_SR)

    # Stage 2: Classify on denoised + baseline on noisy
    for tag, aud in [("noisy", mixed), ("denoised", denoised)]:
        aud_feats = extract_features(normalize_audio(aud))
        x = torch.tensor(aud_feats, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(classifier(x), dim=1).squeeze()
        pred_class = torch.argmax(probs).item()
        score = distress.calculate(probs)
        level, priority = distress.get_alert_level(score)

        results.append({
            "true_class": cls_name,
            "true_label": label,
            "audio_type": tag,
            "pred_class": class_names[pred_class],
            "pred_label": pred_class,
            "correct": pred_class == label,
            "confidence": probs[pred_class].item(),
            "distress_score": score,
            "alert_level": level,
        })

# Print results
print(f"\n{'True':<10} {'Type':<10} {'Predicted':<10} {'Correct':<8} {'Conf':<8} {'Distress':<8} {'Alert':<15}")
print("-" * 75)
correct_noisy = 0
correct_den = 0
total = 0
for r in results:
    print(f"{r['true_class']:<10} {r['audio_type']:<10} {r['pred_class']:<10} {'✓' if r['correct'] else '✗':<8} {r['confidence']:.2%}  {r['distress_score']:.2f}   {r['alert_level']:<15}")
    total += 1
    if r['audio_type'] == 'noisy' and r['correct']: correct_noisy += 1
    if r['audio_type'] == 'denoised' and r['correct']: correct_den += 1

n = total // 2
print(f"\nBaseline (noisy → classify): {correct_noisy}/{n} = {correct_noisy/n*100:.1f}%")
print(f"Pipeline  (denoise → classify): {correct_den}/{n} = {correct_den/n*100:.1f}%")
print(f"\nFiles saved to {out_dir}/ for listening")
