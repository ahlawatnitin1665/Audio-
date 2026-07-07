import os, sys, argparse
import torch
import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, "src")
from model import PatientAudioCNN, NUM_CLASSES, DistressCalculator
from preprocessing import extract_features, normalize_audio, TARGET_SR, N_SAMPLES
from train_spec_denoiser import SpecUNet, stft, istft

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FFT, HOP = 512, 128
CHUNK_SEC = 3
HOP_SEC = 1.5  # 50% overlap
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SEC)
HOP_SAMPLES = int(TARGET_SR * HOP_SEC)

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="Path to audio file to process")
parser.add_argument("--noise", help="Path to noise file (optional, creates noisy mix)")
parser.add_argument("--summary", action="store_true", help="Show aggregate summary only")
parser.add_argument("--verbose", action="store_true", help="Show per-chunk detail with probabilities")
args = parser.parse_args()

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
class_names = ["coughing", "crying", "groaning", "gasping", "normal", "noise"]
print(f"Loaded models on {DEVICE}\n")

# Load audio
noise_dirs = [
    "archive/Hospital noise original/Hospital noise original",
    "archive/icu_machine_sounds/sounddino_hospital",
    "archive/icu_machine_sounds/sounddino_heart",
    "archive/icu_machine_sounds/bigsoundbank",
]

if args.file:
    audio, sr = librosa.load(args.file, sr=TARGET_SR, mono=True)
    print(f"Loaded: {args.file} ({len(audio)/TARGET_SR:.1f}s)")
else:
    cls_choices = ["Cough", "Crying", "Moan", "Pant", "Normal"]
    rand_cls = np.random.choice(cls_choices)
    cln_dir = os.path.join(BASE, "data", "vocal_16k", rand_cls)
    cln_files = [f for f in os.listdir(cln_dir) if f.endswith(".wav")]
    clean_path = os.path.join(cln_dir, np.random.choice(cln_files))
    audio, _ = librosa.load(clean_path, sr=TARGET_SR, mono=True)
    print(f"Clean: {rand_cls} ({os.path.basename(clean_path)})")

audio = audio.astype(np.float32)
peak = np.max(np.abs(audio))
if peak > 0: audio /= peak

# Mix with random ICU noise (skip if user provided own file without --noise flag)
if not args.file or args.noise:
    if args.noise:
        noise_file = args.noise
    else:
        noise_dir = np.random.choice(noise_dirs)
        noise_file = os.path.join(BASE, noise_dir, np.random.choice(os.listdir(os.path.join(BASE, noise_dir))))
    noise, _ = librosa.load(noise_file, sr=TARGET_SR, mono=True)
    noise = noise.astype(np.float32)
    peak_n = np.max(np.abs(noise))
    if peak_n > 0: noise /= peak_n
    sp = np.mean(audio**2) + 1e-8; np_ = np.mean(noise**2) + 1e-8
    noise_scaled = noise[:len(audio)] if len(noise) >= len(audio) else np.pad(noise, (0, len(audio)-len(noise)))
    noise_scaled *= np.sqrt(sp / (np_ * 10**(3/10)))
    audio = np.clip(audio + noise_scaled, -1.0, 1.0)
    print(f"Noise: {os.path.basename(noise_file)[:45]} (SNR=3dB)")

# ─── Helpers ────────────────────────────────────────────────────────
def classify_chunk(audio_chunk):
    feats = extract_features(normalize_audio(audio_chunk))
    x = torch.tensor(feats, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(classifier(x), dim=1).squeeze()
    pred = torch.argmax(probs).item()
    score = distress.calculate(probs)
    level, priority = distress.get_alert_level(score)
    return class_names[pred], probs[pred].item(), score, level, priority, probs.cpu().numpy()

def denoise_chunk(audio_chunk):
    spec = np.abs(stft(audio_chunk)).astype(np.float32)
    spec_log = np.log1p(spec)
    t = torch.tensor(spec_log, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = denoiser(t).squeeze().cpu().numpy()
    out_spec = np.expm1(np.maximum(out, 0))
    phase = np.angle(stft(audio_chunk))
    return np.clip(istft(out_spec, phase), -1.0, 1.0)

def has_voice(denoised_chunk):
    rms = np.sqrt(np.mean(denoised_chunk**2))
    return rms >= 0.008

# ─── Chunked processing ────────────────────────────────────────────
chunks = []
for start in range(0, len(audio), HOP_SAMPLES):
    chunk = audio[start:start + CHUNK_SAMPLES]
    if len(chunk) < TARGET_SR * 0.5:  # skip trailing <0.5s
        break
    if len(chunk) < CHUNK_SAMPLES:
        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
    chunks.append(chunk)

NOISE_PROBS = np.array([0.0] * len(class_names))
NOISE_PROBS[5] = 1.0

def is_speech(chunk, den_chunk):
    den_rms = np.sqrt(np.mean(den_chunk**2))
    if den_rms < 0.008:
        return False
    raw_rms = np.sqrt(np.mean(chunk**2))
    if raw_rms < 0.01:
        return False
    correlation = np.corrcoef(chunk[:len(den_chunk)], den_chunk[:len(chunk)])[0, 1]
    if correlation > 0.95:
        return False
    return True

all_results = []
for i, chunk in enumerate(chunks):
    time_sec = i * HOP_SEC
    den = denoise_chunk(chunk)

    if not is_speech(chunk, den):
        all_results.append({
            "time": time_sec, "chunk": chunk, "denoised": den,
            "raw_class": "noise", "raw_conf": 1.0, "raw_score": 0.0, "raw_level": "Normal",
            "den_class": "noise", "den_conf": 1.0, "den_score": 0.0, "den_level": "Normal",
            "raw_probs": NOISE_PROBS, "den_probs": NOISE_PROBS,
            "best_class": "noise", "best_conf": 1.0, "best_score": 0.0,
            "best_level": "Normal", "best_src": "energy", "best_probs": NOISE_PROBS,
        })
        continue

    pred_r, conf_r, score_r, lvl_r, pri_r, probs_r = classify_chunk(chunk)
    pred_d, conf_d, score_d, lvl_d, pri_d, probs_d = classify_chunk(den)
    use_raw = conf_r >= conf_d
    all_results.append({
        "time": time_sec, "chunk": chunk, "denoised": den,
        "raw_class": pred_r, "raw_conf": conf_r, "raw_score": score_r, "raw_level": lvl_r,
        "den_class": pred_d, "den_conf": conf_d, "den_score": score_d, "den_level": lvl_d,
        "raw_probs": probs_r, "den_probs": probs_d,
        "best_class": pred_r if use_raw else pred_d,
        "best_conf": conf_r if use_raw else conf_d,
        "best_score": score_r if use_raw else score_d,
        "best_level": lvl_r if use_raw else lvl_d,
        "best_src": "raw" if use_raw else "den",
        "best_probs": probs_r if use_raw else probs_d,
    })

# ─── Print results ─────────────────────────────────────────────────
if args.summary:
    classes_best = [r["best_class"] for r in all_results]
    n = len(all_results)
    print(f"\n{'='*60}")
    print(f"  AUDIO ANALYSIS REPORT ({n} chunks)")
    print(f"{'='*60}")
    print(f"\n  CLASS BREAKDOWN (denoised):")
    print(f"  {'─'*35}")
    for cls_name in class_names:
        cnt = sum(1 for r in all_results if r["den_class"] == cls_name)
        pct = cnt / n * 100
        bar = "█" * int(pct / 2)
        print(f"  {cls_name:<10} {cnt:>4d}/{n:<4d} ({pct:>5.1f}%)  {bar}")
    print(f"\n  DISTRESS SCORES:")
    print(f"  {'─'*35}")
    print(f"  Raw:      {np.mean([r['raw_score'] for r in all_results]):.2f}")
    print(f"  Denoised: {np.mean([r['den_score'] for r in all_results]):.2f}")
    print(f"  Best:     {np.mean([r['best_score'] for r in all_results]):.2f}")
    alert_count = sum(1 for r in all_results if r["best_level"] in ("HIGH DISTRESS", "MODERATE DISTRESS"))
    print(f"\n  ALERTS: {alert_count}/{n} chunks triggered")
    if alert_count > 0:
        high = sum(1 for r in all_results if r["best_level"] == "HIGH DISTRESS")
        mod = sum(1 for r in all_results if r["best_level"] == "MODERATE DISTRESS")
        print(f"          {high} HIGH, {mod} MODERATE")
    avg_score = np.mean([r['best_score'] for r in all_results])
    print(f"\n  FINAL VERDICT: ", end="")
    if avg_score > 0.7:
        print("HIGH DISTRESS — Patient likely in significant distress")
    elif avg_score > 0.4:
        print("MODERATE DISTRESS — Patient may need attention")
    elif avg_score > 0.1:
        print("MILD DISTRESS — Monitor for changes")
    else:
        print("NORMAL — No significant distress detected")
if args.verbose or not args.summary:
    print(f"\n{'='*60}")
    print(f"  PER-CHUNK ANALYSIS")
    print(f"{'='*60}")
    for r in all_results:
        probs = r["best_probs"]
        prob_str = " ".join(f"{class_names[i]}={probs[i]:.2f}" for i in range(len(class_names)))
        print(f"\n[{r['time']:>5.1f}s]")
        print(f"  Distress Score: {r['best_score']:.3f}")
        print(f"  Alert: {r['best_level']}")
        print(f"  Dominant: {r['best_class']} ({r['best_conf']:.1%})")
        print(f"  Probs: {prob_str}")
    avg = np.mean([r['best_score'] for r in all_results])
    max_ = max(r['best_score'] for r in all_results)
    alerts = sum(1 for r in all_results if r["best_level"] in ("HIGH DISTRESS", "MODERATE DISTRESS"))
    print(f"\nSummary: avg_distress={avg:.4f}, max_distress={max_:.4f}, alert_count={alerts}, total_chunks={len(all_results)}")

# ─── Save full audio ───────────────────────────────────────────────
out = os.path.join(BASE, "pipeline_output")
os.makedirs(out, exist_ok=True)
# Reconstruct full denoised audio
full_denoised = np.zeros(len(audio))
overlap_count = np.zeros(len(audio))
for r in all_results:
    start = int(r["time"] * TARGET_SR)
    end = start + len(r["denoised"])
    if end <= len(audio):
        full_denoised[start:end] += r["denoised"]
        overlap_count[start:end] += 1
overlap_count[overlap_count == 0] = 1
full_denoised /= overlap_count

sf.write(os.path.join(out, "input.wav"), audio, TARGET_SR)
sf.write(os.path.join(out, "denoised.wav"), full_denoised, TARGET_SR)
print(f"\nSaved to {out}/ — input.wav | denoised.wav")
