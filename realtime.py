import os, sys, time, numpy as np, torch, sounddevice as sd, requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from patient_config import select_patient
from model import PatientAudioCNN, NUM_CLASSES, DistressCalculator
from preprocessing import extract_features, normalize_audio, TARGET_SR, N_SAMPLES
from train_spec_denoiser import SpecUNet, stft, istft

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SEC = 3
HOP_SEC = 1.5
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SEC)
HOP_SAMPLES = int(TARGET_SR * HOP_SEC)

LOG_FILE = os.path.join(BASE, "audio_log.jsonl")

PATIENT_ID = select_patient("AUDIO MONITORING")
BACKEND_URL = f"http://172.20.10.2:8000/api/audio/{PATIENT_ID}"


def send_to_backend(audio_features):
    import json
    print(f"  [JSON] {audio_features}")
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(audio_features) + "\n")
    try:
        requests.post(BACKEND_URL, json=audio_features, timeout=1)
    except Exception as e:
        print(f"  [Backend Error] {e}")

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
print(f"Loaded models on {DEVICE}")

def classify_chunk(audio_chunk):
    feats = extract_features(normalize_audio(audio_chunk))
    x = torch.tensor(feats, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(classifier(x), dim=1).squeeze()
    pred = torch.argmax(probs).item()
    conf = probs[pred].item()
    score = distress.calculate(probs)
    level, priority = distress.get_alert_level(score)
    return class_names[pred], conf, score, level, priority, probs.cpu().numpy()

def denoise_chunk(audio_chunk):
    spec = np.abs(stft(audio_chunk)).astype(np.float32)
    spec_log = np.log1p(spec)
    t = torch.tensor(spec_log, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = denoiser(t).squeeze().cpu().numpy()
    out_spec = np.expm1(np.maximum(out, 0))
    phase = np.angle(stft(audio_chunk))
    return np.clip(istft(out_spec, phase), -1.0, 1.0)

NOISE_PROBS = np.array([0.0] * len(class_names))
NOISE_PROBS[5] = 1.0

def is_speech(chunk, den_chunk, pre_norm_rms=None, pre_norm_peak=None):
    raw_rms = pre_norm_rms if pre_norm_rms is not None else np.sqrt(np.mean(chunk**2))
    peak = pre_norm_peak if pre_norm_peak is not None else np.max(np.abs(chunk))
    crest = peak / max(raw_rms, 1e-8)
    decision = raw_rms < 0.003 and crest < 4.0
    print(f"  [GATE] raw_rms={raw_rms:.5f}  crest={crest:.1f}  is_speech={not decision}")
    return not decision

def process_chunk(raw_chunk, chunk_idx, pre_norm_rms=None, pre_norm_peak=None):
    den = denoise_chunk(raw_chunk)

    if not is_speech(raw_chunk, den, pre_norm_rms=pre_norm_rms, pre_norm_peak=pre_norm_peak):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now}]")
        print(f"  Distress Score: 0.000")
        print(f"  Alert: Normal (OK)")
        print(f"  Dominant: noise (100.0%)")
        print(f"  Probs: " + " ".join(f"{class_names[i]}={NOISE_PROBS[i]:.2f}" for i in range(len(class_names))))
        send_to_backend({
            "patient_id": PATIENT_ID,
            "timestamp_unix": time.time(),
            "distress_score": 0.0,
            "alert_level": "Normal",
            "priority": "OK",
            "dominant_class": "noise",
            "confidence": 1.0,
            "probabilities": {class_names[i]: float(NOISE_PROBS[i]) for i in range(len(class_names))},
        })
        return 0.0, "Normal"

    pred_d, conf_d, score_d, lvl_d, pri_d, probs_d = classify_chunk(den)

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    prob_str = " ".join(f"{class_names[i]}={probs_d[i]:.2f}" for i in range(len(class_names)))

    print(f"\n[{now}]")
    print(f"  Distress Score: {score_d:.3f}")
    print(f"  Alert: {lvl_d} ({pri_d})")
    print(f"  Dominant: {pred_d} ({conf_d:.1%})")
    print(f"  Probs: {prob_str}")

    send_to_backend({
        "patient_id": PATIENT_ID,
        "timestamp_unix": time.time(),
        "distress_score": float(score_d),
        "alert_level": lvl_d,
        "priority": pri_d,
        "dominant_class": pred_d,
        "confidence": float(conf_d),
        "probabilities": {class_names[i]: float(probs_d[i]) for i in range(len(class_names))},
    })

    return score_d, lvl_d

print(f"\n{'='*50}")
print(f"  REAL-TIME MIC MONITORING")
print(f"  Chunk: {CHUNK_SEC}s | Overlap: {HOP_SEC}s")
print(f"  Press Ctrl+C to stop")
print(f"{'='*50}\n")

buffer = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
write_pos = 0
chunk_idx = 0
scores = []

def audio_callback(indata, frames, time_info, status):
    global write_pos, chunk_idx, buffer, scores
    if status:
        print(f"  [audio status: {status}]")

    samples = indata[:, 0].copy()
    space = CHUNK_SAMPLES - write_pos
    if len(samples) <= space:
        buffer[write_pos:write_pos + len(samples)] = samples
        write_pos += len(samples)
    else:
        buffer[write_pos:write_pos + space] = samples[:space]
        chunk = buffer.copy()
        chunk_idx += 1

        raw = chunk.astype(np.float32)
        raw_rms = np.sqrt(np.mean(raw**2))
        peak = max(np.max(np.abs(raw)), 1e-8)
        raw /= peak

        score, level = process_chunk(raw, chunk_idx, pre_norm_rms=raw_rms, pre_norm_peak=peak)
        scores.append(score)

        leftover = samples[space:]
        buffer[:len(leftover)] = leftover
        write_pos = len(leftover)

try:
    with sd.InputStream(
        device=1,
        channels=1,
        samplerate=TARGET_SR,
        blocksize=int(TARGET_SR * 0.5),
        callback=audio_callback,
    ):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    if scores:
        avg = np.mean(scores)
        max_ = max(scores)
        alerts = sum(1 for s in scores if s > 0.4)
        print(f"\n{'='*50}")
        print(f"  SESSION SUMMARY")
        print(f"{'='*50}")
        print(f"  Total chunks: {len(scores)}")
        print(f"  Avg distress: {avg:.4f}")
        print(f"  Max distress: {max_:.4f}")
        print(f"  Alerts: {alerts}/{len(scores)}")
        if avg > 0.7:
            print(f"  VERDICT: HIGH DISTRESS")
        elif avg > 0.4:
            print(f"  VERDICT: MODERATE DISTRESS")
        elif avg > 0.1:
            print(f"  VERDICT: MILD DISTRESS")
        else:
            print(f"  VERDICT: NORMAL")
        print(f"{'='*50}")
