import os
import json
import soundfile as sf
from datasets import load_dataset
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "data", "raw", "vocal_bursts")
os.makedirs(output_dir, exist_ok=True)

LABEL_MAP = {
    "cough": ("Cough", 0),
    "coughing": ("Cough", 0),
    "crying": ("Crying", 1),
    "moaning": ("Moan", 2),
    "sigh": ("Moan", 2),
    "panting": ("Pant", 3),
    "breath": ("Pant", 3),
    "breathing": ("Pant", 3),
}

print("Loading Vocal Bursts dataset from HuggingFace...")
ds = load_dataset("0x3/vocal-bursts", split="train")
print(f"Total samples in dataset: {len(ds)}")

for class_name in set(v[0] for v in LABEL_MAP.values()):
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

existing = {}
for class_name in set(v[0] for v in LABEL_MAP.values()):
    class_dir = os.path.join(output_dir, class_name)
    existing[class_name] = set(os.listdir(class_dir)) if os.path.exists(class_dir) else set()

counts = {v[0]: 0 for v in LABEL_MAP.values()}
skipped = 0
errors = 0

for i in range(len(ds)):
    sample = ds[i]
    meta = sample.get("json")
    if meta is None:
        continue

    label = meta.get("label", "")
    if label not in LABEL_MAP:
        continue

    class_name, label_id = LABEL_MAP[label]
    audio = sample.get("audio")
    if audio is None:
        errors += 1
        continue

    out_fname = f"{class_name}_{i:06d}.wav"
    if out_fname in existing.get(class_name, set()):
        skipped += 1
        continue

    try:
        audio_path = os.path.join(output_dir, class_name, out_fname)
        sf.write(audio_path, audio["array"], audio["sampling_rate"])
        counts[class_name] += 1
    except Exception as e:
        errors += 1

    if (i + 1) % 5000 == 0:
        print(f"  Processed {i + 1}/{len(ds)}...")

print(f"\nDownload complete!")
print(f"New samples saved:")
for cls, cnt in sorted(counts.items()):
    print(f"  {cls}: {cnt}")
print(f"Skipped (already exist): {skipped}")
print(f"Errors: {errors}")

print("\nTotal files per class now:")
for class_name in set(v[0] for v in LABEL_MAP.values()):
    class_dir = os.path.join(output_dir, class_name)
    n = len([f for f in os.listdir(class_dir) if f.endswith('.wav') or f.endswith('.flac')])
    print(f"  {class_name}: {n}")
