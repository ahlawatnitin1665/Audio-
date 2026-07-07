import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import load_audio, normalize_audio, extract_features
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "vocal_bursts")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
label_map = {"Moan": 2, "Pant": 3}

existing = set(os.listdir(PROCESSED_DIR))
metadata_path = os.path.join(PROCESSED_DIR, "metadata.json")
with open(metadata_path) as f:
    records = json.load(f)

for class_name, label_id in label_map.items():
    class_dir = os.path.join(RAW_DIR, class_name)
    audio_files = list(Path(class_dir).rglob("*.flac"))
    print(f"Processing {class_name}: {len(audio_files)} files")
    
    for i, audio_path in enumerate(audio_files):
        out_fname = f"{class_name}_{i:05d}.npy"
        if out_fname in existing:
            continue
        try:
            y = load_audio(str(audio_path))
            y = normalize_audio(y)
            features = extract_features(y)
            out_path = os.path.join(PROCESSED_DIR, out_fname)
            np.save(out_path, features)
            records.append({
                "file": out_path,
                "label": label_id,
                "class": class_name,
            })
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(audio_files)}")
        except Exception as e:
            print(f"  Error: {audio_path}: {e}")

with open(metadata_path, "w") as f:
    json.dump(records, f)

classes = {}
for r in records:
    classes[r["class"]] = classes.get(r["class"], 0) + 1
print(f"Done. Total: {len(records)}, Distribution: {classes}")
