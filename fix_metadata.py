import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
label_map = {"Cough": 0, "Crying": 1, "Moan": 2, "Pant": 3}

records = []
for fname in os.listdir(PROCESSED_DIR):
    if not fname.endswith(".npy"):
        continue
    parts = fname.replace(".npy", "").rsplit("_", 1)
    class_name = parts[0]
    if class_name in label_map:
        records.append({
            "file": os.path.join(PROCESSED_DIR, fname),
            "label": label_map[class_name],
            "class": class_name,
        })

print(f"Total records: {len(records)}")
classes = {}
for r in records:
    classes[r["class"]] = classes.get(r["class"], 0) + 1
print(f"Class distribution: {classes}")

with open(os.path.join(PROCESSED_DIR, "metadata.json"), "w") as f:
    json.dump(records, f)
print("metadata.json saved")
