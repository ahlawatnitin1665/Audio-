import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd


CLASS_NAMES = ["coughing", "crying", "groaning", "gasping", "normal", "noise"]
NUM_CLASSES = 6


def add_icu_noise(y, sr=16000):
    """Synthesize realistic ICU noise: hum, beeps, broadband machine noise"""
    noise = np.zeros_like(y)

    if np.random.random() < 0.5:
        t = np.linspace(0, len(y) / sr, len(y), endpoint=False)
        freq = np.random.choice([50, 60, 100, 120])
        hum = np.random.uniform(0.005, 0.02) * np.sin(2 * np.pi * freq * t)
        noise += hum

    if np.random.random() < 0.3:
        beep_len = int(np.random.uniform(0.05, 0.2) * sr)
        n_beeps = np.random.randint(1, 4)
        for _ in range(n_beeps):
            start = np.random.randint(0, max(1, len(y) - beep_len))
            freq = np.random.uniform(800, 2000)
            t = np.linspace(0, beep_len / sr, beep_len, endpoint=False)
            beep = np.random.uniform(0.01, 0.04) * np.sin(2 * np.pi * freq * t)
            beep *= np.hanning(beep_len)
            noise[start:start + beep_len] += beep

    if np.random.random() < 0.4:
        broadband = np.random.normal(0, np.random.uniform(0.003, 0.015), len(y))
        noise += broadband

    return y + noise


def augment_features(features):
    if np.random.random() < 0.5:
        noise = np.random.normal(0, 0.01, features.shape)
        features = features + noise
    if np.random.random() < 0.5:
        time_steps = features.shape[1]
        max_shift = time_steps // 10
        shift = np.random.randint(-max_shift, max_shift + 1)
        features = np.roll(features, shift, axis=1)
    if np.random.random() < 0.3:
        scale = np.random.uniform(0.8, 1.2)
        features = features * scale
    if np.random.random() < 0.5:
        time_steps = features.shape[1]
        mask_width = max(1, np.random.randint(1, time_steps // 6))
        start = np.random.randint(0, max(1, time_steps - mask_width))
        features[:, start:start + mask_width, :] = 0
    if np.random.random() < 0.5:
        n_mels = features.shape[0]
        freq_mask = max(1, np.random.randint(1, n_mels // 6))
        f_start = np.random.randint(0, max(1, n_mels - freq_mask))
        features[f_start:f_start + freq_mask, :, :] = 0
    return features


class VocalBurstDataset(Dataset):
    def __init__(self, records, augment=False, denoised_dir=None):
        self.records = records
        self.augment = augment
        self.denoised_dir = denoised_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        if self.augment and self.denoised_dir is not None and np.random.random() < 0.5:
            den_path = os.path.join(self.denoised_dir, os.path.basename(rec["file"]))
            if os.path.exists(den_path):
                features = np.load(den_path)
            else:
                features = np.load(rec["file"])
        else:
            features = np.load(rec["file"])
        if self.augment:
            features = augment_features(features)
        features = torch.tensor(features, dtype=torch.float32)
        features = features.permute(2, 0, 1)
        label = torch.tensor(rec["label"], dtype=torch.long)
        return features, label


def create_splits(processed_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    metadata_path = os.path.join(processed_dir, "metadata.json")
    with open(metadata_path) as f:
        records = json.load(f)

    train_records, temp_records = train_test_split(
        records, test_size=(1 - train_ratio), stratify=[r["label"] for r in records],
        random_state=42
    )
    relative_ratio = val_ratio / (val_ratio + test_ratio)
    val_records, test_records = train_test_split(
        temp_records, test_size=(1 - relative_ratio), stratify=[r["label"] for r in temp_records],
        random_state=42
    )

    splits_dir = os.path.join(os.path.dirname(processed_dir), "splits")
    os.makedirs(splits_dir, exist_ok=True)

    for name, split_records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        path = os.path.join(splits_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(split_records, f, indent=2)

    print(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    class_counts = {}
    for split_name, split_data in [("train", train_records), ("val", val_records), ("test", test_records)]:
        counts = {}
        for r in split_data:
            counts[r["class"]] = counts.get(r["class"], 0) + 1
        class_counts[split_name] = counts
        print(f"  {split_name}: {counts}")

    return train_records, val_records, test_records


def load_split(splits_dir, split_name):
    path = os.path.join(splits_dir, f"{split_name}.json")
    with open(path) as f:
        return json.load(f)


def get_dataloaders(splits_dir, batch_size=32, max_per_class=None, denoised_dir=None):
    train_records = load_split(splits_dir, "train")
    val_records = load_split(splits_dir, "val")
    test_records = load_split(splits_dir, "test")

    if max_per_class is not None:
        capped = []
        counts_per_class = {i: 0 for i in range(NUM_CLASSES)}
        for r in train_records:
            lbl = r["label"]
            if counts_per_class[lbl] < max_per_class:
                capped.append(r)
                counts_per_class[lbl] += 1
        train_records = capped
        print(f"Capped training to {max_per_class}/class: {len(train_records)} total")
        for i in range(NUM_CLASSES):
            print(f"  {CLASS_NAMES[i]}: {counts_per_class[i]}")

    train_dataset = VocalBurstDataset(train_records, augment=True, denoised_dir=denoised_dir)
    val_dataset = VocalBurstDataset(val_records, augment=False)
    test_dataset = VocalBurstDataset(test_records, augment=False)

    labels = [r["label"] for r in train_records]
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    train_records, val_records, test_records = create_splits(PROCESSED_DIR)
