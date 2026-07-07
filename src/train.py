import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES
from model import PatientAudioCNN, DistressCalculator, count_parameters


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 80
PATIENCE = 15
MIXUP_ALPHA = 0.6
FOCAL_GAMMA = 3.0
MAX_SAMPLES_PER_CLASS = 6000
GRAD_CLIP_NORM = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def compute_class_weights(records):
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    total = sum(counts.values())
    weights = []
    for cls_id in range(NUM_CLASSES):
        w = total / (NUM_CLASSES * counts[cls_id])
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in tqdm(loader, desc="Training", leave=False):
        features, labels = features.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        use_mixup = np.random.random() < 0.5 and MIXUP_ALPHA > 0
        if use_mixup:
            features, y_a, y_b, lam = mixup_data(features, labels)
            outputs = model(features)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            _, predicted = torch.max(outputs, 1)
            correct += (lam * (predicted == y_a).float() + (1 - lam) * (predicted == y_b).float()).sum().item()
        else:
            outputs = model(features)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        running_loss += loss.item() * features.size(0)
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Validating", leave=False):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    denoised_dir = os.path.join(BASE_DIR, "data", "processed_denoised")
    train_loader, val_loader, test_loader = get_dataloaders(SPLITS_DIR, BATCH_SIZE, max_per_class=MAX_SAMPLES_PER_CLASS, denoised_dir=denoised_dir)

    class_counts = np.bincount([r["label"] for r in train_loader.dataset.records], minlength=NUM_CLASSES)
    class_weights = torch.tensor(
        [sum(class_counts) / (NUM_CLASSES * c) for c in class_counts],
        dtype=torch.float32
    ).to(DEVICE)
    print(f"Class weights (capped): {class_weights.cpu().numpy()}")
    print(f"Capped class counts: {class_counts}")

    model = PatientAudioCNN(NUM_CLASSES).to(DEVICE)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.2e}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
            }, checkpoint_path)
            print(f"  Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))


if __name__ == "__main__":
    train()
