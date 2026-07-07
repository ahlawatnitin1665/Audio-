import json, torch, numpy as np, sys
sys.path.insert(0, "src")
from model import PatientAudioCNN, NUM_CLASSES

model = PatientAudioCNN(NUM_CLASSES)
model.eval()
# Count untrained model predictions
val = json.load(open("data/splits/val.json"))
counts = {i:0 for i in range(5)}
true_counts = {i:0 for i in range(5)}
for f in val:
    x = np.load(f["file"])
    x = torch.tensor(x, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).argmax().item()
    counts[pred] += 1
    true_counts[f["label"]] += 1
print("True labels:", dict(true_counts))
print("Untrained predictions:", dict(counts))
print("If predictions are ~uniform (20% each), random init is fine")
