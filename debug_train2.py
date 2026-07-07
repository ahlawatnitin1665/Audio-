import json, torch, numpy as np, sys
sys.path.insert(0, "src")
from model import PatientAudioCNN, NUM_CLASSES

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PatientAudioCNN(NUM_CLASSES).to(device)

# Load the partly-trained model
ckpt = torch.load("checkpoints/best_model.pth", map_location=device)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

val = json.load(open("data/splits/val.json"))
preds = {i:0 for i in range(5)}
true_counts = {i:0 for i in range(5)}
correct = 0
for f in val:
    x = np.load(f["file"])
    x = torch.tensor(x, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax().item()
    preds[pred] += 1
    true_counts[f["label"]] += 1
    if pred == f["label"]:
        correct += 1

total = len(val)
print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
print(f"True labels: {dict(true_counts)}")
print(f"Predictions: {dict(preds)}")
print(f"If predictions match true distribution, model is learning distribution not features")
