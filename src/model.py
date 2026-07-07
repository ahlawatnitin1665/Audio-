import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 6


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        shortcut = []
        if stride != 1 or in_ch != out_ch:
            shortcut.append(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False))
            shortcut.append(nn.BatchNorm2d(out_ch))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class PatientAudioCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.block1 = ResidualBlock(3, 96, stride=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block2 = ResidualBlock(96, 192, stride=2)

        self.block3 = ResidualBlock(192, 384, stride=2)

        self.block4 = nn.Sequential(
            nn.Conv2d(384, 768, 3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(768, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.block4(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return probs


class DistressCalculator:
    def __init__(self):
        self.weights = {
            0: 0.4,   # coughing
            1: 0.8,   # crying
            2: 0.6,   # groaning
            3: 1.0,   # gasping
            4: 0.0,   # normal
            5: 0.0,   # noise
        }
        self.class_names = ["coughing", "crying", "groaning", "gasping", "normal", "noise"]

    def calculate(self, probabilities):
        score = 0.0
        for cls_id, weight in self.weights.items():
            if cls_id < len(probabilities):
                score += probabilities[cls_id].item() * weight
        return min(score, 1.0)

    def get_alert_level(self, score, high_threshold=0.7, moderate_threshold=0.4):
        if score > high_threshold:
            return "HIGH DISTRESS", "URGENT"
        elif score > moderate_threshold:
            return "MODERATE DISTRESS", "WARNING"
        else:
            return "Normal", "OK"

    def get_dominant_class(self, probabilities):
        pred_class = torch.argmax(probabilities).item()
        return self.class_names[pred_class], probabilities[pred_class].item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PatientAudioCNN(NUM_CLASSES)
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 3, 64, 94)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")

    probs = torch.softmax(out, dim=1)
    calc = DistressCalculator()
    for i in range(2):
        score = calc.calculate(probs[i])
        level, priority = calc.get_alert_level(score)
        dominant, conf = calc.get_dominant_class(probs[i])
        print(f"  Sample {i}: score={score:.3f}, level={level}, dominant={dominant} ({conf:.2%})")
