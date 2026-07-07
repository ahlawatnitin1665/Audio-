import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel//2)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, stride=2):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=kernel//2, output_padding=stride-1)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x, skip=None):
        x = torch.relu(self.bn(self.conv(x)))
        if skip is not None:
            if x.shape[2] != skip.shape[2]:
                diff = skip.shape[2] - x.shape[2]
                x = F.pad(x, (0, diff))
            x = x + skip
        return x


class AudioUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(1, 32, kernel=7, stride=1)
        self.enc2 = EncoderBlock(32, 64, kernel=5, stride=2)
        self.enc3 = EncoderBlock(64, 128, kernel=5, stride=2)
        self.enc4 = EncoderBlock(128, 256, kernel=5, stride=2)

        self.dec4 = DecoderBlock(256, 128, kernel=5, stride=2)
        self.dec3 = DecoderBlock(128, 64, kernel=5, stride=2)
        self.dec2 = DecoderBlock(64, 32, kernel=5, stride=2)
        self.dec1 = nn.Conv1d(32, 1, kernel_size=7, padding=3)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        out = self.dec1(d2)

        if out.shape[2] != x.shape[2]:
            out = out[:, :, :x.shape[2]]
        return out


SimpleDenoiser = AudioUNet


class SimpleTwoStagePipeline:
    def __init__(self, denoiser_path=None, classifier_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Proper import handling
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from model import PatientAudioCNN, NUM_CLASSES
        from preprocessing import extract_features
        
        self.denoiser = SimpleDenoiser().to(self.device)
        self.classifier = PatientAudioCNN(NUM_CLASSES).to(self.device)
        
        if denoiser_path and os.path.exists(denoiser_path):
            checkpoint = torch.load(denoiser_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.denoiser.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.denoiser.load_state_dict(checkpoint)
            print(f"Loaded denoiser from {denoiser_path}")
        
        if classifier_path and os.path.exists(classifier_path):
            checkpoint = torch.load(classifier_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.classifier.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.classifier.load_state_dict(checkpoint)
            print(f"Loaded classifier from {classifier_path}")
        
        self.denoiser.eval()
        self.classifier.eval()
        
        self.class_names = ["coughing", "crying", "groaning", "gasping", "normal", "noise"]
    
    def denoise_audio(self, audio, sr=16000):
        from preprocessing import normalize_audio
        
        audio = normalize_audio(audio)
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            denoised = self.denoiser(audio.to(self.device))
        
        denoised = denoised.squeeze(0).squeeze(0).cpu().numpy()
        return denoised
    
    def classify_audio(self, audio):
        from preprocessing import extract_features
        
        features = extract_features(audio)
        x = torch.tensor(features, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            probs = torch.softmax(self.classifier(x.to(self.device)), dim=1).squeeze()
        
        return probs
    
    def predict(self, audio, sr=16000):
        denoised_audio = self.denoise_audio(audio, sr)
        probs = self.classify_audio(denoised_audio)
        
        dominant_class = torch.argmax(probs).item()
        confidence = probs[dominant_class].item()
        
        return {
            "class": self.class_names[dominant_class],
            "confidence": confidence,
            "probabilities": {self.class_names[i]: probs[i].item() for i in range(len(self.class_names))},
            "denoised": denoised_audio
        }
    
    def save_models(self, denoiser_path, classifier_path):
        torch.save(self.denoiser.state_dict(), denoiser_path)
        torch.save(self.classifier.state_dict(), classifier_path)
        print(f"Models saved to {denoiser_path} and {classifier_path}")