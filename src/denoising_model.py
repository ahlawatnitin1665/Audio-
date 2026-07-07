import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        return x


class TwoStagePipeline:
    def __init__(self, denoiser_path=None, classifier_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model import PatientAudioCNN, NUM_CLASSES
        from preprocessing import extract_features
        
        self.denoiser = AudioDenoiser().to(self.device)
        self.classifier = PatientAudioCNN(NUM_CLASSES).to(self.device)
        
        if denoiser_path and torch.path.exists(denoiser_path):
            self.denoiser.load_state_dict(torch.load(denoiser_path, map_location=self.device))
            print(f"Loaded denoiser from {denoiser_path}")
        
        if classifier_path and torch.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
            print(f"Loaded classifier from {classifier_path}")
        
        self.denoiser.eval()
        self.classifier.eval()
        
        self.class_names = ["coughing", "crying", "groaning", "gasping", "normal", "noise"]
    
    def denoise_audio(self, audio, sr=16000):
        from preprocessing import normalize_audio, extract_features
        
        audio = normalize_audio(audio)
        features = extract_features(audio)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            denoised_features = self.denoiser(features.to(self.device))
        
        denoised_features = denoised_features.squeeze(0).cpu().numpy()
        return denoised_features
    
    def classify_audio(self, audio_features):
        x = torch.tensor(audio_features, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        
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
