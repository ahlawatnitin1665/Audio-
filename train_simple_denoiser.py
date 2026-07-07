import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simple_denoiser import SimpleDenoiser
from src.preprocessing import load_audio, normalize_audio, TARGET_SR, DURATION


class NoisyAudioDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, max_samples=None):
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) 
                                  if f.endswith('.flac') or f.endswith('.wav')])
        self.noise_files = sorted([os.path.join(noise_dir, f) for f in os.listdir(noise_dir) 
                                 if f.endswith('.flac') or f.endswith('.wav')])
        
        if max_samples:
            self.clean_files = self.clean_files[:max_samples]
            self.noise_files = self.noise_files[:max_samples]
        
        print(f"Dataset: {len(self.clean_files)} clean files, {len(self.noise_files)} noise files")
    
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean_audio = load_audio(self.clean_files[idx], sr=TARGET_SR, duration=DURATION)
        noise_audio = load_audio(self.noise_files[idx], sr=TARGET_SR, duration=DURATION)
        
        mixed_audio = clean_audio + noise_audio
        mixed_audio = normalize_audio(mixed_audio)
        clean_audio = normalize_audio(clean_audio)
        
        return torch.tensor(mixed_audio, dtype=torch.float32).unsqueeze(0), torch.tensor(clean_audio, dtype=torch.float32).unsqueeze(0)


def train_denoiser(model, dataloader, epochs=30, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for mixed, clean in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            mixed = mixed.to(device)
            clean = clean.to(device)
            
            optimizer.zero_grad()
            output = model(mixed)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def main():
    os.makedirs("checkpoints", exist_ok=True)
    
    clean_dir = "data/raw/vocal_bursts"
    noise_dir = "data/noise_samples"  # Create this directory with machine noise
    
    dataset = NoisyAudioDataset(clean_dir, noise_dir, max_samples=2000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    denoiser = SimpleDenoiser()
    denoiser = train_denoiser(denoiser, dataloader, epochs=20)
    
    denoiser_path = "checkpoints/simple_denoiser.pth"
    torch.save(denoiser.state_dict(), denoiser_path)
    print(f"Denoiser saved to {denoiser_path}")


if __name__ == "__main__":
    main()
