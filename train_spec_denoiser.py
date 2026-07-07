import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TARGET_SR = 16000
DURATION = 2.0
N_SAMPLES = int(TARGET_SR * DURATION)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

N_FFT = 512
HOP_LENGTH = 128
N_FREQ = N_FFT // 2 + 1
N_FRAMES = N_SAMPLES // HOP_LENGTH + 1

VOCAL_WAV_DIR = "data/vocal_16k"
CLASS_MAP = {"Cough": "coughing", "Crying": "crying", "Moan": "groaning", "Pant": "gasping", "Normal": "normal"}
AUDIO_EXT = ('.wav', '.flac')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NOISE_CACHE = "data/noise_cache"


def stft(y):
    return librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)


def istft(mag, phase):
    complex_spec = mag * np.exp(1j * phase)
    return librosa.istft(complex_spec, hop_length=HOP_LENGTH)


class SpecUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.enc1_bn = nn.BatchNorm2d(32)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc2_bn = nn.BatchNorm2d(64)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc3_bn = nn.BatchNorm2d(128)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc4_bn = nn.BatchNorm2d(256)

        self.dec4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec4_bn = nn.BatchNorm2d(128)
        self.dec3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.dec3_bn = nn.BatchNorm2d(64)
        self.dec2 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.dec2_bn = nn.BatchNorm2d(32)
        self.dec1 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        e1 = F.relu(self.enc1_bn(self.enc1(x)))
        e2 = F.relu(self.enc2_bn(self.enc2(e1)))
        e3 = F.relu(self.enc3_bn(self.enc3(e2)))
        e4 = F.relu(self.enc4_bn(self.enc4(e3)))

        d4 = F.relu(self.dec4_bn(self.dec4(e4)))
        d4 = torch.cat([self._pad_to(d4, e3), e3], dim=1)

        d3 = F.relu(self.dec3_bn(self.dec3(d4)))
        d3 = torch.cat([self._pad_to(d3, e2), e2], dim=1)

        d2 = F.relu(self.dec2_bn(self.dec2(d3)))
        d2 = torch.cat([self._pad_to(d2, e1), e1], dim=1)

        out = self.dec1(d2)
        out = self._pad_to(out, x)
        return out

    def _pad_to(self, x, target):
        diff_h = target.shape[2] - x.shape[2]
        diff_w = target.shape[3] - x.shape[3]
        if diff_h > 0 or diff_w > 0:
            x = F.pad(x, (0, diff_w, 0, diff_h))
        elif diff_h < 0 or diff_w < 0:
            x = x[:, :, :target.shape[2], :target.shape[3]]
        return x


def preconvert_noise_to_wav():
    noise_dirs = [
        os.path.join(BASE_DIR, "archive", "Hospital noise original", "Hospital noise original"),
        os.path.join(BASE_DIR, "archive", "icu_machine_sounds", "sounddino_hospital"),
        os.path.join(BASE_DIR, "archive", "icu_machine_sounds", "sounddino_heart"),
        os.path.join(BASE_DIR, "archive", "icu_machine_sounds", "bigsoundbank"),
    ]
    os.makedirs(NOISE_CACHE, exist_ok=True)
    all_noise = []
    for nd in noise_dirs:
        if os.path.isdir(nd):
            files = [os.path.join(nd, f) for f in os.listdir(nd) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
            for nf in tqdm(files, desc=f"Caching {os.path.basename(nd)}"):
                base = os.path.splitext(os.path.basename(nf))[0]
                out = os.path.join(NOISE_CACHE, base + ".wav")
                if not os.path.exists(out):
                    try:
                        y, sr = librosa.load(nf, sr=TARGET_SR, mono=True)
                        sf.write(out, y, TARGET_SR)
                    except Exception:
                        continue
                all_noise.append(out)
    print(f"  Total cached noise files: {len(all_noise)}")
    return all_noise


class SpecDenoisingDataset(Dataset):
    def __init__(self, clean_files, noise_files):
        self.clean_files = clean_files
        self.noise_files = noise_files

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        try:
            clean_audio, sr = sf.read(clean_path)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.clean_files))

        if len(clean_audio) > N_SAMPLES:
            start = np.random.randint(0, len(clean_audio) - N_SAMPLES)
            clean_audio = clean_audio[start:start + N_SAMPLES]
        elif len(clean_audio) < N_SAMPLES:
            clean_audio = np.pad(clean_audio, (0, N_SAMPLES - len(clean_audio)))
        clean_audio = clean_audio.astype(np.float32)
        peak = np.max(np.abs(clean_audio))
        if peak > 0:
            clean_audio = clean_audio / peak

        noise_path = self.noise_files[np.random.randint(0, len(self.noise_files))]
        noise_audio, _ = sf.read(noise_path)
        if len(noise_audio) > N_SAMPLES:
            start = np.random.randint(0, len(noise_audio) - N_SAMPLES)
            noise_audio = noise_audio[start:start + N_SAMPLES]
        elif len(noise_audio) < N_SAMPLES:
            noise_audio = np.pad(noise_audio, (0, N_SAMPLES - len(noise_audio)))
        noise_audio = noise_audio.astype(np.float32)
        peak = np.max(np.abs(noise_audio))
        if peak > 0:
            noise_audio = noise_audio / peak

        snr_db = np.random.uniform(-2, 8)
        signal_power = np.mean(clean_audio ** 2) + 1e-8
        noise_power = np.mean(noise_audio ** 2) + 1e-8
        snr_linear = 10 ** (snr_db / 10)
        noise_scaled = noise_audio * np.sqrt(signal_power / (noise_power * snr_linear))

        mixed_audio = np.clip(clean_audio + noise_scaled, -1.0, 1.0)

        clean_spec = np.abs(stft(clean_audio)).astype(np.float32)
        mixed_spec = np.abs(stft(mixed_audio)).astype(np.float32)

        clean_spec = np.log1p(clean_spec)
        mixed_spec = np.log1p(mixed_spec)

        mixed_spec = torch.tensor(mixed_spec, dtype=torch.float32).unsqueeze(0)
        clean_spec = torch.tensor(clean_spec, dtype=torch.float32).unsqueeze(0)

        return mixed_spec, clean_spec


def collect_clean_files():
    all_files = []
    for class_dir in CLASS_MAP.keys():
        dir_path = os.path.join(VOCAL_WAV_DIR, class_dir)
        if os.path.exists(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                     if f.lower().endswith(AUDIO_EXT)]
            all_files.extend(files)
            print(f"  {class_dir} ({CLASS_MAP[class_dir]}): {len(files)} files")
    return all_files


def train_denoiser(model, train_loader, val_loader):
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for mixed, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            mixed = mixed.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()
            output = model(mixed)
            loss = criterion(output, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mixed, clean in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                mixed = mixed.to(device)
                clean = clean.to(device)
                output = model(mixed)
                loss = criterion(output, clean)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS}: LR={lr:.6f}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/spec_denoiser.pth")
            print(f"  Saved best model (val_loss={avg_val_loss:.6f})")

    return model


def main():
    os.makedirs("checkpoints", exist_ok=True)

    print("Collecting clean vocal audio files...")
    clean_files = collect_clean_files()
    print(f"\nTotal clean files: {len(clean_files)}")

    if len(clean_files) == 0:
        print("No audio files found!")
        return

    print("\nPre-converting noise files to 16kHz WAV (one-time)...")
    noise_files = preconvert_noise_to_wav()
    if len(noise_files) == 0:
        print("No noise files found!")
        return

    train_files, val_files = train_test_split(clean_files, test_size=0.15, random_state=42)
    print(f"\nTrain: {len(train_files)} files, Val: {len(val_files)} files")

    train_dataset = SpecDenoisingDataset(train_files, noise_files)
    val_dataset = SpecDenoisingDataset(val_files, noise_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SpecUNet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_denoiser(model, train_loader, val_loader)

    print(f"\nTraining complete! Best model saved to checkpoints/spec_denoiser.pth")


if __name__ == "__main__":
    main()
