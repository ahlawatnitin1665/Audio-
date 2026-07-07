import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
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

VOCAL_WAV_DIR = "data/vocal_16k"
CLASS_MAP = {"Cough": "coughing", "Crying": "crying", "Moan": "groaning", "Pant": "gasping", "Normal": "normal"}
AUDIO_EXT = ('.wav', '.flac')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NOISE_DIRS = [
    os.path.join(BASE_DIR, "archive", "Hospital noise original", "Hospital noise original"),
    os.path.join(BASE_DIR, "archive", "icu_machine_sounds", "sounddino_hospital"),
    os.path.join(BASE_DIR, "archive", "icu_machine_sounds", "sounddino_heart"),
    os.path.join(BASE_DIR, "archive", "icu_machine_sounds", "bigsoundbank"),
]


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel//2)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, stride=2):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=kernel//2, output_padding=stride-1)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x, skip=None):
        x = F.relu(self.bn(self.conv(x)))
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


def collect_noise_files():
    all_noise = []
    for noise_dir in NOISE_DIRS:
        if os.path.isdir(noise_dir):
            files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir)
                     if f.lower().endswith(('.wav', '.mp3', '.flac'))]
            all_noise.extend(files)
            print(f"  {os.path.basename(noise_dir)}: {len(files)} files")
    print(f"  Total noise files: {len(all_noise)}")
    return all_noise


def load_audio(path, target_sr, n_samples):
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)
    if len(y) > n_samples:
        start = np.random.randint(0, len(y) - n_samples)
        y = y[start:start + n_samples]
    elif len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    return y.astype(np.float32)


def normalize_audio(y):
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y


class VocalBurstsDenoisingDataset(Dataset):
    def __init__(self, clean_files, noise_files):
        self.clean_files = clean_files
        self.noise_files = noise_files

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        try:
            import soundfile as sf
            clean_audio, sr = sf.read(clean_path)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.clean_files))

        if len(clean_audio) > N_SAMPLES:
            start = np.random.randint(0, len(clean_audio) - N_SAMPLES)
            clean_audio = clean_audio[start:start + N_SAMPLES]
        elif len(clean_audio) < N_SAMPLES:
            clean_audio = np.pad(clean_audio, (0, N_SAMPLES - len(clean_audio)))

        clean_audio = normalize_audio(clean_audio).astype(np.float32)

        noise_path = self.noise_files[np.random.randint(0, len(self.noise_files))]
        noise_audio, _ = sf.read(noise_path)
        if len(noise_audio) > N_SAMPLES:
            start = np.random.randint(0, len(noise_audio) - N_SAMPLES)
            noise_audio = noise_audio[start:start + N_SAMPLES]
        elif len(noise_audio) < N_SAMPLES:
            noise_audio = np.pad(noise_audio, (0, N_SAMPLES - len(noise_audio)))
        noise_audio = normalize_audio(noise_audio).astype(np.float32)

        snr_db = np.random.uniform(-2, 8)
        signal_power = np.mean(clean_audio ** 2) + 1e-8
        noise_power = np.mean(noise_audio ** 2) + 1e-8
        snr_linear = 10 ** (snr_db / 10)
        noise_scaled = noise_audio * np.sqrt(signal_power / (noise_power * snr_linear))

        mixed_audio = clean_audio + noise_scaled
        mixed_audio = np.clip(mixed_audio, -1.0, 1.0)

        return torch.tensor(mixed_audio, dtype=torch.float32).unsqueeze(0), \
               torch.tensor(clean_audio, dtype=torch.float32).unsqueeze(0)


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
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
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
            loss = 0.03 * l2_loss(output, clean) + l1_loss(output, clean)
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
                loss = 0.03 * l2_loss(output, clean) + l1_loss(output, clean)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS}: LR={lr:.6f}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/simple_denoiser.pth")
            print(f"  Saved best model (val_loss={avg_val_loss:.6f})")

    return model


def preconvert_noise_to_wav(noise_files, cache_dir="data/noise_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cached = []
    for nf in tqdm(noise_files, desc="Pre-converting noise to 16kHz WAV"):
        base = os.path.splitext(os.path.basename(nf))[0]
        out = os.path.join(cache_dir, base + ".wav")
        if not os.path.exists(out):
            try:
                y, sr = librosa.load(nf, sr=TARGET_SR, mono=True)
                import soundfile as sf
                sf.write(out, y, TARGET_SR)
            except Exception:
                continue
        cached.append(out)
    return cached


def main():
    os.makedirs("checkpoints", exist_ok=True)

    print("Collecting clean vocal audio files...")
    clean_files = collect_clean_files()
    print(f"\nTotal clean files: {len(clean_files)}")

    if len(clean_files) == 0:
        print("No audio files found!")
        return

    print("\nCollecting real ICU noise files...")
    noise_files = collect_noise_files()
    if len(noise_files) == 0:
        print("No noise files found!")
        return

    print("\nPre-converting noise to 16kHz WAV (one-time)...")
    noise_files = preconvert_noise_to_wav(noise_files)

    train_files, val_files = train_test_split(clean_files, test_size=0.15, random_state=42)
    print(f"\nTrain: {len(train_files)} files, Val: {len(val_files)} files")

    train_dataset = VocalBurstsDenoisingDataset(train_files, noise_files)
    val_dataset = VocalBurstsDenoisingDataset(val_files, noise_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SimpleDenoiser()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_denoiser(model, train_loader, val_loader)

    print(f"\nTraining complete! Best model saved to checkpoints/simple_denoiser.pth")


if __name__ == "__main__":
    main()
