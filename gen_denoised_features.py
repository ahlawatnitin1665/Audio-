import sys; sys.path.insert(0, 'src')
import torch, os, json, numpy as np, soundfile as sf
from tqdm import tqdm
from simple_denoiser import SimpleDenoiser
from preprocessing import extract_features, normalize_audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

BASE = r'C:\Users\Lenovo\OneDrive\Desktop\elc\patient_audio_monitor'

denoiser = SimpleDenoiser().to(device)
denoiser.load_state_dict(torch.load(os.path.join(BASE, 'checkpoints', 'simple_denoiser.pth'), map_location=device))
denoiser.eval()

with open(os.path.join(BASE, 'data', 'processed', 'metadata.json')) as f:
    records = json.load(f)

out_dir = os.path.join(BASE, 'data', 'processed_denoised')
os.makedirs(out_dir, exist_ok=True)

vocal_16k = os.path.join(BASE, 'data', 'vocal_16k')
count = 0
for r in tqdm(records, desc='Denoising'):
    orig_path = r['original']
    # orig is: C:\...\data\raw\vocal_bursts\Cough\Cough_00000000.flac
    # extract: Cough\Cough_00000000.flac -> Cough\Cough_00000000.wav
    parts = orig_path.split('vocal_bursts\\')
    if len(parts) < 2:
        continue
    rel = parts[1].replace('.flac', '.wav')
    wav_path = os.path.join(vocal_16k, rel)
    
    out_name = os.path.basename(r['file'])
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        count += 1
        continue
    
    if not os.path.exists(wav_path):
        continue
    
    try:
        audio, sr = sf.read(wav_path)
        if len(audio) < 48000:
            audio = np.pad(audio, (0, 48000 - len(audio)))
        audio = audio[:48000]
        audio = normalize_audio(audio)
        
        inp = torch.tensor(audio).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            den = denoiser(inp).squeeze(0).squeeze(0).cpu().numpy()
        den = normalize_audio(den)
        
        features = extract_features(den)
        np.save(out_path, features)
        count += 1
    except Exception as e:
        pass

print(f'Generated {count} denoised feature files')
