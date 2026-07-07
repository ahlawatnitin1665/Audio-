import os, soundfile as sf, librosa, numpy as np
from tqdm import tqdm

src = 'data/raw/vocal_bursts'
dst = 'data/vocal_16k'
for c in ['Cough', 'Crying', 'Moan', 'Normal', 'Pant']:
    os.makedirs(os.path.join(dst, c), exist_ok=True)
    files = [f for f in os.listdir(os.path.join(src, c)) if f.lower().endswith(('.wav', '.flac'))]
    for f in tqdm(files, desc=c):
        out_path = os.path.join(dst, c, f.replace('.flac', '.wav'))
        if os.path.exists(out_path):
            continue
        try:
            data, sr = sf.read(os.path.join(src, c, f))
            if sr != 16000:
                data = librosa.resample(data, orig_sr=sr, target_sr=16000).astype(np.float32)
            sf.write(out_path, data, 16000)
        except Exception as e:
            print(f"  Skipping corrupt file: {c}/{f} - {e}")
print('Done')
