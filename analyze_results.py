import os, numpy as np, soundfile as sf, librosa

TARGET_SR = 16000
N_FFT = 512
HOP_LENGTH = 128
base = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base, "eval_output")

def si_snr(estimate, target):
    eps = 1e-8
    target = target / (np.linalg.norm(target) + eps)
    estimate = estimate / (np.linalg.norm(estimate) + eps)
    s_proj = np.dot(estimate, target) * target
    e_noise = estimate - s_proj
    return 10 * np.log10((np.linalg.norm(s_proj)**2 + eps) / (np.linalg.norm(e_noise)**2 + eps))

def compute_snr(clean, signal):
    noise = signal - clean
    return 10 * np.log10((np.linalg.norm(clean)**2 + 1e-8) / (np.linalg.norm(noise)**2 + 1e-8))

def lsd(clean_spec, est_spec):
    diff = np.log1p(np.abs(clean_spec) + 1e-8) - np.log1p(np.abs(est_spec) + 1e-8)
    return np.sqrt(np.mean(diff**2))

files = sorted([f.replace("_clean.wav", "") for f in os.listdir(out_dir) if f.endswith("_clean.wav")])
print(f"{'File':<25} {'Input SNR':>9} {'Output SNR':>10} {'SI-SNR':>9} {'LSD':>7}")
print("-" * 65)

avg_in_snr, avg_out_snr, avg_si, avg_lsd = 0, 0, 0, 0
for fname in files:
    clean, _ = sf.read(os.path.join(out_dir, f"{fname}_clean.wav"))
    noisy, _ = sf.read(os.path.join(out_dir, f"{fname}_noisy.wav"))
    denoised, _ = sf.read(os.path.join(out_dir, f"{fname}_denoised.wav"))

    in_snr = compute_snr(clean, noisy)
    out_snr = compute_snr(clean, denoised)
    si = si_snr(denoised, clean)

    clean_spec = librosa.stft(clean, n_fft=N_FFT, hop_length=HOP_LENGTH)
    denoised_spec = librosa.stft(denoised, n_fft=N_FFT, hop_length=HOP_LENGTH)
    l = lsd(clean_spec, denoised_spec)

    avg_in_snr += in_snr
    avg_out_snr += out_snr
    avg_si += si
    avg_lsd += l

    print(f"{fname:<25} {in_snr:>8.2f}dB {out_snr:>8.2f}dB {si:>7.2f}dB {l:>6.3f}")

n = len(files)
print("-" * 65)
print(f"{'AVERAGE':<25} {avg_in_snr/n:>8.2f}dB {avg_out_snr/n:>8.2f}dB {avg_si/n:>7.2f}dB {avg_lsd/n:>6.3f}")

imp = avg_out_snr / n - avg_in_snr / n
print(f"\nSNR Improvement: {imp:.2f} dB")
if imp > 5:
    print("Rating: GOOD - denoiser is removing noise effectively")
elif imp > 2:
    print("Rating: DECENT - some noise reduction but room for improvement")
elif imp > 0:
    print("Rating: POOR - barely any improvement")
else:
    print("Rating: BAD - denoiser is making it worse")
