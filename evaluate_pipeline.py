import os
import sys
import glob
import numpy as np
import torch
import librosa
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import PatientAudioCNN, NUM_CLASSES
from src.preprocessing import extract_features, normalize_audio, TARGET_SR, DURATION, N_SAMPLES
from src.simple_denoiser import SimpleTwoStagePipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
DENOISER_PATH = os.path.join(BASE_DIR, "checkpoints", "simple_denoiser.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OriginalClassifierOnly:
    """Original model without denoiser"""
    def __init__(self):
        self.model = PatientAudioCNN(NUM_CLASSES).to(DEVICE)
        checkpoint = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.class_names = ["coughing", "crying", "groaning", "gasping", "normal"]
    
    def predict(self, audio):
        audio = normalize_audio(audio)
        features = extract_features(audio)
        x = torch.tensor(features, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x.to(DEVICE)), dim=1).squeeze()
        dominant_class = torch.argmax(probs).item()
        confidence = probs[dominant_class].item()
        return {
            "class": self.class_names[dominant_class],
            "confidence": confidence,
            "probs": probs.cpu().numpy()
        }

def evaluate_models():
    print("=" * 70)
    print("PIPELINE EVALUATION: Original vs Two-Stage Denoiser")
    print("=" * 70)
    
    original = OriginalClassifierOnly()
    pipeline = SimpleTwoStagePipeline(denoiser_path=DENOISER_PATH, classifier_path=CLASSIFIER_PATH)
    
    test_dir = os.path.join(BASE_DIR, "data/hls_organized/test")
    test_files = []
    
    for category in ['heart', 'lung', 'mixed']:
        cat_dir = os.path.join(test_dir, category)
        if os.path.exists(cat_dir):
            files = glob.glob(os.path.join(cat_dir, "*.wav"))
            test_files.extend([(f, category) for f in files])
    
    print(f"\nTest set: {len(test_files)} files")
    print(f"  Heart: {sum(1 for _, c in test_files if c == 'heart')}")
    print(f"  Lung: {sum(1 for _, c in test_files if c == 'lung')}")
    print(f"  Mixed: {sum(1 for _, c in test_files if c == 'mixed')}")
    
    results = {
        "original": [],
        "pipeline": [],
        "category_original": {"heart": [], "lung": [], "mixed": []},
        "category_pipeline": {"heart": [], "lung": [], "mixed": []}
    }
    
    print("\nProcessing test files...")
    for audio_path, category in tqdm(test_files, desc="Evaluating"):
        audio, sr = librosa.load(audio_path, sr=TARGET_SR, duration=DURATION)
        if len(audio) < N_SAMPLES:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
        else:
            audio = audio[:N_SAMPLES]
        
        # Original model
        orig_result = original.predict(audio)
        results["original"].append({
            "file": os.path.basename(audio_path),
            "category": category,
            "class": orig_result["class"],
            "confidence": orig_result["confidence"],
            "probs": orig_result["probs"]
        })
        results["category_original"][category].append(orig_result["confidence"])
        
        # Two-stage pipeline
        pipe_result = pipeline.predict(audio, TARGET_SR)
        results["pipeline"].append({
            "file": os.path.basename(audio_path),
            "category": category,
            "class": pipe_result["class"],
            "confidence": pipe_result["confidence"],
            "probs": np.array([pipe_result["probabilities"][name] for name in pipeline.class_names])
        })
        results["category_pipeline"][category].append(pipe_result["confidence"])
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("RESULTS BY AUDIO SOURCE CATEGORY")
    print("=" * 70)
    
    for category in ['heart', 'lung', 'mixed']:
        orig_confs = results["category_original"][category]
        pipe_confs = results["category_pipeline"][category]
        
        print(f"\n{category.upper()} sounds ({len(orig_confs)} files):")
        print(f"  Original Model  - Avg Confidence: {np.mean(orig_confs):.1%} (±{np.std(orig_confs):.1%})")
        print(f"  Pipeline (2-st) - Avg Confidence: {np.mean(pipe_confs):.1%} (±{np.std(pipe_confs):.1%})")
        print(f"  Improvement: {(np.mean(pipe_confs) - np.mean(orig_confs))*100:+.1f}%")
    
    # Overall stats
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    
    orig_all = [r["confidence"] for r in results["original"]]
    pipe_all = [r["confidence"] for r in results["pipeline"]]
    
    print(f"\nOriginal Model (clean audio only):")
    print(f"  Avg Confidence: {np.mean(orig_all):.1%}")
    print(f"  Std Deviation:  {np.std(orig_all):.1%}")
    print(f"  Min/Max:        {np.min(orig_all):.1%} / {np.max(orig_all):.1%}")
    
    print(f"\nTwo-Stage Pipeline (with denoiser):")
    print(f"  Avg Confidence: {np.mean(pipe_all):.1%}")
    print(f"  Std Deviation:  {np.std(pipe_all):.1%}")
    print(f"  Min/Max:        {np.min(pipe_all):.1%} / {np.max(pipe_all):.1%}")
    
    improvement = np.mean(pipe_all) - np.mean(orig_all)
    print(f"\nNet Improvement: {improvement*100:+.2f}%")
    
    if improvement > 0.01:
        print("[+] Pipeline improves confidence on HLS-CMDS test set")
    elif improvement < -0.01:
        print("[-] Pipeline slightly reduces confidence on HLS-CMDS test set")
    else:
        print("[~] Pipeline has minimal effect on confidence")
    
    # Sample outputs
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS")
    print("=" * 70)
    
    for i in range(min(5, len(results["original"]))):
        orig = results["original"][i]
        pipe = results["pipeline"][i]
        print(f"\n{orig['file']} ({orig['category']}):")
        print(f"  Original:  {orig['class']:10s} {orig['confidence']:.1%}")
        print(f"  Pipeline:  {pipe['class']:10s} {pipe['confidence']:.1%}")

if __name__ == "__main__":
    evaluate_models()
