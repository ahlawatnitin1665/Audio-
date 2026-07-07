import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from denoising_model import TwoStagePipeline
from preprocessing import extract_features, TARGET_SR
import torch
import librosa
import numpy as np


def test_pipeline():
    print("Testing Two-Stage Pipeline")
    print("=" * 50)
    
    pipeline = TwoStagePipeline()
    
    test_files = [
        "C:/Users/Lenovo/OneDrive/Desktop/elc/test/Male Deep _ Heavy Breathing Sound Effect - SoundEffectsFactory.mp3",
        "C:/Users/Lenovo/OneDrive/Desktop/elc/test/Baby Crying Sound Effect #2 - SoundEffectsFactory.mp3",
    ]
    
    for file_path in test_files:
        print(f"\nTesting: {os.path.basename(file_path)}")
        
        audio, sr = librosa.load(file_path, sr=TARGET_SR, duration=3)
        audio = audio[:TARGET_SR * 3] if len(audio) >= TARGET_SR * 3 else np.pad(audio, (0, TARGET_SR * 3 - len(audio)))
        
        result = pipeline.predict(audio, sr)
        
        print(f"  Class: {result['class']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print("  Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"    {class_name}: {prob:.1%}")
    
    print("\n" + "=" * 50)
    print("Pipeline test completed!")


if __name__ == "__main__":
    test_pipeline()
