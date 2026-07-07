import os
import json
import soundfile as sf
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "data", "raw", "vocal_bursts")
os.makedirs(output_dir, exist_ok=True)

# Load dataset from HuggingFace
print("Loading Vocal Bursts dataset from HuggingFace...")
ds = load_dataset("0x3/vocal-bursts")

# Get the train split
train_data = ds['train']
print(f"Total samples: {len(train_data)}")

# Filter only the 4 target classes
target_classes = ['Cough', 'Crying', 'Moan', 'Pant']
filtered_data = [sample for sample in train_data if sample['label'] in target_classes]
print(f"Filtered samples (4 classes): {len(filtered_data)}")

# Create subdirectories for each class
for cls in target_classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# Save samples
print("Saving audio files...")
for i, sample in enumerate(filtered_data):
    label = sample['label']
    audio = sample['audio']
    
    # Save audio file
    audio_path = os.path.join(output_dir, label, f"{i:05d}.wav")
    sf.write(audio_path, audio['array'], audio['sampling_rate'])
    
    if (i + 1) % 100 == 0:
        print(f"  Saved {i + 1}/{len(filtered_data)} samples")

print(f"Done! Saved {len(filtered_data)} samples to {output_dir}")

# Print class distribution
print("\nClass distribution:")
for cls in target_classes:
    count = len([s for s in filtered_data if s['label'] == cls])
    print(f"  {cls}: {count}")
