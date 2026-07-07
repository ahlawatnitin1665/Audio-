"""
Test with real audio model integration
Sends audio + face data and verifies audio processing
"""
import sys
sys.path.insert(0, 'backend')
sys.path.insert(0, 'src')

from backend.client import DistressMonitoringClient
import numpy as np
import time

results = []

def on_result(r):
    results.append(r)
    score = r.get('distress_score', 0)
    level = r.get('alert_level', 'unknown')
    breakdown = r.get('breakdown', {})
    individual = r.get('individual_scores', {})
    
    print(f"  Score: {score:.3f} | Level: {level}")
    print(f"    Audio: {individual.get('audio', 0):.3f} (weighted: {breakdown.get('audio', 0):.3f})")
    print(f"    Face:  {individual.get('face', 0):.3f} (weighted: {breakdown.get('face', 0):.3f})")
    print(f"    Eye:   {individual.get('eye', 0):.3f} (weighted: {breakdown.get('eye', 0):.3f})")
    print(f"    Posture: {individual.get('posture', 0):.3f} (weighted: {breakdown.get('posture', 0):.3f})")

print("=" * 70)
print("AUDIO MODEL INTEGRATION TEST")
print("=" * 70)

client = DistressMonitoringClient('http://localhost:5000', on_result=on_result)
if not client.connect():
    print("FAIL: Could not connect")
    exit(1)
print("Connected")

if not client.start_session(patient_id='audio_test'):
    print("FAIL: Could not start session")
    exit(1)
print("Session started\n")

# Test 1: Normal audio (silence)
print("Test 1: Silence (should be mostly normal/noise)")
audio = np.zeros(16000, dtype=np.float32)
face = {'detected': True, 'emotion': 'neutral', 'emotion_confidence': 0.9, 'mouth_open': 0.1}
client.send_data(audio=audio, face=face)
time.sleep(8)

# Test 2: Random audio (noise-like)
print("\nTest 2: Random noise")
audio = np.random.randn(16000).astype(np.float32) * 0.1
face = {'detected': True, 'emotion': 'neutral', 'emotion_confidence': 0.8, 'mouth_open': 0.2}
client.send_data(audio=audio, face=face)
time.sleep(8)

# Test 3: Sad face with silence
print("\nTest 3: Sad face with silence")
audio = np.zeros(16000, dtype=np.float32)
face = {'detected': True, 'emotion': 'sad', 'emotion_confidence': 0.9, 'mouth_open': 0.4}
client.send_data(audio=audio, face=face)
time.sleep(8)

# Test 4: All distress signals
print("\nTest 4: Distress signals (sad face + noise)")
audio = np.random.randn(16000).astype(np.float32) * 0.5
face = {'detected': True, 'emotion': 'sad', 'emotion_confidence': 0.9, 'mouth_open': 0.6}
eye = {'detected': True, 'blink_rate': 30, 'eye_openness': 0.9, 'pupil_dilation': 0.8}
posture = {'detected': True, 'hunched': 0.7, 'head_down': 0.5, 'movement_intensity': 0.8, 'hand_to_face': True}
client.send_data(audio=audio, face=face, eye=eye, posture=posture)
time.sleep(8)

print("\n" + "=" * 70)
print(f"Results received: {len(results)}")
print("=" * 70)

client.end_session()
client.disconnect()
print("Done!")
