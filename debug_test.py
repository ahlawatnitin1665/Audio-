"""
Simple debug test to check what's happening
"""
import sys
sys.path.insert(0, 'backend')

from backend.client import DistressMonitoringClient
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)

results = []
alerts = []

def on_result(r):
    results.append(r)
    metadata = r.get('metadata', {})
    print(f"RESULT: Score={r.get('distress_score', 0):.3f}, Alert={r.get('alert_level')}")
    print(f"  Frame ID: {metadata.get('frame_id')}")
    print(f"  Capture-to-Process Latency: {metadata.get('capture_to_process_latency_ms', 0):.1f}ms")
    print(f"  Processing Time: {metadata.get('processing_time_ms', 0):.1f}ms")

def on_alert(a):
    alerts.append(a)
    print(f"ALERT: {a['level']} - {a['message'][:40]}")

print("Connecting...")
client = DistressMonitoringClient('http://localhost:5000', on_result=on_result, on_alert=on_alert)

if not client.connect():
    print("FAIL: Could not connect")
    exit(1)

print("OK: Connected")
print("Starting session...")

if not client.start_session(patient_id='test'):
    print("FAIL: Could not start session")
    exit(1)

print("OK: Session started")
print("Sending test data...")

audio = np.random.randn(16000).astype(np.float32)
face = {'detected': True, 'emotion': 'sad', 'emotion_confidence': 0.8, 'mouth_open': 0.3}
eye = {'detected': True, 'blink_rate': 22, 'eye_openness': 0.7, 'pupil_dilation': 0.6}
posture = {'detected': True, 'hunched': 0.4, 'head_down': 0.2, 'movement_intensity': 0.6, 'hand_to_face': False}

success = client.send_data(audio=audio, face=face, eye=eye, posture=posture)
print(f"Send result: {success}")

print("Waiting for response...")
time.sleep(2)

print(f"\nResults received: {len(results)}")
print(f"Alerts generated: {len(alerts)}")

if len(results) > 0:
    print(f"\nFirst result: {results[0]}")

client.end_session()
client.disconnect()
print("Done")
