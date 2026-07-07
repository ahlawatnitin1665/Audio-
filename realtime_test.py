"""
Real-time test - sends synchronized data at 10 FPS
Tests timing and synchronization
"""
import sys
sys.path.insert(0, 'backend')

from backend.client import DistressMonitoringClient
import numpy as np
import time

results = []
alerts = []
frame_count = 10

def on_result(r):
    results.append(r)
    metadata = r.get('metadata', {})
    score = r.get('distress_score', 0)
    level = r.get('alert_level', 'unknown')
    frame_id = metadata.get('frame_id', '?')
    latency = metadata.get('capture_to_process_latency_ms', 0)
    proc_time = metadata.get('processing_time_ms', 0)
    print(f"  Frame {frame_id}: Score={score:.3f} | {level:8} | Latency={latency:.1f}ms | Proc={proc_time:.1f}ms")

def on_alert(a):
    alerts.append(a)
    print(f"  ALERT: {a['level'].upper()} - {a['message'][:50]}")

print("=" * 70)
print("REAL-TIME SYNC TEST - 10 FPS")
print("=" * 70)

client = DistressMonitoringClient(
    'http://localhost:5000', 
    on_result=on_result, 
    on_alert=on_alert,
    frame_rate=10
)

if not client.connect():
    print("FAIL: Could not connect")
    exit(1)
print("Connected")

if not client.start_session(patient_id='realtime_test'):
    print("FAIL: Could not start session")
    exit(1)
print("Session started")
print(f"\nSending {frame_count} frames at 10 FPS...\n")

start_time = time.time()

for i in range(frame_count):
    audio = np.random.randn(16000).astype(np.float32)
    face = {
        'detected': True,
        'emotion': ['sad', 'fearful', 'angry', 'neutral'][i % 4],
        'emotion_confidence': 0.8,
        'mouth_open': 0.2 + (i * 0.05)
    }
    eye = {
        'detected': True,
        'blink_rate': 18 + i,
        'eye_openness': 0.7,
        'pupil_dilation': 0.5
    }
    posture = {
        'detected': True,
        'hunched': 0.3 + (i * 0.05),
        'head_down': 0.1,
        'movement_intensity': 0.5,
        'hand_to_face': i > 6
    }

    client.send_data(audio=audio, face=face, eye=eye, posture=posture)
    time.sleep(0.5)

print("\nWaiting for all responses...")
time.sleep(5)

total_time = time.time() - start_time
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Frames sent: {frame_count}")
print(f"Results received: {len(results)}")
print(f"Alerts generated: {len(alerts)}")

if len(results) > 0:
    scores = [r.get('distress_score', 0) for r in results]
    latencies = [r.get('metadata', {}).get('capture_to_process_latency_ms', 0) for r in results]
    proc_times = [r.get('metadata', {}).get('processing_time_ms', 0) for r in results]

    print(f"\nTiming Stats:")
    print(f"  Avg Latency: {sum(latencies)/len(latencies):.1f}ms")
    print(f"  Max Latency: {max(latencies):.1f}ms")
    print(f"  Avg Processing: {sum(proc_times)/len(proc_times):.1f}ms")

    print(f"\nDistress Scores:")
    print(f"  Min: {min(scores):.3f} | Max: {max(scores):.3f} | Avg: {sum(scores)/len(scores):.3f}")
    print(f"  All: {[f'{s:.2f}' for s in scores]}")

client.end_session()
client.disconnect()
print("\nDone!")
