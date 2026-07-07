"""Minimal client test - find exactly where it breaks"""
import sys, time
sys.path.insert(0, 'backend')
from backend.client import DistressMonitoringClient
import numpy as np

print("STEP 1: Create client")
client = DistressMonitoringClient(
    'http://localhost:5000',
    on_result=lambda r: print(f"  >>> CALLBACK: score={r.get('distress_score', '?')}"),
    on_alert=lambda a: print(f"  >>> ALERT: {a}")
)

print("STEP 2: Connect")
client.connect()
print(f"  connected={client.connected}")

print("STEP 3: Start session")
client.start_session(patient_id='minimal')
print(f"  session_active={client.session_active}")

print("STEP 4: Send data")
audio = np.random.randn(8000).astype(np.float32)
face = {'detected': True, 'emotion': 'sad', 'emotion_confidence': 0.8, 'mouth_open': 0.3}
result = client.send_data(audio=audio, face=face)
print(f"  send_result={result}")

print("STEP 5: Wait 5 seconds...")
time.sleep(5)

print("STEP 6: Check")
print(f"  connected={client.connected}")
print(f"  session_active={client.session_active}")

client.end_session()
client.disconnect()
print("DONE")
