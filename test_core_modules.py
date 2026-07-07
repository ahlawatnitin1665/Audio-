"""
Simplified test - no audio model required
"""
import sys
sys.path.insert(0, 'backend')

from backend.distress_calculator import CombinedDistressCalculator
from backend.alert_system import AlertManager
import numpy as np

print("=" * 60)
print("TESTING CORE MODULES (No server needed)")
print("=" * 60)

# Test 1: Distress Calculator
print("\n1. Testing Distress Calculator...")
calc = CombinedDistressCalculator()

test_data = {
    'audio': {'crying': 0.6, 'normal': 0.4, 'coughing': 0, 'groaning': 0, 'gasping': 0},
    'face': {'detected': True, 'emotion': 'sad', 'emotion_confidence': 0.8, 'mouth_open': 0.3},
    'eye': {'detected': True, 'blink_rate': 22, 'eye_openness': 0.7, 'pupil_dilation': 0.6},
    'posture': {'detected': True, 'hunched': 0.4, 'head_down': 0.2, 'movement_intensity': 0.6, 'hand_to_face': False}
}

result = calc.calculate_combined_score(
    audio_probs=test_data['audio'],
    face_data=test_data['face'],
    eye_data=test_data['eye'],
    posture_data=test_data['posture'],
    smooth=False
)

print(f"   Distress Score: {result['score']:.3f}")
print(f"   Audio: {result['audio_score']:.3f}")
print(f"   Face: {result['face_score']:.3f}")
print(f"   Eye: {result['eye_score']:.3f}")
print(f"   Posture: {result['posture_score']:.3f}")

# Test 2: Alert Manager
print("\n2. Testing Alert Manager...")
alerts = []
def on_alert(a):
    alerts.append(a)

alert_mgr = AlertManager(callback=on_alert)
alert = alert_mgr.generate_alert(
    distress_score=result['score'],
    audio_score=result['audio_score'],
    face_score=result['face_score'],
    eye_score=result['eye_score'],
    posture_score=result['posture_score'],
    trend='stable'
)

print(f"   Alert Level: {alert.level.value}")
print(f"   Message: {alert.message[:50]}...")
print(f"   Alerts generated: {len(alerts)}")

print("\n" + "=" * 60)
print("CORE MODULES OK - Backend server issue to investigate")
print("=" * 60)

print("\nNow test the backend server:")
print("1. Make sure backend server is running with NEW code")
print("2. Run this test again: python debug_test.py")
print("3. Check server terminal for log messages")
