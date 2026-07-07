"""
Quick test of distress calculator and alert manager
"""
import sys
sys.path.insert(0, '.')

from distress_calculator import CombinedDistressCalculator
from alert_system import AlertManager
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("DISTRESS CALCULATOR TEST")
logger.info("=" * 60)

calc = CombinedDistressCalculator()

scenarios = [
    ("NORMAL", {"normal": 1.0, "coughing": 0, "crying": 0, "groaning": 0, "gasping": 0}, "happy", "normal"),
    ("MILD", {"coughing": 0.7, "normal": 0.3, "crying": 0, "groaning": 0, "gasping": 0}, "neutral", "mild"),
    ("MODERATE", {"crying": 0.6, "normal": 0.4, "coughing": 0, "groaning": 0, "gasping": 0}, "sad", "moderate"),
    ("SEVERE", {"gasping": 0.5, "crying": 0.3, "normal": 0.2, "coughing": 0, "groaning": 0}, "fearful", "severe"),
]

for name, audio, emotion, expected_level in scenarios:
    result = calc.calculate_combined_score(
        audio_probs=audio,
        face_data={"detected": True, "emotion": emotion, "emotion_confidence": 0.8, "mouth_open": 0.2},
        eye_data={"detected": True, "blink_rate": 20, "eye_openness": 0.7, "pupil_dilation": 0.5},
        posture_data={"detected": True, "hunched": 0.3, "head_down": 0.1, "movement_intensity": 0.5, "hand_to_face": False},
        smooth=False
    )
    alert_level, _ = calc.get_alert_level(result["score"])
    status = "OK" if alert_level == expected_level else f"FAIL (got {alert_level}, expected {expected_level})"
    logger.info(f"{name:10} | Score: {result['score']:.3f} | Alert: {alert_level:10} | {status}")

logger.info("")
logger.info("=" * 60)
logger.info("ALERT MANAGER TEST")
logger.info("=" * 60)

alerts = []
def on_alert(alert):
    alerts.append(alert)

manager = AlertManager(callback=on_alert)

test_scores = [(0.15, "normal"), (0.40, "mild"), (0.60, "moderate"), (0.85, "severe")]
for score, expected in test_scores:
    alert = manager.generate_alert(score, score, score, score, score)
    status = "OK" if alert.level.value == expected else f"FAIL (expected {expected})"
    logger.info(f"{expected:10} | Score: {score:.2f} | {status}")

logger.info("")
logger.info("=" * 60)
logger.info(f"TESTS COMPLETE - {len(alerts)} alerts generated")
logger.info("=" * 60)
