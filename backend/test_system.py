"""
Test and demo script for the distress monitoring backend
Simulates real-time data streams and verifies system functionality
"""
import sys
import os
import time
import numpy as np
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from distress_calculator import CombinedDistressCalculator
from alert_system import AlertManager, AlertLevel
from client import DistressMonitoringClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_distress_calculator():
    """Test the distress calculator with various scenarios"""
    logger.info("=" * 60)
    logger.info("Testing Distress Calculator")
    logger.info("=" * 60)

    calc = CombinedDistressCalculator()

    test_scenarios = [
        {
            'name': 'Normal patient (all normal)',
            'audio': {'normal': 1.0, 'coughing': 0.0, 'crying': 0.0, 'groaning': 0.0, 'gasping': 0.0},
            'face': {'detected': True, 'emotion': 'happy', 'emotion_confidence': 0.9, 'mouth_open': 0.1},
            'eye': {'detected': True, 'blink_rate': 18, 'eye_openness': 0.6, 'pupil_dilation': 0.5},
            'posture': {'detected': True, 'hunched': 0.1, 'head_down': 0.0, 'movement_intensity': 0.3, 'hand_to_face': False}
        },
        {
            'name': 'Mild distress (slight coughing)',
            'audio': {'coughing': 0.7, 'normal': 0.3, 'crying': 0.0, 'groaning': 0.0, 'gasping': 0.0},
            'face': {'detected': True, 'emotion': 'neutral', 'emotion_confidence': 0.7, 'mouth_open': 0.3},
            'eye': {'detected': True, 'blink_rate': 22, 'eye_openness': 0.7, 'pupil_dilation': 0.6},
            'posture': {'detected': True, 'hunched': 0.4, 'head_down': 0.2, 'movement_intensity': 0.6, 'hand_to_face': False}
        },
        {
            'name': 'Moderate distress (crying)',
            'audio': {'crying': 0.6, 'normal': 0.4, 'coughing': 0.0, 'groaning': 0.0, 'gasping': 0.0},
            'face': {'detected': True, 'emotion': 'sad', 'emotion_confidence': 0.85, 'mouth_open': 0.4},
            'eye': {'detected': True, 'blink_rate': 28, 'eye_openness': 0.8, 'pupil_dilation': 0.7},
            'posture': {'detected': True, 'hunched': 0.6, 'head_down': 0.5, 'movement_intensity': 0.7, 'hand_to_face': True}
        },
        {
            'name': 'Severe distress (gasping + crying)',
            'audio': {'gasping': 0.5, 'crying': 0.3, 'normal': 0.2, 'coughing': 0.0, 'groaning': 0.0},
            'face': {'detected': True, 'emotion': 'fearful', 'emotion_confidence': 0.9, 'mouth_open': 0.8},
            'eye': {'detected': True, 'blink_rate': 35, 'eye_openness': 0.9, 'pupil_dilation': 0.8},
            'posture': {'detected': True, 'hunched': 0.8, 'head_down': 0.7, 'movement_intensity': 0.9, 'hand_to_face': True}
        }
    ]

    for scenario in test_scenarios:
        result = calc.calculate_combined_score(
            audio_probs=scenario['audio'],
            face_data=scenario['face'],
            eye_data=scenario['eye'],
            posture_data=scenario['posture'],
            smooth=False
        )

        alert_level, priority = calc.get_alert_level(result['score'])

        logger.info(f"\nScenario: {scenario['name']}")
        logger.info(f"  Distress Score: {result['score']:.3f}")
        logger.info(f"  Audio Score: {result['audio_score']:.3f}")
        logger.info(f"  Face Score: {result['face_score']:.3f}")
        logger.info(f"  Eye Score: {result['eye_score']:.3f}")
        logger.info(f"  Posture Score: {result['posture_score']:.3f}")
        logger.info(f"  Alert Level: {alert_level} ({priority})")


def test_alert_manager():
    """Test the alert manager"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Alert Manager")
    logger.info("=" * 60)

    alerts_generated = []

    def on_alert(alert):
        alerts_generated.append(alert)

    manager = AlertManager(callback=on_alert)

    # Generate various alerts
    test_cases = [
        (0.15, 0.1, 0.1, 0.1, 0.1, "normal level"),
        (0.40, 0.4, 0.3, 0.3, 0.3, "mild level"),
        (0.60, 0.6, 0.5, 0.5, 0.5, "moderate level"),
        (0.85, 0.9, 0.8, 0.8, 0.8, "severe level"),
    ]

    for distress, audio, face, eye, posture, desc in test_cases:
        alert = manager.generate_alert(
            distress_score=distress,
            audio_score=audio,
            face_score=face,
            eye_score=eye,
            posture_score=posture,
            trend="worsening"
        )
        logger.info(f"\n{desc}:")
        logger.info(f"  Level: {alert.level.value}")
        logger.info(f"  Message: {alert.message}")

    logger.info(f"\nTotal alerts generated: {len(alerts_generated)}")
    logger.info(f"Statistics: {manager.get_statistics()}")


def test_client_connection():
    """Test client connection (requires server running)"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Client Connection")
    logger.info("=" * 60)

    results = []
    alerts = []

    def on_result(result):
        results.append(result)
        if 'distress_score' in result:
            logger.info(f"Result: Distress={result['distress_score']:.3f}, Alert={result.get('alert_level', 'N/A')}")

    def on_alert(alert):
        alerts.append(alert)
        logger.warning(f"🚨 Alert: {alert['level'].upper()} - {alert['message']}")

    def on_error(error):
        logger.error(f"Error: {error}")

    client = DistressMonitoringClient(
        server_url='http://localhost:5000',
        on_result=on_result,
        on_alert=on_alert,
        on_error=on_error
    )

    logger.info("Connecting to server...")
    if not client.connect(timeout=5):
        logger.warning("Could not connect to server (is it running?)")
        return

    logger.info("Starting session...")
    if not client.start_session(patient_id='test_patient_001'):
        logger.error("Failed to start session")
        client.disconnect()
        return

    # Send data
    logger.info("Sending multimodal data...")
    for i in range(5):
        audio = np.random.randn(16000)

        face = {
            'detected': True,
            'emotion': 'sad' if i > 2 else 'neutral',
            'emotion_confidence': 0.8,
            'mouth_open': 0.2 + (i * 0.05)
        }

        eye = {
            'detected': True,
            'blink_rate': 20 + (i * 2),
            'eye_openness': 0.7,
            'pupil_dilation': 0.5 + (i * 0.05)
        }

        posture = {
            'detected': True,
            'hunched': 0.3 + (i * 0.05),
            'head_down': 0.1 + (i * 0.03),
            'movement_intensity': 0.5,
            'hand_to_face': i > 2
        }

        client.send_data(
            audio=audio,
            face=face,
            eye=eye,
            posture=posture
        )

        time.sleep(0.5)

    logger.info("Ending session...")
    client.end_session()
    client.disconnect()

    logger.info(f"Results received: {len(results)}")
    logger.info(f"Alerts generated: {len(alerts)}")


def run_all_tests():
    """Run all tests"""
    logger.info("Starting Backend System Tests\n")

    # Test 1: Distress Calculator
    test_distress_calculator()

    # Test 2: Alert Manager
    test_alert_manager()

    # Test 3: Client Connection (optional)
    try:
        test_client_connection()
    except Exception as e:
        logger.warning(f"Client connection test skipped: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("All Tests Completed")
    logger.info("=" * 60)


if __name__ == '__main__':
    run_all_tests()
