"""
Simple test client to verify backend is working
Sends sample multimodal data and displays results
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.client import DistressMonitoringClient
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_backend():
    """Test the backend with sample data"""
    
    logger.info("=" * 60)
    logger.info("BACKEND CLIENT TEST")
    logger.info("=" * 60)
    logger.info("")

    results_received = []
    alerts_received = []

    def on_result(result):
        results_received.append(result)
        distress = result.get('distress_score', 0)
        alert_level = result.get('alert_level', 'unknown')
        data_point = result.get('data_point', 0)
        logger.info(f"[{data_point}] Distress: {distress:.3f} | Level: {alert_level:10} | Trend: {result.get('trend', 'N/A')}")

    def on_alert(alert):
        alerts_received.append(alert)
        logger.warning(f"🚨 ALERT: {alert['level'].upper()} - {alert['message']}")

    def on_error(error):
        logger.error(f"Error: {error}")

    # Connect to backend
    logger.info("Connecting to backend server...")
    client = DistressMonitoringClient(
        server_url='http://localhost:5000',
        on_result=on_result,
        on_alert=on_alert,
        on_error=on_error
    )

    if not client.connect(timeout=5):
        logger.error("Failed to connect to backend server!")
        logger.error("Is the server running? Try: python backend/server.py")
        return False

    logger.info("✅ Connected to backend")
    logger.info("")

    # Start session
    logger.info("Starting monitoring session...")
    if not client.start_session(patient_id='test_patient_001'):
        logger.error("Failed to start session")
        client.disconnect()
        return False

    logger.info("✅ Session started")
    logger.info("")
    logger.info("Sending multimodal data...")
    logger.info("-" * 60)

    # Send data through different scenarios
    scenarios = [
        {
            "name": "Normal Patient",
            "audio": {"normal": 1.0, "coughing": 0, "crying": 0, "groaning": 0, "gasping": 0},
            "emotion": "happy",
            "hunched": 0.1,
        },
        {
            "name": "Mild Distress (coughing)",
            "audio": {"coughing": 0.7, "normal": 0.3, "crying": 0, "groaning": 0, "gasping": 0},
            "emotion": "neutral",
            "hunched": 0.3,
        },
        {
            "name": "Moderate Distress (crying)",
            "audio": {"crying": 0.6, "normal": 0.4, "coughing": 0, "groaning": 0, "gasping": 0},
            "emotion": "sad",
            "hunched": 0.5,
        },
        {
            "name": "Severe Distress (gasping + crying)",
            "audio": {"gasping": 0.5, "crying": 0.3, "normal": 0.2, "coughing": 0, "groaning": 0},
            "emotion": "fearful",
            "hunched": 0.7,
        },
    ]

    for scenario in scenarios:
        logger.info(f"\nScenario: {scenario['name']}")

        audio_chunk = np.random.randn(16000).astype(np.float32)

        face_data = {
            "detected": True,
            "emotion": scenario["emotion"],
            "emotion_confidence": 0.8,
            "mouth_open": 0.2
        }

        eye_data = {
            "detected": True,
            "blink_rate": 20,
            "eye_openness": 0.7,
            "pupil_dilation": 0.5
        }

        posture_data = {
            "detected": True,
            "hunched": scenario["hunched"],
            "head_down": 0.1,
            "movement_intensity": 0.5,
            "hand_to_face": False
        }

        client.send_data(
            audio=audio_chunk,
            face=face_data,
            eye=eye_data,
            posture=posture_data
        )

        time.sleep(0.5)  # Wait for response

    logger.info("")
    logger.info("-" * 60)
    logger.info("Ending session...")
    client.end_session()
    time.sleep(0.5)

    client.disconnect()
    logger.info("✅ Disconnected")

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Results received: {len(results_received)}")
    logger.info(f"Alerts generated: {len(alerts_received)}")

    if len(results_received) > 0:
        logger.info("")
        logger.info("✅ Backend is working correctly!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run: python backend/mediapipe_integration.py")
        logger.info("   (to use webcam for face/eye/posture detection)")
        logger.info("2. Or integrate the client with your own detection system")
        logger.info("3. See INTEGRATION_GUIDE.md for more examples")
        return True
    else:
        logger.error("❌ No results received from backend")
        return False


if __name__ == '__main__':
    success = test_backend()
    sys.exit(0 if success else 1)
