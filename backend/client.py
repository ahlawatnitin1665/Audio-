"""
Client for communicating with the distress monitoring backend
Sends audio, face, eye, and posture data via WebSocket
"""
import logging
import numpy as np
from typing import Dict, Callable, Optional
import socketio
import json
import time
import threading

logger = logging.getLogger(__name__)


class DistressMonitoringClient:
    """
    Client for sending multimodal data to the distress monitoring backend.
    
    Sends synchronized multimodal data with timestamps for real-time processing.
    
    Usage:
        client = DistressMonitoringClient('http://localhost:5000')
        client.connect()
        client.start_session(patient_id='patient_123')
        
        # Send data (all modalities together in one synchronized frame)
        client.send_data(
            audio=audio_chunk,
            face={'detected': True, 'emotion': 'sad', 'emotion_confidence': 0.8, 'mouth_open': 0.3},
            eye={'detected': True, 'blink_rate': 25, 'eye_openness': 0.7, 'pupil_dilation': 0.6},
            posture={'detected': True, 'hunched': 0.3, 'head_down': 0.1, 'movement_intensity': 0.5, 'hand_to_face': False}
        )
        
        client.end_session()
        client.disconnect()
    """

    def __init__(
        self,
        server_url: str = 'http://localhost:5000',
        on_result: Callable = None,
        on_alert: Callable = None,
        on_error: Callable = None,
        frame_rate: int = 10  # Frames per second
    ):
        """
        Initialize client.
        
        Args:
            server_url: URL of the backend server
            on_result: Callback for analysis results
            on_alert: Callback for alerts
            on_error: Callback for errors
            frame_rate: Expected frame rate in FPS (for timing validation)
        """
        self.server_url = server_url
        self.on_result = on_result
        self.on_alert = on_alert
        self.on_error = on_error
        self.frame_rate = frame_rate
        self.frame_interval = 1.0 / frame_rate  # Seconds between frames

        self.sio = socketio.Client()
        self.connected = False
        self.session_active = False
        self.client_id = None
        self._session_event = threading.Event()
        
        # Frame tracking
        self.frame_counter = 0
        self.last_send_time = 0
        self.session_start_time = None

        # Register event handlers
        self._register_handlers()

    @staticmethod
    def _to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: DistressMonitoringClient._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DistressMonitoringClient._to_serializable(v) for v in obj]
        return obj

    def _register_handlers(self):
        """Register socketio event handlers"""

        @self.sio.on('connection_response')
        def on_connection(data):
            logger.info(f"Connected: {data}")
            self.connected = True
            self.client_id = data.get('client_id')

        @self.sio.on('session_started')
        def on_session_started(data):
            logger.info(f"Session started: {data}")
            self.session_active = True
            self._session_event.set()

        @self.sio.on('session_ended')
        def on_session_ended(data):
            logger.info(f"Session ended: {data}")
            self.session_active = False

        @self.sio.on('analysis_result')
        def on_analysis_result(data):
            logger.warning(f"ANALYSIS_RESULT EVENT RECEIVED: score={data.get('distress_score', '?')}")
            if self.on_result:
                self.on_result(data)
            else:
                logger.warning("on_result callback is None!")

        @self.sio.on('audio_result')
        def on_audio_result(data):
            logger.warning(f"AUDIO_RESULT EVENT RECEIVED")
            if self.on_result:
                self.on_result(data)

        @self.sio.on('vision_result')
        def on_vision_result(data):
            logger.warning(f"VISION_RESULT EVENT RECEIVED")
            if self.on_result:
                self.on_result(data)

        @self.sio.on('alert')
        def on_alert(data):
            logger.warning(f"ALERT EVENT RECEIVED: {data['level']}")
            if self.on_alert:
                self.on_alert(data)

        @self.sio.on('error')
        def on_error(data):
            logger.error(f"ERROR EVENT RECEIVED: {data}")
            if self.on_error:
                self.on_error(data)

    def connect(self, timeout: int = 10) -> bool:
        """
        Connect to the backend server.
        
        Args:
            timeout: Connection timeout in seconds
        
        Returns:
            True if connected, False otherwise
        """
        try:
            logger.info(f"Connecting to {self.server_url}...")
            self.sio.connect(self.server_url, wait_timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from the backend server"""
        if self.connected:
            self.sio.disconnect()
            self.connected = False
            logger.info("Disconnected from server")

    def start_session(self, patient_id: str = None, timeout: int = 5) -> bool:
        """
        Start a monitoring session.
        
        Args:
            patient_id: Optional patient identifier
            timeout: Timeout in seconds to wait for session to start
        
        Returns:
            True if session started, False otherwise
        """
        if not self.connected:
            logger.error("Not connected to server")
            return False

        try:
            data = {'patient_id': patient_id or self.client_id}
            self._session_event.clear()
            self.sio.emit('start_session', data)
            
            # Wait for session to be activated (non-blocking)
            if self._session_event.wait(timeout=timeout):
                logger.info("Session activated successfully")
                return True
            else:
                logger.error("Session activation timeout")
                return False
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return False

    def end_session(self) -> bool:
        """
        End the current monitoring session.
        
        Returns:
            True if session ended, False otherwise
        """
        if not self.connected:
            logger.error("Not connected to server")
            return False

        try:
            self.sio.emit('end_session')
            return True
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return False

    def send_data(
        self,
        audio: np.ndarray = None,
        face: Dict = None,
        eye: Dict = None,
        posture: Dict = None,
        sample_rate: int = 16000,
        capture_timestamp: float = None,
        frame_id: int = None
    ) -> bool:
        """
        Send multimodal data to the backend.
        
        Args:
            audio: Audio chunk (numpy array)
            face: Face data dict
            eye: Eye data dict
            posture: Posture data dict
            sample_rate: Audio sample rate
            capture_timestamp: Unix timestamp when data was captured (optional, auto-generated)
            frame_id: Sequential frame identifier for sync tracking
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.connected:
            logger.error("Not connected to server")
            return False
            
        if not self.session_active:
            logger.error("Session not active")
            return False

        try:
            # Generate timestamp if not provided
            if capture_timestamp is None:
                capture_timestamp = time.time()
            
            # Generate frame ID if not provided
            if frame_id is None:
                if not hasattr(self, '_frame_counter'):
                    self._frame_counter = 0
                frame_id = self._frame_counter
                self._frame_counter += 1

            data = {
                'metadata': {
                    'capture_timestamp': capture_timestamp,
                    'frame_id': frame_id
                }
            }

            if audio is not None:
                # Convert audio to list for JSON serialization
                if isinstance(audio, np.ndarray):
                    # Convert to Python list and sample down to reduce size
                    audio_list = audio[:8000].tolist()  # Take first 0.5 seconds
                else:
                    audio_list = audio[:8000]
                data['audio'] = {
                    'chunk': audio_list,
                    'sample_rate': sample_rate
                }

            if face:
                data['face'] = face

            if eye:
                data['eye'] = eye

            if posture:
                data['posture'] = posture

            # Convert all numpy types to native Python types
            data = self._to_serializable(data)

            logger.debug(f"Emitting frame {frame_id} at {capture_timestamp:.3f}")
            self.sio.emit('data', data)
            logger.debug("Data emitted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def send_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Send audio data only.
        
        Args:
            audio: Audio chunk (numpy array)
            sample_rate: Audio sample rate
        
        Returns:
            True if sent successfully, False otherwise
        """
        if isinstance(audio, np.ndarray):
            audio = audio.tolist()

        try:
            self.sio.emit('audio', {
                'chunk': audio,
                'sample_rate': sample_rate
            })
            return True
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")
            return False

    def send_vision(
        self,
        face: Dict = None,
        eye: Dict = None,
        posture: Dict = None
    ) -> bool:
        """
        Send vision data (face, eye, posture).
        
        Args:
            face: Face data dict
            eye: Eye data dict
            posture: Posture data dict
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            data = {}
            if face:
                data['face'] = face
            if eye:
                data['eye'] = eye
            if posture:
                data['posture'] = posture

            data = self._to_serializable(data)
            self.sio.emit('vision', data)
            return True
        except Exception as e:
            logger.error(f"Failed to send vision data: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to server"""
        return self.connected

    def is_session_active(self) -> bool:
        """Check if session is active"""
        return self.session_active


# Example usage function
def example_usage():
    """Example of how to use the client"""

    def on_result(result):
        print(f"Result: Distress score = {result.get('distress_score', 'N/A'):.3f}")
        print(f"  Alert: {result.get('alert_level')} - {result.get('alert_message')}")

    def on_alert(alert):
        print(f"\n🚨 ALERT: {alert['level']} - {alert['message']}")

    def on_error(error):
        print(f"Error: {error}")

    # Create client
    client = DistressMonitoringClient(
        server_url='http://localhost:5000',
        on_result=on_result,
        on_alert=on_alert,
        on_error=on_error
    )

    # Connect and start session
    if not client.connect():
        print("Failed to connect")
        return

    if not client.start_session(patient_id='patient_001'):
        print("Failed to start session")
        client.disconnect()
        return

    # Simulate sending data
    print("Sending sample data...")
    for i in range(5):
        # Simulate audio data
        audio = np.random.randn(16000)  # 1 second of audio at 16kHz

        # Simulate face data
        face = {
            'detected': True,
            'emotion': 'sad' if i > 2 else 'neutral',
            'emotion_confidence': 0.8,
            'mouth_open': 0.2
        }

        # Simulate eye data
        eye = {
            'detected': True,
            'blink_rate': 20 + i * 2,
            'eye_openness': 0.7,
            'pupil_dilation': 0.5
        }

        # Simulate posture data
        posture = {
            'detected': True,
            'hunched': 0.3,
            'head_down': 0.1,
            'movement_intensity': 0.5,
            'hand_to_face': False
        }

        client.send_data(
            audio=audio,
            face=face,
            eye=eye,
            posture=posture
        )

        import time
        time.sleep(1)

    # End session
    print("\nEnding session...")
    client.end_session()
    client.disconnect()
    print("Done!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    example_usage()
