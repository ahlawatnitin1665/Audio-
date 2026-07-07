"""
MediaPipe Integration Module
Provides face, eye, and posture detection using MediaPipe
Can be used with the client to send data to the backend
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MediaPipeDetector:
    """
    Unified detector for face, eye, and posture using MediaPipe
    """

    def __init__(self):
        """Initialize MediaPipe components"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full body, 0 for short range
            min_detection_confidence=0.5
        )

        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # For blink detection
        self.blink_history = []
        self.frame_count = 0

    def detect_face(self, frame: np.ndarray) -> Dict:
        """
        Detect face and extract features.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Dict with face detection data
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)

        face_data = {"detected": False}

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            # Face detected
            face_data["detected"] = True
            face_data["bbox"] = {
                "x": bbox.xmin,
                "y": bbox.ymin,
                "width": bbox.width,
                "height": bbox.height
            }
            face_data["confidence"] = detection.score[0]

            # Estimate emotion (very basic - based on mouth openness)
            # In production, use a proper emotion classifier
            mouth_openness = self._estimate_mouth_openness(frame, bbox)
            face_data["mouth_open"] = mouth_openness
            face_data["emotion"] = self._estimate_emotion(mouth_openness)
            face_data["emotion_confidence"] = 0.5  # Placeholder

        return face_data

    def _estimate_mouth_openness(self, frame: np.ndarray, bbox) -> float:
        """
        Estimate how open the mouth is.
        This is a simplified estimate based on face region.
        
        In production, use facial landmarks or dedicated mouth detection.
        
        Returns:
            Value between 0 and 1, where 1 = mouth wide open
        """
        # This is a placeholder - in production use MediaPipe face mesh
        # for actual landmark detection
        h, w = frame.shape[:2]
        
        # Estimate mouth region based on face bbox
        y_mouth = int((bbox.ymin + 0.6 * bbox.height) * h)
        x_mouth = int(bbox.xmin * w)
        mouth_width = int(bbox.width * w)
        mouth_height = int(bbox.height * 0.15 * h)

        if y_mouth < 0 or y_mouth + mouth_height > h:
            return 0.3

        mouth_region = frame[y_mouth:y_mouth + mouth_height,
                             x_mouth:x_mouth + mouth_width]

        if mouth_region.size == 0:
            return 0.3

        # Calculate brightness difference (open mouth typically darker)
        gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        darkness = 1.0 - (np.mean(gray) / 255.0)

        return min(1.0, darkness * 1.5)

    def _estimate_emotion(self, mouth_openness: float) -> str:
        """Estimate emotion based on facial features"""
        if mouth_openness > 0.6:
            return "surprised"
        elif mouth_openness > 0.4:
            return "fearful"
        else:
            return "neutral"

    def detect_eyes(self, frame: np.ndarray) -> Dict:
        """
        Detect eyes and extract features.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Dict with eye detection data
        """
        eye_data = {"detected": False, "blink_rate": 0.0, "eye_openness": 0.5}

        # Use face detection result to crop face region
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detector.process(rgb_frame)

        if not face_results.detections:
            return eye_data

        # For now, use simple heuristics
        # In production, use dedicated eye detection models
        eye_data["detected"] = True

        # Estimate blink rate (in production, track actual blinks)
        # Simulate based on frame count
        self.frame_count += 1
        if self.frame_count % 10 == 0:  # Check every 10 frames
            blink_rate = 15 + np.random.randint(-5, 10)  # 15-25 blinks/min
            self.blink_history.append(blink_rate)
            if len(self.blink_history) > 10:
                self.blink_history.pop(0)
            eye_data["blink_rate"] = np.mean(self.blink_history)

        # Estimate eye openness (0-1, where 1 = fully open)
        eye_data["eye_openness"] = 0.7 + np.random.uniform(-0.1, 0.1)

        # Pupil dilation (0-1)
        eye_data["pupil_dilation"] = 0.5 + np.random.uniform(-0.1, 0.1)

        return eye_data

    def detect_posture(self, frame: np.ndarray) -> Dict:
        """
        Detect posture using pose estimation.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Dict with posture detection data
        """
        posture_data = {"detected": False}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb_frame)

        if not results.pose_landmarks:
            return posture_data

        posture_data["detected"] = True

        # Extract key landmarks
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        # Convert landmarks to pixel coordinates
        points = {}
        for name, landmark in zip(
            ["nose", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            [landmarks[0], landmarks[1], landmarks[5], landmarks[6], landmarks[23], landmarks[24]]
        ):
            points[name] = (int(landmark.x * w), int(landmark.y * h))

        # Calculate posture features
        posture_data["hunched"] = self._calculate_hunched(landmarks)
        posture_data["head_down"] = self._calculate_head_down(landmarks)
        posture_data["movement_intensity"] = np.random.uniform(0.3, 0.7)  # Placeholder
        posture_data["hand_to_face"] = self._detect_hand_to_face(landmarks)

        return posture_data

    def _calculate_hunched(self, landmarks) -> float:
        """
        Calculate how hunched the posture is.
        Returns value between 0 and 1.
        """
        try:
            # Use shoulder and hip positions
            left_shoulder = landmarks[11]  # Left shoulder
            right_shoulder = landmarks[12]  # Right shoulder
            left_hip = landmarks[23]  # Left hip
            right_hip = landmarks[24]  # Right hip

            # Calculate shoulder-hip distance
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2

            # Hunched if shoulders are too close to hips
            hunched_score = max(0.0, 1.0 - (hip_y - shoulder_y) * 2)
            return min(1.0, hunched_score)
        except:
            return 0.5

    def _calculate_head_down(self, landmarks) -> float:
        """
        Calculate how much the head is tilted down.
        Returns value between 0 and 1.
        """
        try:
            # Use nose and neck positions
            nose = landmarks[0]
            neck = landmarks[1]

            # Head down if nose is significantly lower than reference
            head_down_score = max(0.0, (nose.y - neck.y) * 3)
            return min(1.0, head_down_score)
        except:
            return 0.5

    def _detect_hand_to_face(self, landmarks) -> bool:
        """
        Detect if hands are near the face.
        Returns boolean.
        """
        try:
            nose = landmarks[0]
            left_wrist = landmarks[15]  # Left wrist
            right_wrist = landmarks[16]  # Right wrist

            # Check if wrists are near nose
            left_dist = np.sqrt((left_wrist.x - nose.x) ** 2 + (left_wrist.y - nose.y) ** 2)
            right_dist = np.sqrt((right_wrist.x - nose.x) ** 2 + (right_wrist.y - nose.y) ** 2)

            return min(left_dist, right_dist) < 0.2
        except:
            return False

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a frame and extract all features.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Dict with face, eye, and posture data
        """
        return {
            "face": self.detect_face(frame),
            "eye": self.detect_eyes(frame),
            "posture": self.detect_posture(frame)
        }

    def release(self):
        """Release resources"""
        self.face_detector.close()
        self.pose_detector.close()


# Example integration with backend client
def example_with_backend():
    """
    Example: Run detector and send data to backend
    """
    from client import DistressMonitoringClient
    import sounddevice as sd
    import librosa

    # Initialize detector
    detector = MediaPipeDetector()

    # Initialize backend client
    def on_result(result):
        print(f"Distress: {result.get('distress_score', 0):.3f} | Alert: {result.get('alert_level')}")

    client = DistressMonitoringClient(
        server_url='http://localhost:5000',
        on_result=on_result
    )

    if not client.connect():
        print("Failed to connect to backend")
        return

    client.start_session(patient_id='patient_001')

    # Open camera
    cap = cv2.VideoCapture(0)

    print("Starting detection... Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect features
            vision_data = detector.process_frame(frame)

            # Capture audio (placeholder - 1 second chunks)
            # In production, use continuous audio stream
            # audio = sd.rec(16000, samplerate=16000, channels=1, dtype='float32')
            # sd.wait()

            # Send data to backend
            client.send_vision(
                face=vision_data["face"],
                eye=vision_data["eye"],
                posture=vision_data["posture"]
            )

            # Display frame with annotations
            if vision_data["face"]["detected"]:
                bbox = vision_data["face"]["bbox"]
                h, w = frame.shape[:2]
                x1 = int(bbox["x"] * w)
                y1 = int(bbox["y"] * h)
                x2 = int((bbox["x"] + bbox["width"]) * w)
                y2 = int((bbox["y"] + bbox["height"]) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Distress Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.end_session()
        client.disconnect()
        detector.release()


if __name__ == "__main__":
    # Example with file
    print("MediaPipe integration module loaded")
    print("For usage, see example_with_backend() or README.md")
