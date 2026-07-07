"""
Complete Webcam + Microphone Integration
Detects face, eyes, posture and sends to backend with audio
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import time
import threading
import queue
from backend.client import DistressMonitoringClient

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("WARNING: mediapipe not installed. Install with: pip install mediapipe")

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("WARNING: sounddevice not installed. Install with: pip install sounddevice")


class WebcamDistressMonitor:
    """Real-time distress monitoring using webcam and microphone"""
    
    def __init__(self, server_url='http://localhost:5000', patient_id='webcam_patient'):
        self.server_url = server_url
        self.patient_id = patient_id
        self.running = False
        
        # Audio settings
        self.sample_rate = 16000
        self.audio_chunk_size = 16000  # 1 second
        
        # Audio queue for threading
        self.audio_queue = queue.Queue(maxsize=5)
        
        # Stats
        self.frame_count = 0
        self.results_received = 0
        self.last_score = 0
        self.last_level = 'normal'
        
        # Client
        self.client = None
        
        # MediaPipe
        if HAS_MEDIAPIPE:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
    def on_result(self, result):
        """Handle results from backend"""
        self.results_received += 1
        self.last_score = result.get('distress_score', 0)
        self.last_level = result.get('alert_level', 'normal')
        
        # Print alert if concerning
        if self.last_level in ['moderate', 'severe']:
            print(f"\n*** ALERT: {self.last_level.upper()} - Score: {self.last_score:.3f} ***\n")
    
    def on_alert(self, alert):
        """Handle alerts from backend"""
        print(f"\n!!! ALERT: {alert['level'].upper()} - {alert['message'][:60]} !!!\n")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio capture callback"""
        if status:
            print(f"Audio status: {status}")
        audio_chunk = indata.flatten().astype(np.float32)
        try:
            self.audio_queue.put_nowait(audio_chunk)
        except queue.Full:
            pass  # Drop frame if queue full
    
    def detect_face(self, frame):
        """Detect face features using MediaPipe"""
        if not HAS_MEDIAPIPE:
            return {'detected': False}
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return {'detected': False}
        
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Get key landmarks
        nose = landmarks.landmark[1]
        mouth_top = landmarks.landmark[13]
        mouth_bottom = landmarks.landmark[14]
        left_eye_top = landmarks.landmark[159]
        left_eye_bottom = landmarks.landmark[145]
        right_eye_top = landmarks.landmark[386]
        right_eye_bottom = landmarks.landmark[374]
        
        # Calculate mouth openness
        mouth_open = abs(mouth_bottom.y - mouth_top.y)
        mouth_open = min(mouth_open * 5, 1.0)  # Normalize
        
        # Calculate eye openness (average of both eyes)
        left_eye = abs(left_eye_bottom.y - left_eye_top.y)
        right_eye = abs(right_eye_bottom.y - right_eye_top.y)
        eye_openness = (left_eye + right_eye) / 2
        eye_openness = min(eye_openness * 5, 1.0)
        
        # Simple emotion estimation based on mouth and eye
        if mouth_open > 0.5:
            emotion = "surprised"
            emotion_conf = 0.7
        elif eye_openness < 0.3:
            emotion = "sad"
            emotion_conf = 0.6
        else:
            emotion = "neutral"
            emotion_conf = 0.8
        
        return {
            'detected': True,
            'emotion': emotion,
            'emotion_confidence': emotion_conf,
            'mouth_open': mouth_open,
            'eye_openness': eye_openness
        }
    
    def detect_eyes(self, frame):
        """Detect eye features"""
        if not HAS_MEDIAPIPE:
            return {'detected': False}
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return {'detected': False}
        
        landmarks = results.multi_face_landmarks[0]
        
        # Eye openness
        left_eye_top = landmarks.landmark[159]
        left_eye_bottom = landmarks.landmark[145]
        right_eye_top = landmarks.landmark[386]
        right_eye_bottom = landmarks.landmark[374]
        
        left_eye = abs(left_eye_bottom.y - left_eye_top.y)
        right_eye = abs(right_eye_bottom.y - right_eye_top.y)
        eye_openness = (left_eye + right_eye) / 2
        eye_openness = min(eye_openness * 5, 1.0)
        
        # Estimate blink rate (simplified - count eye closes)
        blink_rate = 17.5  # Normal default
        
        # Pupil dilation (simplified estimate)
        pupil_dilation = 0.5 + (eye_openness * 0.3)
        
        return {
            'detected': True,
            'blink_rate': blink_rate,
            'eye_openness': eye_openness,
            'pupil_dilation': pupil_dilation
        }
    
    def detect_posture(self, frame):
        """Detect posture using MediaPipe Pose"""
        if not HAS_MEDIAPIPE:
            return {'detected': False}
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return {'detected': False}
        
        landmarks = results.pose_landmarks.landmark
        
        # Key landmarks
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Calculate hunched (shoulders forward)
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        torso_length = hip_mid_y - shoulder_mid_y
        hunched = max(0, 1.0 - (torso_length * 2))
        hunched = min(max(hunched, 0), 1)
        
        # Head down
        head_down = max(0, (nose.y - shoulder_mid_y) * 2)
        head_down = min(max(head_down, 0), 1)
        
        # Hand to face
        nose_dist_left = np.sqrt((left_wrist.x - nose.x)**2 + (left_wrist.y - nose.y)**2)
        nose_dist_right = np.sqrt((right_wrist.x - nose.x)**2 + (right_wrist.y - nose.y)**2)
        hand_to_face = min(nose_dist_left, nose_dist_right) < 0.15
        
        return {
            'detected': True,
            'hunched': hunched,
            'head_down': head_down,
            'movement_intensity': 0.5,  # Simplified
            'hand_to_face': hand_to_face
        }
    
    def audio_thread(self):
        """Thread for audio processing"""
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.5)
                self.client.send_data(
                    audio=audio_chunk,
                    face=self.last_face_data,
                    eye=self.last_eye_data,
                    posture=self.last_posture_data
                )
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Audio thread error: {e}")
    
    def run(self):
        """Main monitoring loop"""
        print("=" * 60)
        print("WEBCAM + MICROPHONE DISTRESS MONITOR")
        print("=" * 60)
        
        # Connect to backend
        self.client = DistressMonitoringClient(
            self.server_url,
            on_result=self.on_result,
            on_alert=self.on_alert
        )
        
        if not self.client.connect():
            print("ERROR: Could not connect to backend server")
            print("Make sure server is running: python backend/server.py")
            return
        
        if not self.client.start_session(patient_id=self.patient_id):
            print("ERROR: Could not start session")
            return
        
        print("Connected to backend")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            self.client.end_session()
            self.client.disconnect()
            return
        
        print("Webcam opened")
        
        # Start audio
        audio_stream = None
        if HAS_AUDIO:
            try:
                audio_stream = sd.InputStream(
                    callback=self.audio_callback,
                    samplerate=self.sample_rate,
                    channels=1,
                    blocksize=self.audio_chunk_size
                )
                audio_stream.start()
                print("Microphone started")
            except Exception as e:
                print(f"WARNING: Could not start microphone: {e}")
                audio_stream = None
        
        # Start audio thread
        self.running = True
        self.last_face_data = {}
        self.last_eye_data = {}
        self.last_posture_data = {}
        
        audio_thread = threading.Thread(target=self.audio_thread, daemon=True)
        audio_thread.start()
        
        print("\nMonitoring... Press 'q' to quit\n")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect features
                face_data = self.detect_face(frame)
                eye_data = self.detect_eyes(frame)
                posture_data = self.detect_posture(frame)
                
                self.last_face_data = face_data
                self.last_eye_data = eye_data
                self.last_posture_data = posture_data
                
                # Draw overlays
                if face_data.get('detected'):
                    cv2.putText(frame, f"Emotion: {face_data.get('emotion', 'N/A')}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw score
                color = (0, 255, 0) if self.last_level == 'normal' else (0, 165, 255) if self.last_level == 'mild' else (0, 0, 255)
                cv2.putText(frame, f"Score: {self.last_score:.3f} ({self.last_level})", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Results: {self.results_received}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw face mesh if available
                if HAS_MEDIAPIPE:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_results = self.face_mesh.process(rgb)
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
                                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1)
                            )
                
                cv2.imshow('Distress Monitor', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Also send vision-only data (audio sent in thread)
                self.client.send_vision(
                    face=face_data,
                    eye=eye_data,
                    posture=posture_data
                )
                
                self.frame_count += 1
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.running = False
            if audio_stream:
                audio_stream.stop()
                audio_stream.close()
            cap.release()
            cv2.destroyAllWindows()
            self.client.end_session()
            self.client.disconnect()
            
            print(f"\nSession complete:")
            print(f"  Frames processed: {self.frame_count}")
            print(f"  Results received: {self.results_received}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Webcam Distress Monitor')
    parser.add_argument('--server', default='http://localhost:5000', help='Backend server URL')
    parser.add_argument('--patient', default='webcam_patient', help='Patient ID')
    args = parser.parse_args()
    
    monitor = WebcamDistressMonitor(
        server_url=args.server,
        patient_id=args.patient
    )
    monitor.run()
