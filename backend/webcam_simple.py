"""
Webcam monitor with real face detection using OpenCV
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import time
from backend.client import DistressMonitoringClient

print("=" * 60)
print("WEBCAM DISTRESS MONITOR - FACE DETECTION")
print("=" * 60)

results = []

def on_result(r):
    results.append(r)
    score = r.get('distress_score', 0)
    level = r.get('alert_level', 'unknown')
    scores = r.get('individual_scores', {})
    print(f"  Score: {score:.3f} | Level: {level:8} | Face: {scores.get('face', 0):.3f}")

client = DistressMonitoringClient('http://localhost:5000', on_result=on_result)

if not client.connect():
    print("ERROR: Could not connect")
    exit(1)
print("Connected!")

if not client.start_session(patient_id='webcam_patient'):
    print("ERROR: Could not start session")
    exit(1)
print("Session started!")

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit(1)
print("Webcam opened!")
print("\nMonitoring... Press 'q' to quit\n")

frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_data = {'detected': False, 'emotion': 'neutral', 'emotion_confidence': 0.5, 'mouth_open': 0.1}
        eye_data = {'detected': False, 'blink_rate': 17, 'eye_openness': 0.7, 'pupil_dilation': 0.5}
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            face_data['detected'] = True
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)
            eye_data['detected'] = len(eyes) > 0
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            # Detect smile
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
            if len(smiles) > 0:
                face_data['emotion'] = 'happy'
                face_data['emotion_confidence'] = 0.7
                face_data['mouth_open'] = 0.6
            else:
                # Check mouth region
                mouth_y = int(h * 0.6)
                mouth_h = int(h * 0.3)
                mouth_region = roi_gray[mouth_y:mouth_y+mouth_h, :]
                if mouth_region.size > 0:
                    mouth_brightness = np.mean(mouth_region)
                    if mouth_brightness < 100:
                        face_data['mouth_open'] = 0.5
                        face_data['emotion'] = 'sad'
                        face_data['emotion_confidence'] = 0.6
                    else:
                        face_data['mouth_open'] = 0.2
        
        posture_data = {
            'detected': len(faces) > 0,
            'hunched': 0.3,
            'head_down': 0.1,
            'movement_intensity': 0.5,
            'hand_to_face': False
        }
        
        client.send_vision(face=face_data, eye=eye_data, posture=posture_data)
        
        # Draw info
        score = results[-1].get('distress_score', 0) if results else 0
        level = results[-1].get('alert_level', 'waiting') if results else 'waiting'
        color = (0, 255, 0) if level == 'normal' else (0, 165, 255) if level == 'mild' else (0, 0, 255)
        
        cv2.putText(frame, f"Score: {score:.3f} ({level})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Emotion: {face_data['emotion']} | Eyes: {'Y' if eye_data['detected'] else 'N'}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Distress Monitor', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    client.end_session()
    client.disconnect()
    print(f"\nDone! {frame_count} frames, {len(results)} results")
