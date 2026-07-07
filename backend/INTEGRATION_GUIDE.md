# Integration Guide: Face, Eye, Posture Detection with Backend

This guide shows how to integrate face, eye, and posture detection with the distress monitoring backend.

## Quick Start

### 1. Install Dependencies

```bash
pip install mediapipe opencv-python sounddevice librosa
```

### 2. Start Backend Server

```bash
cd backend
python server.py
```

The server will run on `http://localhost:5000`

### 3. Run Detection Client

Option A - Using the MediaPipe integration (recommended):

```bash
cd backend
python mediapipe_integration.py
```

Option B - Custom integration:

See examples below for your specific use case.

---

## Architecture

```
┌──────────────────────────┐
│   Camera & Microphone    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   MediaPipe Detector     │
│  - Face Detection        │
│  - Pose Estimation       │
│  - Eye Tracking          │
└────────────┬─────────────┘
             │ Vision Data
             ▼
┌──────────────────────────┐       ┌──────────────────┐
│   Backend Client         │───►   │  Backend Server  │
│  - Audio Processing      │       │  - Score Calc    │
│  - Data Aggregation      │       │  - Alerts        │
└──────────────────────────┘       └──────────────────┘
```

---

## Integration Methods

### Method 1: Using MediaPipe Integration Module (Recommended)

The `mediapipe_integration.py` module provides everything you need.

**Basic Setup:**

```python
import cv2
from mediapipe_integration import MediaPipeDetector
from backend.client import DistressMonitoringClient

# Initialize
detector = MediaPipeDetector()
client = DistressMonitoringClient('http://localhost:5000')
client.connect()
client.start_session(patient_id='patient_001')

# Process video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect all features
    data = detector.process_frame(frame)

    # Send to backend
    client.send_vision(
        face=data['face'],
        eye=data['eye'],
        posture=data['posture']
    )

    cv2.imshow('Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.end_session()
client.disconnect()
```

---

### Method 2: Custom Face Detection with OpenCV

If you want to use your own face detection models:

```python
import cv2
import numpy as np
from backend.client import DistressMonitoringClient

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

client = DistressMonitoringClient('http://localhost:5000')
client.connect()
client.start_session()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_data = {"detected": len(faces) > 0}
    
    if faces:
        # Extract features
        face_data.update({
            "emotion": "neutral",  # Use emotion classifier here
            "emotion_confidence": 0.7,
            "mouth_open": 0.3,
            "bbox": {
                "x": faces[0][0],
                "y": faces[0][1],
                "width": faces[0][2],
                "height": faces[0][3]
            }
        })

    client.send_vision(face=face_data)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.end_session()
client.disconnect()
```

---

### Method 3: Using MediaPipe Face Mesh for More Detailed Features

For finer emotion and expression detection:

```python
import mediapipe as mp
import cv2
import numpy as np
from backend.client import DistressMonitoringClient

mp_face_mesh = mp.solutions.face_mesh

client = DistressMonitoringClient('http://localhost:5000')
client.connect()
client.start_session()

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        face_data = {"detected": results.multi_face_landmarks is not None}

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Calculate features from landmarks
            # Mouth corners (indices 61, 291)
            mouth_top = landmarks[13].y
            mouth_bottom = landmarks[14].y
            mouth_open = abs(mouth_bottom - mouth_top)

            # Eye aspect ratio
            left_eye_top = landmarks[159].y
            left_eye_bottom = landmarks[145].y
            eye_aspect_ratio = abs(left_eye_bottom - left_eye_top)

            face_data.update({
                "emotion": "sad" if mouth_open > 0.05 else "neutral",
                "emotion_confidence": min(mouth_open * 2, 1.0),
                "mouth_open": min(mouth_open * 3, 1.0)
            })

        client.send_vision(face=face_data)

        cv2.imshow('Face Mesh', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
client.end_session()
client.disconnect()
```

---

### Method 4: Combining Video and Audio

Send both video and audio data to the backend:

```python
import cv2
import sounddevice as sd
import numpy as np
from backend.client import DistressMonitoringClient
from mediapipe_integration import MediaPipeDetector

detector = MediaPipeDetector()
client = DistressMonitoringClient('http://localhost:5000')
client.connect()
client.start_session(patient_id='patient_001')

cap = cv2.VideoCapture(0)

# Audio parameters
sample_rate = 16000
chunk_size = sample_rate // 2  # 0.5 second chunks

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio error: {status}")
    audio_chunk = indata.flatten().astype(np.float32)
    client.send_data(audio=audio_chunk, sample_rate=sample_rate)

# Start audio stream
stream = sd.InputStream(
    callback=audio_callback,
    samplerate=sample_rate,
    channels=1,
    blocksize=chunk_size
)
stream.start()

print("Recording and monitoring... Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get vision data
        vision_data = detector.process_frame(frame)

        # Send vision data (audio is sent via callback)
        client.send_vision(
            face=vision_data['face'],
            eye=vision_data['eye'],
            posture=vision_data['posture']
        )

        cv2.imshow('Distress Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stream.stop()
    stream.close()
    cap.release()
    cv2.destroyAllWindows()
    client.end_session()
    client.disconnect()
```

---

## Data Flow Example

Here's what happens when you send data:

```python
# 1. Capture frame from camera
frame = cv2.imread('patient.jpg')

# 2. Detect features
face_data = detector.detect_face(frame)
eye_data = detector.detect_eyes(frame)
posture_data = detector.detect_posture(frame)

# 3. Send to backend
client.send_vision(
    face=face_data,
    eye=eye_data,
    posture=posture_data
)

# 4. Backend processes and calculates distress score
# 5. Client receives alert via on_result callback

def on_result(result):
    print(f"Distress Score: {result['distress_score']:.3f}")
    print(f"Alert Level: {result['alert_level']}")
    print(f"Message: {result['alert_message']}")
```

---

## Advanced: Custom Emotion Classifier

Integrate your own emotion detection model:

```python
import torch
from backend.client import DistressMonitoringClient

# Load your emotion model
emotion_model = YourCustomEmotionModel()

client = DistressMonitoringClient('http://localhost:5000')
client.connect()
client.start_session()

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(...)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        # Run custom emotion classifier
        emotion, confidence = emotion_model.predict(face_roi)

        face_data = {
            "detected": True,
            "emotion": emotion,
            "emotion_confidence": confidence,
            "mouth_open": estimate_mouth_open(face_roi)
        }

        client.send_vision(face=face_data)

cap.release()
client.end_session()
client.disconnect()
```

---

## Performance Optimization

### 1. Frame Skipping

Process every Nth frame to reduce latency:

```python
frame_count = 0
skip_rate = 2  # Process every 2nd frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_rate != 0:
        continue

    # Process frame
    data = detector.process_frame(frame)
    client.send_vision(**data)
```

### 2. Multi-threading

Separate capture and processing:

```python
import threading
import queue

frame_queue = queue.Queue(maxsize=2)

def capture_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

def process_frames():
    while True:
        frame = frame_queue.get()
        data = detector.process_frame(frame)
        client.send_vision(**data)

capture_thread = threading.Thread(target=capture_frames, daemon=True)
process_thread = threading.Thread(target=process_frames, daemon=True)
capture_thread.start()
process_thread.start()
```

### 3. Resize Frames

Process smaller frames for speed:

```python
scale = 0.5
resized = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
data = detector.process_frame(resized)
```

---

## Troubleshooting

### Issue: Low FPS

- Reduce frame resolution
- Skip frames (process every 2nd frame)
- Use GPU acceleration if available
- Profile with `cProfile`

### Issue: Detector not detecting face

- Ensure good lighting
- Check camera calibration
- Lower detection confidence threshold
- Increase face size relative to frame

### Issue: Connection refused

- Ensure backend server is running
- Check firewall settings
- Verify correct server URL and port

### Issue: High latency

- Reduce model complexity
- Process frames in separate thread
- Use UDP instead of WebSocket (see advanced)

---

## Testing

Test your integration without the backend:

```python
import cv2
from mediapipe_integration import MediaPipeDetector

detector = MediaPipeDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    data = detector.process_frame(frame)

    print(f"Face detected: {data['face']['detected']}")
    print(f"Emotion: {data['face'].get('emotion', 'N/A')}")
    print(f"Posture hunched: {data['posture'].get('hunched', 0):.2f}")
    print()

    cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Next Steps

1. ✅ Set up backend server
2. ✅ Choose integration method above
3. ✅ Install required dependencies
4. ✅ Run your integration script
5. 📊 Monitor backend dashboard at `http://localhost:5000/stats`
6. 🔧 Tune thresholds for your use case
7. 📈 Collect and analyze data

---

## More Information

- **Backend README**: `backend/README.md`
- **Client Documentation**: `backend/client.py`
- **Distress Calculator**: `backend/distress_calculator.py`
- **MediaPipe Docs**: https://mediapipe.dev/
- **OpenCV Docs**: https://opencv.org/

