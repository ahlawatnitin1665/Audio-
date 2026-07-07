# Patient Distress Monitoring Backend

A modular backend system for monitoring patient distress by combining audio, face, eye, and posture detection.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client (WebSocket)                   │
│  - Camera (Face/Eye/Posture Detection)                  │
│  - Microphone (Audio)                                   │
└─────────────────────────────────────────────────────────┘
                            ↓
                        WebSocket
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  Backend Server (Flask)                 │
│  ┌─────────────────────────────────────────────────────┐│
│  │  Data Ingestion                                     ││
│  │  - Audio processor                                  ││
│  │  - Vision data aggregator                           ││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │  Combined Distress Calculator                       ││
│  │  - Audio score (40%)                                ││
│  │  - Face score (30%)                                 ││
│  │  - Eye score (15%)                                  ││
│  │  - Posture score (15%)                              ││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │  Alert System                                       ││
│  │  - Normal / Mild / Moderate / Severe                ││
│  │  - Alert suppression & history                      ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  Alert Notifications                    │
│  - Real-time WebSocket events                           │
│  - REST API for historical data                         │
└─────────────────────────────────────────────────────────┘
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m pip install python-socketio python-engineio
```

### 2. Ensure Audio Model is Available

The backend looks for:
- `checkpoints/best_model.pth` - Audio classification model
- `checkpoints/simple_denoiser.pth` - Audio denoiser model

## Running the Backend

### Option 1: Simple Start

```bash
cd backend
python server.py
```

Server will start at `http://localhost:5000`

### Option 2: With Custom Configuration

```bash
export SERVER_PORT=5000
export DEBUG=False
export LOG_LEVEL=INFO
cd backend
python server.py
```

### Option 3: Using Gunicorn (Production)

```bash
pip install gunicorn python-socketio python-engineio
cd backend
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 server:app
```

## Client Usage

### Basic Example

```python
from backend.client import DistressMonitoringClient
import numpy as np

def on_result(result):
    print(f"Distress: {result['distress_score']:.3f}")
    print(f"Alert: {result['alert_level']}")

def on_alert(alert):
    print(f"🚨 {alert['level']}: {alert['message']}")

# Connect to backend
client = DistressMonitoringClient(
    server_url='http://localhost:5000',
    on_result=on_result,
    on_alert=on_alert
)

client.connect()
client.start_session(patient_id='patient_001')

# Send data
audio_chunk = np.random.randn(16000)  # 1 second at 16kHz

face_data = {
    'detected': True,
    'emotion': 'sad',
    'emotion_confidence': 0.8,
    'mouth_open': 0.2
}

eye_data = {
    'detected': True,
    'blink_rate': 25,
    'eye_openness': 0.7,
    'pupil_dilation': 0.5
}

posture_data = {
    'detected': True,
    'hunched': 0.3,
    'head_down': 0.1,
    'movement_intensity': 0.5,
    'hand_to_face': False
}

client.send_data(
    audio=audio_chunk,
    face=face_data,
    eye=eye_data,
    posture=posture_data
)

client.end_session()
client.disconnect()
```

## Data Structures

### Audio Data

```json
{
  "audio": {
    "chunk": [float, float, ...],  // Audio samples
    "sample_rate": 16000            // Sample rate in Hz
  }
}
```

### Face Data

```json
{
  "face": {
    "detected": true,
    "emotion": "sad",                   // angry, sad, fearful, disgusted, surprised, neutral, happy
    "emotion_confidence": 0.85,         // 0-1
    "mouth_open": 0.3                   // 0-1, how open is mouth
  }
}
```

### Eye Data

```json
{
  "eye": {
    "detected": true,
    "blink_rate": 20,                   // Blinks per minute
    "eye_openness": 0.7,                // 0-1
    "pupil_dilation": 0.5               // 0-1
  }
}
```

### Posture Data

```json
{
  "posture": {
    "detected": true,
    "hunched": 0.3,                     // 0-1, how hunched
    "head_down": 0.1,                   // 0-1
    "movement_intensity": 0.5,          // 0-1
    "hand_to_face": false               // Boolean
  }
}
```

## WebSocket Events

### Client → Server

- `start_session(data)` - Start monitoring session
- `end_session()` - End monitoring session
- `data(data)` - Send combined multimodal data
- `audio(data)` - Send audio data only
- `vision(data)` - Send vision data only

### Server → Client

- `connection_response(data)` - Connection established
- `session_started(data)` - Session started
- `session_ended(data)` - Session ended with summary
- `analysis_result(data)` - Analysis result with distress score
- `audio_result(data)` - Audio-only analysis result
- `vision_result(data)` - Vision-only analysis result
- `alert(data)` - Alert generated
- `error(data)` - Error occurred

## REST API Endpoints

### Health Check
```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-06-23T10:30:45.123456",
  "connected_clients": 5,
  "active_sessions": 3
}
```

### Statistics
```
GET /stats
```

Response:
```json
{
  "connected_clients": 5,
  "active_sessions": 3,
  "alerts": {
    "total_alerts": 42,
    "normal": 20,
    "mild": 15,
    "moderate": 5,
    "severe": 2
  },
  "calculator_history_length": 150
}
```

### Recent Alerts
```
GET /alerts?limit=10
```

### Reset System
```
POST /reset
```

## Distress Score Calculation

### Component Weights
- **Audio**: 40% - Primary indicator of distress
- **Face**: 30% - Emotional expression analysis
- **Eye**: 15% - Eye movement and blinking patterns
- **Posture**: 15% - Body language and posture

### Alert Levels
- **Normal**: 0.0 - 0.3 (Info priority)
- **Mild**: 0.3 - 0.5 (Low priority)
- **Moderate**: 0.5 - 0.7 (Medium priority)
- **Severe**: 0.7 - 1.0 (High priority)

### Audio Classification Importance
- **Crying**: 0.95
- **Gasping**: 0.90
- **Coughing**: 0.80
- **Groaning**: 0.70
- **Normal**: 0.00

## Integration with Face/Eye/Posture Detection

You can use any face, eye, and posture detection library:

- **MediaPipe**: `pip install mediapipe`
- **OpenCV**: `pip install opencv-python`
- **dlib**: `pip install dlib`
- **Custom models**: Integrate your own detection models

Example with MediaPipe:

```python
import mediapipe as mp
import cv2
from backend.client import DistressMonitoringClient

# Setup MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

client = DistressMonitoringClient('http://localhost:5000')
client.connect()
client.start_session(patient_id='patient_001')

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection() as face_detection:
    with mp_pose.Pose() as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detections
            face_results = face_detection.process(frame)
            pose_results = pose.process(frame)

            # Extract data and send to backend
            face_data = extract_face_data(face_results)
            posture_data = extract_posture_data(pose_results)

            client.send_vision(face=face_data, posture=posture_data)

cap.release()
client.end_session()
client.disconnect()
```

## Testing

Run the example client:

```bash
cd backend
python client.py
```

This will simulate sending data to the backend for 5 seconds.

## Logs

Backend logs are written to `logs/backend.log` or stdout based on configuration.

## Performance Notes

- **Latency**: ~50-100ms per analysis (varies with model)
- **Throughput**: Can handle 10+ concurrent sessions
- **Memory**: ~500MB base + ~50MB per concurrent session
- **CPU**: Optimized for CPU inference; GPU support available

## Troubleshooting

### "Connection refused"
- Ensure backend server is running
- Check firewall settings
- Verify server URL is correct

### "No active session"
- Call `start_session()` before sending data
- Ensure client is connected

### Audio model not found
- Check that `checkpoints/best_model.pth` exists
- Download model if needed

### High latency
- Reduce data processing load
- Use GPU if available
- Increase WebSocket buffer size in config

## License

MIT
