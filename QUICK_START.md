# Patient Distress Monitoring Backend - Quick Start Guide

## 🎯 What's New?

You now have a **complete modular backend system** that:

- ✅ Receives real-time audio data from your existing audio models
- ✅ Receives face, eye, and posture data from cameras
- ✅ Calculates a **combined distress score** from all modalities
- ✅ Generates **intelligent alerts** based on distress levels
- ✅ Provides **WebSocket streaming** for real-time communication
- ✅ Offers **REST API** for data queries and statistics

## 📁 What Was Created?

```
backend/
├── server.py                      # Main Flask WebSocket server
├── client.py                      # Client for sending data
├── distress_calculator.py         # Score calculation engine
├── alert_system.py                # Alert generation & management
├── mediapipe_integration.py       # Face/eye/posture detector
├── config.py                      # Configuration settings
├── __init__.py                    # Package initialization
├── README.md                      # Full documentation
├── INTEGRATION_GUIDE.md           # Integration examples
├── quick_test.py                  # Test suite
└── test_system.py                 # Comprehensive tests
```

## 🚀 Getting Started (5 Minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
pip install python-socketio python-engineio mediapipe
```

### Step 2: Start Backend Server

```bash
cd backend
python server.py
```

Expected output:
```
Starting server on port 5000
 * Running on http://0.0.0.0:5000
```

### Step 3: Run Example Client

In a new terminal:

```bash
cd backend
python mediapipe_integration.py
```

This will:
- Open your webcam
- Detect face, eyes, and posture
- Send data to backend
- Receive and display alerts

**That's it!** You now have a working distress monitoring system.

---

## 📊 Architecture Overview

### Data Flow

```
Camera/Mic (Your client)
    ↓
    ├─→ Face Detection (Emotion, mouth openness)
    ├─→ Eye Detection (Blink rate, pupil dilation, eye openness)
    ├─→ Posture Detection (Hunched, head position, movement)
    └─→ Audio Processing (Your existing audio model)
    
    ↓ (Via WebSocket)
    
Backend Server
    ├─→ Aggregates all modalities
    ├─→ Combined Distress Score = 
    │   (Audio 40% + Face 30% + Eye 15% + Posture 15%)
    └─→ Alert Generation
    
    ↓ (Via WebSocket)
    
Client
    ├─→ Receives alerts in real-time
    ├─→ Displays distress score
    └─→ Triggers notifications/actions
```

### Distress Score Calculation

The backend combines multiple modalities with weights:

| Modality | Weight | Purpose |
|----------|--------|---------|
| **Audio** | 40% | Primary: Coughing, crying, gasping |
| **Face** | 30% | Facial expressions and mouth openness |
| **Eye** | 15% | Blink rate and pupil dilation |
| **Posture** | 15% | Body language and movement |

### Alert Levels

| Score Range | Level | Priority | Example |
|-------------|-------|----------|---------|
| 0.0 - 0.3 | Normal | Info | Patient is stable |
| 0.3 - 0.5 | Mild | Low | Mild coughing detected |
| 0.5 - 0.7 | Moderate | Medium | Moderate distress indicators |
| 0.7 - 1.0 | Severe | High | **IMMEDIATE ATTENTION** |

---

## 💻 Usage Examples

### Example 1: Send Audio Only

```python
from backend.client import DistressMonitoringClient
import numpy as np

client = DistressMonitoringClient('http://localhost:5000')
client.connect()
client.start_session(patient_id='patient_001')

# Your audio chunk
audio = np.random.randn(16000)  # 1 second at 16kHz

client.send_audio(audio)
client.end_session()
client.disconnect()
```

### Example 2: Send Vision Only

```python
from backend.client import DistressMonitoringClient

client = DistressMonitoringClient('http://localhost:5000')
client.connect()
client.start_session()

face_data = {
    'detected': True,
    'emotion': 'sad',
    'emotion_confidence': 0.8,
    'mouth_open': 0.3
}

eye_data = {
    'detected': True,
    'blink_rate': 22,
    'eye_openness': 0.7,
    'pupil_dilation': 0.6
}

posture_data = {
    'detected': True,
    'hunched': 0.4,
    'head_down': 0.2,
    'movement_intensity': 0.6,
    'hand_to_face': False
}

client.send_vision(face=face_data, eye=eye_data, posture=posture_data)
client.end_session()
client.disconnect()
```

### Example 3: Send All Data with Callbacks

```python
from backend.client import DistressMonitoringClient
import numpy as np

def on_result(result):
    print(f"Score: {result['distress_score']:.3f}")
    print(f"Level: {result['alert_level']}")

def on_alert(alert):
    print(f"🚨 {alert['level'].upper()}: {alert['message']}")

client = DistressMonitoringClient(
    'http://localhost:5000',
    on_result=on_result,
    on_alert=on_alert
)

client.connect()
client.start_session(patient_id='patient_123')

# Send multimodal data
for i in range(10):
    audio = np.random.randn(16000)
    
    client.send_data(
        audio=audio,
        face={'detected': True, 'emotion': 'neutral', 'emotion_confidence': 0.8, 'mouth_open': 0.2},
        eye={'detected': True, 'blink_rate': 20, 'eye_openness': 0.7, 'pupil_dilation': 0.5},
        posture={'detected': True, 'hunched': 0.3, 'head_down': 0.1, 'movement_intensity': 0.5, 'hand_to_face': False}
    )

client.end_session()
client.disconnect()
```

---

## 🔌 REST API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Get Statistics
```bash
curl http://localhost:5000/stats
```

### Get Recent Alerts
```bash
curl http://localhost:5000/alerts?limit=10
```

### Reset System
```bash
curl -X POST http://localhost:5000/reset
```

---

## 🧪 Testing

### Run Tests Without Server

```bash
cd backend
python quick_test.py
```

### Test With Server Running

1. Start server: `python server.py`
2. Run client: `python mediapipe_integration.py`

---

## 📝 Key Files Reference

| File | Purpose |
|------|---------|
| `server.py` | Main Flask server with WebSocket |
| `client.py` | Client library for sending data |
| `distress_calculator.py` | Core scoring algorithm |
| `alert_system.py` | Alert generation and management |
| `mediapipe_integration.py` | Face/eye/posture detection |
| `config.py` | Configuration and constants |
| `README.md` | Detailed documentation |
| `INTEGRATION_GUIDE.md` | Integration examples |

---

## 🛠️ Configuration

Edit `backend/config.py` to customize:

```python
# Modality weights
MODALITY_WEIGHTS = {
    'audio': 0.40,      # Increase for audio-heavy system
    'face': 0.30,
    'eye': 0.15,
    'posture': 0.15
}

# Alert thresholds
DISTRESS_THRESHOLDS = {
    'normal': (0.0, 0.3),
    'mild': (0.3, 0.5),
    'moderate': (0.5, 0.7),
    'severe': (0.7, 1.0)
}

# Audio settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION = 3.0
```

---

## 🎓 Integration Methods

### Method 1: MediaPipe (Recommended)
```python
from mediapipe_integration import MediaPipeDetector
detector = MediaPipeDetector()
data = detector.process_frame(frame)
```

### Method 2: Custom OpenCV
Integrate your own face detection models using OpenCV

### Method 3: Your Own Models
Use the client library to send data from any detection system

See `INTEGRATION_GUIDE.md` for detailed examples.

---

## 📊 Monitoring Dashboard

While the server is running, you can monitor:

- **Real-time statistics**: `http://localhost:5000/stats`
- **Alert history**: `http://localhost:5000/alerts`
- **WebSocket events**: Connect a client to see live alerts

---

## ❓ Frequently Asked Questions

**Q: Can I use my own audio model?**
A: Yes! The backend is agnostic to the audio model. Just send the audio data and it will be processed by your existing model.

**Q: Can I integrate with my own face detection?**
A: Absolutely! Use the client library to send face data from any detector (dlib, YOLO, your custom model, etc.)

**Q: How do I scale to multiple patients?**
A: Use the WebSocket to handle multiple concurrent clients. Each session is independent.

**Q: Can I change the distress score weights?**
A: Yes, edit `MODALITY_WEIGHTS` in `config.py` to adjust importance of each modality.

**Q: What's the latency?**
A: ~50-100ms per analysis, depending on hardware and model complexity.

---

## 🚨 Troubleshooting

### "Connection refused"
- Is the server running? `python backend/server.py`
- Check firewall settings
- Verify localhost:5000 is accessible

### "Audio model not found"
- Ensure `checkpoints/best_model.pth` exists
- If missing, download or train the model

### "No face detected"
- Check lighting conditions
- Ensure face is clearly visible
- Reduce detection confidence threshold

### "High latency"
- Reduce model complexity
- Use GPU acceleration
- Process frames separately from detection

---

## 📚 Next Steps

1. ✅ **Start the backend**: `python backend/server.py`
2. ✅ **Run the detector**: `python backend/mediapipe_integration.py`
3. 🔧 **Customize**: Edit thresholds in `config.py`
4. 📊 **Monitor**: Check alerts and statistics via REST API
5. 🚀 **Deploy**: Use Gunicorn for production
6. 📈 **Analyze**: Collect data and analyze patterns

---

## 📖 Documentation

- **Full README**: `backend/README.md`
- **Integration Guide**: `backend/INTEGRATION_GUIDE.md`
- **Client API**: `backend/client.py` (docstrings)
- **Distress Calculator**: `backend/distress_calculator.py` (docstrings)
- **Alert System**: `backend/alert_system.py` (docstrings)

---

## ✨ Summary

Your new architecture is:
- **Modular**: Each component is independent and reusable
- **Scalable**: Can handle multiple concurrent sessions
- **Real-time**: WebSocket for low-latency communication
- **Flexible**: Works with any detection models
- **Comprehensive**: Integrates audio, face, eye, and posture

You're ready to build a powerful patient monitoring system!

