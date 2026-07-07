# 🚀 Backend Server - Now Running!

Your backend server is **live and ready** on `http://localhost:5000`

## What's Running?

✅ **Audio Processing**: Using your trained audio models
✅ **Distress Calculator**: Combining audio + face + eye + posture
✅ **Alert System**: Real-time alert generation
✅ **WebSocket Server**: Ready for client connections

---

## 🧪 Test It Now

### Terminal 1: Backend Server (Already Running ✅)
```
python backend/server.py
```

### Terminal 2: Test Client
```
python test_backend_client.py
```

This sends sample data and verifies everything works.

---

## 📡 Integration Options

### Option 1: Use Your Webcam (Recommended)
```bash
python backend/mediapipe_integration.py
```
- Opens your webcam
- Detects face, eyes, and posture
- Sends everything to backend in real-time

### Option 2: Send Audio Only
```python
from backend.client import DistressMonitoringClient
import numpy as np

client = DistressMonitoringClient('http://localhost:5000')
client.connect()
client.start_session()

# Capture audio from microphone
audio = get_audio_chunk()  # Your audio capture code

# Send to backend (your audio model will process it)
client.send_audio(audio)

client.end_session()
client.disconnect()
```

### Option 3: Send Vision Only
```python
client.send_vision(
    face={'detected': True, 'emotion': 'sad', 'emotion_confidence': 0.8, 'mouth_open': 0.3},
    eye={'detected': True, 'blink_rate': 22, 'eye_openness': 0.7, 'pupil_dilation': 0.6},
    posture={'detected': True, 'hunched': 0.4, 'head_down': 0.2, 'movement_intensity': 0.6, 'hand_to_face': False}
)
```

### Option 4: Send Everything Together
```python
client.send_data(
    audio=audio_chunk,
    face=face_data,
    eye=eye_data,
    posture=posture_data
)
```

---

## 📊 Monitor Your System

### Check Health
```bash
curl http://localhost:5000/health
```

### Get Statistics
```bash
curl http://localhost:5000/stats
```

### View Recent Alerts
```bash
curl http://localhost:5000/alerts?limit=10
```

---

## 📁 Key Files

| File | What It Does |
|------|-------------|
| `backend/server.py` | Main backend (currently running) |
| `backend/client.py` | Send data to backend |
| `backend/distress_calculator.py` | Calculates distress scores |
| `backend/alert_system.py` | Generates alerts |
| `backend/mediapipe_integration.py` | Face/eye/posture detection |
| `test_backend_client.py` | Test client (run this next!) |

---

## 🎯 Your Next Steps

1. **Run the test client** (in new terminal):
   ```bash
   python test_backend_client.py
   ```

2. **Verify results** - You should see:
   - ✅ Connected to backend
   - ✅ Session started
   - ✅ Data processing with distress scores
   - ✅ Alerts generated

3. **Choose an integration**:
   - Option A: Use `python backend/mediapipe_integration.py` with webcam
   - Option B: Integrate with your own detection system using `backend/client.py`

4. **Customize** (optional):
   - Edit `backend/config.py` to adjust distress thresholds
   - Adjust modality weights (audio, face, eye, posture importance)

---

## 💡 Quick Reference

### Start Backend
```bash
python backend/server.py
```

### Test With Sample Data
```bash
python test_backend_client.py
```

### Use With Webcam
```bash
python backend/mediapipe_integration.py
```

### View Documentation
```bash
type QUICK_START.md
type backend/README.md
type backend/INTEGRATION_GUIDE.md
```

---

## ✨ What Happens When Data Is Sent

```
Your System                Backend Server              Output
   │                            │                         │
   ├─ Audio Data ────────────►  Distress                 │
   ├─ Face Data ───────────►   Calculator  ──────────►  Score: 0.42
   ├─ Eye Data ────────────►   (Combined)  ──────────►  Level: Mild
   └─ Posture Data ───────►    ──────────────────────►  Alert: Yes
                                │
                                Alert Manager
                                │
                                ├─► Generate Alert
                                ├─► Send to Client
                                └─► Log History
```

---

## 🚨 Common Issues

**Q: "Connection refused"**
- Backend server might not be running
- Run `python backend/server.py` in a terminal first

**Q: "No results received"**
- Check server logs for errors
- Ensure network connectivity

**Q: "How do I stop the server?"**
- Press `Ctrl+C` in the server terminal

---

## 📚 Learn More

- **QUICK_START.md** - Complete getting started guide
- **backend/README.md** - Full technical documentation
- **backend/INTEGRATION_GUIDE.md** - Integration code examples

---

**You're all set! 🎉 Run `python test_backend_client.py` to verify everything works.**
