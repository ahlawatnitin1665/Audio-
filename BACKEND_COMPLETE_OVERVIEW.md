# Backend System - Complete Overview

## 🏗️ What We Built

A **modular backend server** that:
1. Receives multimodal data (audio, face, eye, posture) from clients
2. Processes and combines all data types
3. Calculates a **combined distress score**
4. Generates **intelligent alerts**
5. Sends results back to clients in real-time

---

## 📊 Data Flow Diagram

```
CLIENT SIDE (Your Application)
═════════════════════════════════════════════

[Camera & Microphone Inputs]
         ↓
    [Face Detection]  [Audio Model]  [Eye Tracking]  [Posture Detection]
         ↓              ↓               ↓               ↓
    [Data Aggregation]
         ↓
    [DistressMonitoringClient]
         ↓
    [Send via WebSocket]
         ↓
    ════════════════════════════════════════════════════════════════
                      INTERNET / LOCAL NETWORK
    ════════════════════════════════════════════════════════════════
         ↓
SERVER SIDE (Flask Backend)
═════════════════════════════════════════════

    [WebSocket Receiver]
         ↓
    [Audio Processor]  [Face Data]  [Eye Data]  [Posture Data]
         ↓              ↓            ↓           ↓
    [Combined Distress Calculator]
         ↓
    [Score: 0.42]
    [Breakdown by modality]
         ↓
    [Alert Manager]
         ↓
    [Alert Level: MILD]
    [Alert Message: "Mild distress detected..."]
         ↓
    [Emit via WebSocket]
         ↓
    ════════════════════════════════════════════════════════════════
                      INTERNET / LOCAL NETWORK
    ════════════════════════════════════════════════════════════════
         ↓
CLIENT SIDE
═════════════════════════════════════════════

    [Receive Result]
         ↓
    [Display Alert]
    [Update Dashboard]
    [Trigger Actions]
```

---

## 📤 SENDING TO BACKEND (Client → Server)

### 1. **Initial Connection**
```json
{
  "event": "connect",
  "data": {}
}
```

### 2. **Start Session**
```json
{
  "event": "start_session",
  "data": {
    "patient_id": "patient_001"
  }
}
```

### 3. **Send Multimodal Data** (Main Data Flow)
```json
{
  "event": "data",
  "data": {
    "audio": {
      "chunk": [0.001, -0.002, 0.003, ...],  // Audio samples (float array)
      "sample_rate": 16000                    // Hz
    },
    "face": {
      "detected": true,
      "emotion": "sad",
      "emotion_confidence": 0.8,
      "mouth_open": 0.3                       // 0-1 scale (0=closed, 1=wide open)
    },
    "eye": {
      "detected": true,
      "blink_rate": 22,                       // Blinks per minute
      "eye_openness": 0.7,                    // 0-1 scale
      "pupil_dilation": 0.6                   // 0-1 scale (0=constricted, 1=dilated)
    },
    "posture": {
      "detected": true,
      "hunched": 0.4,                         // 0-1 scale (how hunched)
      "head_down": 0.2,                       // 0-1 scale (how much head is down)
      "movement_intensity": 0.6,              // 0-1 scale (agitation level)
      "hand_to_face": false                   // Boolean
    }
  }
}
```

### 4. **Alternative: Audio Only**
```json
{
  "event": "audio",
  "data": {
    "chunk": [0.001, -0.002, 0.003, ...],
    "sample_rate": 16000
  }
}
```

### 5. **Alternative: Vision Only**
```json
{
  "event": "vision",
  "data": {
    "face": { ... },
    "eye": { ... },
    "posture": { ... }
  }
}
```

### 6. **End Session**
```json
{
  "event": "end_session",
  "data": {}
}
```

---

## 📥 RECEIVING FROM BACKEND (Server → Client)

### 1. **Connection Confirmed**
```json
{
  "event": "connection_response",
  "data": {
    "status": "connected",
    "client_id": "XAe9-2VJZ7O8zT70AAAD",
    "message": "Successfully connected to distress monitoring backend"
  }
}
```

### 2. **Session Started**
```json
{
  "event": "session_started",
  "data": {
    "status": "ok",
    "session_id": "XAe9-2VJZ7O8zT70AAAD",
    "patient_id": "patient_001"
  }
}
```

### 3. **Analysis Result** (Main Response)
```json
{
  "event": "analysis_result",
  "data": {
    "status": "ok",
    "timestamp": "2026-06-23T15:04:09.954408",
    "distress_score": 0.42,
    
    "breakdown": {
      "audio": 0.168,        // 40% weight
      "face": 0.192,         // 30% weight
      "eye": 0.0,            // 15% weight
      "posture": 0.0         // 15% weight
    },
    
    "individual_scores": {
      "audio": 0.42,         // Raw audio score (before weighting)
      "face": 0.64,          // Raw face score
      "eye": 0.0,            // Raw eye score
      "posture": 0.0         // Raw posture score
    },
    
    "alert_level": "mild",
    "alert_message": "Mild distress detected (face). Score: 0.42. Status: consistent",
    "trend": "stable",
    "data_point": 1
  }
}
```

### 4. **Alert Event** (If Alert Occurs)
```json
{
  "event": "alert",
  "data": {
    "alert_id": "alert_1687529049954",
    "level": "severe",
    "message": "SEVERE distress alert! Primary: crying (0.85). Overall score: 0.75. IMMEDIATE ATTENTION REQUIRED",
    "distress_score": 0.75,
    "modality_scores": {
      "audio": 0.85,
      "face": 0.70,
      "eye": 0.60,
      "posture": 0.65
    },
    "timestamp": "2026-06-23T15:04:10.123456"
  }
}
```

### 5. **Session Ended Summary**
```json
{
  "event": "session_ended",
  "data": {
    "patient_id": "patient_001",
    "duration_seconds": 120.5,
    "data_points": 45,
    "alert_stats": {
      "total_alerts": 5,
      "normal": 30,
      "mild": 10,
      "moderate": 4,
      "severe": 1
    },
    "recent_alerts": [
      {
        "level": "severe",
        "timestamp": "2026-06-23T15:05:45.123456",
        "message": "..."
      }
    ]
  }
}
```

### 6. **Error Response**
```json
{
  "event": "error",
  "data": {
    "message": "No active session. Call start_session first."
  }
}
```

---

## 🧮 Distress Score Calculation

### Formula
```
Combined Distress Score = 
  (Audio Score × 0.40) +
  (Face Score × 0.30) +
  (Eye Score × 0.15) +
  (Posture Score × 0.15)
```

### Example Calculation
```
Audio Score = 0.0  (not provided)
Face Score = 0.64  (emotion = sad, confidence = 0.8, mouth_open = 0.3)
Eye Score = 0.0    (not provided)
Posture Score = 0.0 (not provided)

Result = (0.0 × 0.40) + (0.64 × 0.30) + (0.0 × 0.15) + (0.0 × 0.15)
       = 0.0 + 0.192 + 0.0 + 0.0
       = 0.192 ✅
```

---

## 🚨 Alert Levels

| Score Range | Level | Priority | Meaning |
|-------------|-------|----------|---------|
| 0.0 - 0.3 | **Normal** | Info | Patient is stable |
| 0.3 - 0.5 | **Mild** | Low | Minor distress indicators |
| 0.5 - 0.7 | **Moderate** | Medium | Moderate distress detected |
| 0.7 - 1.0 | **Severe** | High | **IMMEDIATE ATTENTION NEEDED** |

---

## 📊 Audio Score Importance

The audio model calculates class probabilities. These are weighted by importance:

```
Audio Score = 
  (Crying probability × 0.95) +
  (Gasping probability × 0.90) +
  (Coughing probability × 0.80) +
  (Groaning probability × 0.70) +
  (Normal probability × 0.00)

Example:
  Audio input: "patient is crying"
  Probabilities: {crying: 0.8, normal: 0.2, others: 0}
  Audio Score = (0.8 × 0.95) + (0.2 × 0.00) = 0.76
```

---

## 🔄 Complete Communication Example

### Step 1: Connect
```
CLIENT:  "I want to connect"
SERVER:  "OK, your client_id is: ABC123"
```

### Step 2: Start Session
```
CLIENT:  "Start session for patient_001"
SERVER:  "Session started, session_id: ABC123"
```

### Step 3: Send Data (Repeat many times)
```
CLIENT:  "Here's data: audio=[...], face={emotion: sad}, ..."
SERVER:  "Calculated distress_score: 0.42, alert_level: mild"
```

### Step 4: End Session
```
CLIENT:  "End session"
SERVER:  "Session ended. Total data points: 45, alerts: 5"
```

---

## 🎯 Key Features

### ✅ Real-time Processing
- Data received → Processed → Result sent within 50-100ms
- Uses WebSocket for low-latency communication

### ✅ Modular Design
- Each data type (audio, face, eye, posture) is independent
- Can use any/all modalities - missing data is handled gracefully

### ✅ Trend Analysis
- Tracks if distress is improving, stable, or worsening
- Compares recent scores to determine trend

### ✅ Alert Suppression
- Prevents alert spam
- Severe alerts only trigger once per 5 seconds

### ✅ Session Management
- Each client has independent session
- Can handle multiple concurrent clients
- Tracks data points and statistics per session

---

## 📂 Files Involved

| File | Purpose |
|------|---------|
| `backend/server.py` | Main WebSocket server (receives and sends data) |
| `backend/client.py` | Client library (what you use to send data) |
| `backend/distress_calculator.py` | Core scoring algorithm |
| `backend/alert_system.py` | Alert generation and management |
| `backend/config.py` | Configuration (weights, thresholds) |

---

## 💻 Example: What Happens Step-by-Step

```python
# 1. CLIENT SIDE - Create client and connect
from backend.client import DistressMonitoringClient

client = DistressMonitoringClient('http://localhost:5000')
client.connect()  
# ← SENDS: "connect" event
# → RECEIVES: connection_response with client_id

# 2. CLIENT SIDE - Start session
client.start_session(patient_id='patient_001')
# ← SENDS: start_session event
# → RECEIVES: session_started confirmation

# 3. CLIENT SIDE - Send data
audio = get_audio_from_microphone()
face_data = detect_face_from_camera()
client.send_data(audio=audio, face=face_data, ...)
# ← SENDS: data event with all modalities
# ↓
# SERVER SIDE - Process data
# 1. Receive multimodal data
# 2. Process audio through your model
# 3. Calculate face/eye/posture scores
# 4. Combine scores: 0.42
# 5. Generate alert: "mild"
# ↓
# → RECEIVES: analysis_result with score=0.42, level=mild

# 4. CLIENT SIDE - Handle result
def on_result(result):
    print(f"Score: {result['distress_score']:.3f}")
    print(f"Alert: {result['alert_level']}")
    # Update UI, trigger notifications, etc.

client = DistressMonitoringClient(
    'http://localhost:5000',
    on_result=on_result
)
```

---

## 🔗 Data Types Reference

### Audio Data
```python
{
    "chunk": [float, float, float, ...],  # 16000 samples = 1 second
    "sample_rate": 16000                  # Hz
}
```

### Face Data
```python
{
    "detected": bool,
    "emotion": str,                       # "happy", "sad", "angry", "fearful", etc.
    "emotion_confidence": float,          # 0.0 to 1.0
    "mouth_open": float                   # 0.0 to 1.0
}
```

### Eye Data
```python
{
    "detected": bool,
    "blink_rate": float,                  # Blinks per minute (normal: 15-20)
    "eye_openness": float,                # 0.0 to 1.0
    "pupil_dilation": float               # 0.0 to 1.0
}
```

### Posture Data
```python
{
    "detected": bool,
    "hunched": float,                     # 0.0 to 1.0 (how hunched)
    "head_down": float,                   # 0.0 to 1.0
    "movement_intensity": float,          # 0.0 to 1.0 (agitation)
    "hand_to_face": bool                  # Self-soothing indicator
}
```

---

## 🎓 Summary

**You now have:**

1. **Backend Server** that:
   - Listens on `http://localhost:5000`
   - Receives multimodal data via WebSocket
   - Processes all data types simultaneously
   - Calculates combined distress score
   - Generates intelligent alerts
   - Sends results back in real-time

2. **Client Library** that:
   - Connects to the backend
   - Sends audio, face, eye, and posture data
   - Receives analysis results
   - Handles alerts and callbacks

3. **Scoring System** that:
   - Weights each modality (audio 40%, face 30%, eye 15%, posture 15%)
   - Calculates individual scores
   - Combines them into overall distress score
   - Determines alert level

4. **Alert System** that:
   - Generates alerts based on thresholds
   - Tracks alert history
   - Prevents spam with suppression
   - Analyzes trends

