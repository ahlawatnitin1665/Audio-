"""FastAPI backend for ICU patient monitoring.

Receives posture, face, and audio data from separate capture modules,
processes audio through the trained model, and exposes a fusion endpoint.
"""

import os
import sys
import time
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional

# Add src directory to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = FastAPI(title="ICU Patient Monitor Backend")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
face_data: dict = {}
posture_data: dict = {}
audio_data: dict = {}

# ---------------------------------------------------------------------------
# Audio model loading
# ---------------------------------------------------------------------------
audio_model = None

def _load_audio_model():
    global audio_model
    try:
        from inference import RealTimeMonitor
        audio_model = RealTimeMonitor()
        print("[Audio] Model loaded successfully")
    except Exception as e:
        print(f"[Audio] Failed to load model: {e}")
        audio_model = None

_load_audio_model()

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class AudioPayload(BaseModel):
    chunk: List[float]
    sample_rate: int = 16000

# ---------------------------------------------------------------------------
# Face endpoints
# ---------------------------------------------------------------------------
@app.post("/face")
def receive_face(data: dict):
    global face_data
    face_data = data
    return {"received": True}

@app.get("/face")
def get_face():
    return face_data

# ---------------------------------------------------------------------------
# Posture endpoints
# ---------------------------------------------------------------------------
@app.post("/posture")
def receive_posture(data: dict):
    global posture_data
    posture_data = data
    return {"received": True}

@app.get("/posture")
def get_posture():
    return posture_data

# ---------------------------------------------------------------------------
# Audio endpoints
# ---------------------------------------------------------------------------
@app.post("/audio")
def receive_audio(payload: AudioPayload):
    """Receive an audio chunk, run it through the model, store results."""
    global audio_data

    if audio_model is None:
        audio_data = {"error": "Audio model not loaded"}
        return audio_data

    try:
        audio_np = np.array(payload.chunk, dtype=np.float32)
        result = audio_model.monitor_chunk(audio_np)
        audio_data = {
            "timestamp": result["timestamp"],
            "distress_score": result["distress_score"],
            "alert_level": result["alert_level"],
            "priority": result["priority"],
            "dominant_class": result["dominant_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
        }
        return {"received": True, "audio_analysis": audio_data}
    except Exception as e:
        audio_data = {"error": str(e)}
        return {"received": False, "error": str(e)}

@app.get("/audio")
def get_audio():
    return audio_data

# ---------------------------------------------------------------------------
# Fusion endpoint
# ---------------------------------------------------------------------------
@app.get("/fusion")
def fusion():
    return {
        **face_data,
        **posture_data,
        **audio_data,
    }

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "audio_model_loaded": audio_model is not None,
        "has_face_data": bool(face_data),
        "has_posture_data": bool(posture_data),
        "has_audio_data": bool(audio_data),
    }
