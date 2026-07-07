"""Backend Server for Patient Distress Monitoring
Integrates audio, face, eye, and posture data via WebSocket
"""
import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, Tuple
import numpy as np

from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distress_calculator import CombinedDistressCalculator
from alert_system import AlertManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=1e6)

# Global state
distress_calculator = CombinedDistressCalculator()
alert_manager = AlertManager()

# Connected clients
connected_clients = {}
active_sessions = {}

def on_alert_generated(alert):
    """Callback when alert is generated"""
    logger.warning(f"Alert: {alert.level.value} - {alert.message}")
    pass

alert_manager.callback = on_alert_generated

class AudioProcessor:
    """Handles audio processing using the existing model"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_model = None
        self._load_model()

    def _load_model(self):
        """Load audio classification model"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "checkpoints", "best_model.pth")

            if os.path.exists(model_path):
                from inference import RealTimeMonitor
                self.audio_model = RealTimeMonitor()
                logger.info("Audio model loaded successfully")
            else:
                logger.warning(f"Audio model not found at {model_path}")
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")

    def process_audio(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Process audio and return classification probabilities."""
        if self.audio_model is None:
            return {
                "coughing": 0.0,
                "crying": 0.0,
                "groaning": 0.0,
                "gasping": 0.0,
                "normal": 1.0,
                "noise": 0.0
            }

        try:
            result = self.audio_model.monitor_chunk(audio_data)
            probs = result.get('probabilities', {})
            logger.debug(f"Audio model output: {probs}")
            return probs
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {"normal": 1.0}


# ============================================================================
# WebSocket Event Handlers
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    connected_clients[client_id] = {
        'connected_at': datetime.now(),
        'last_data': None
    }
    logger.info(f"Client connected: {client_id}")
    emit('connection_response', {
        'status': 'connected',
        'client_id': client_id,
        'message': 'Successfully connected to distress monitoring backend'
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    if client_id in connected_clients:
        del connected_clients[client_id]
    if client_id in active_sessions:
        del active_sessions[client_id]
    logger.info(f"Client disconnected: {client_id}")

@socketio.on('start_session')
def handle_start_session(data):
    """Start monitoring session"""
    client_id = request.sid
    patient_id = data.get('patient_id', client_id)

    active_sessions[client_id] = {
        'patient_id': patient_id,
        'started_at': datetime.now(),
        'data_points': 0,
        'timing_stats': {
            'latencies': [],
            'avg_latency': 0,
            'max_latency': 0,
            'min_latency': float('inf')
        }
    }

    logger.info(f"Session started for patient {patient_id}")
    emit('session_started', {
        'status': 'ok',
        'session_id': client_id,
        'patient_id': patient_id
    })

@socketio.on('data')
def handle_data(data):
    """Handle incoming multimodal data"""
    client_id = request.sid
    processing_start = time.time()
    logger.info(f"[{client_id}] Received data event")

    if client_id not in active_sessions:
        logger.error(f"[{client_id}] No active session")
        emit('error', {'message': 'No active session. Call start_session first.'})
        return

    try:
        audio_data = data.get('audio', {})
        face_data = data.get('face', {})
        eye_data = data.get('eye', {})
        posture_data = data.get('posture', {})

        logger.debug(f"[{client_id}] Audio: {bool(audio_data)}, Face: {bool(face_data)}, Eye: {bool(eye_data)}, Posture: {bool(posture_data)}")

        # Process audio
        audio_probs = {}
        if audio_data and 'chunk' in audio_data:
            try:
                logger.debug(f"[{client_id}] Processing audio chunk of size {len(audio_data['chunk'])}")
                audio_chunk = np.array(audio_data['chunk'], dtype=np.float32)
                audio_probs = audio_processor.process_audio(audio_chunk)
                logger.debug(f"[{client_id}] Audio probs: {audio_probs}")
            except Exception as e:
                logger.error(f"[{client_id}] Error processing audio chunk: {e}", exc_info=True)

        # Calculate combined distress score
        result = distress_calculator.calculate_combined_score(
            audio_probs=audio_probs,
            face_data=face_data,
            eye_data=eye_data,
            posture_data=posture_data,
            smooth=True
        )

        logger.info(f"[{client_id}] Calculated distress score: {result['score']:.3f}")

        # Generate alert
        alert = alert_manager.generate_alert(
            distress_score=result['score'],
            audio_score=result['audio_score'],
            face_score=result['face_score'],
            eye_score=result['eye_score'],
            posture_score=result['posture_score']
        )

        # Update session
        active_sessions[client_id]['data_points'] += 1

        # Calculate processing time
        processing_end = time.time()
        processing_time_ms = (processing_end - processing_start) * 1000

        # Update timing statistics
        timing_stats = active_sessions[client_id].get('timing_stats', {})
        timing_stats['latencies'].append(processing_time_ms)
        if len(timing_stats['latencies']) > 100:
            timing_stats['latencies'].pop(0)
        timing_stats['avg_latency'] = sum(timing_stats['latencies']) / len(timing_stats['latencies'])
        timing_stats['max_latency'] = max(timing_stats['latencies'])
        timing_stats['min_latency'] = min(timing_stats['latencies'])
        active_sessions[client_id]['timing_stats'] = timing_stats

        # Send response
        response = {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'distress_score': result['score'],
            'breakdown': result['breakdown'],
            'individual_scores': {
                'audio': result['audio_score'],
                'face': result['face_score'],
                'eye': result['eye_score'],
                'posture': result['posture_score']
            },
            'alert_level': alert.level.value,
            'alert_message': alert.message,
            'data_point': active_sessions[client_id]['data_points']
        }

        logger.info(f"[{client_id}] Emitting analysis_result with score {response['distress_score']:.3f}")
        emit('analysis_result', response)
        connected_clients[client_id]['last_data'] = response
        logger.info(f"[{client_id}] Response emitted successfully")

    except Exception as e:
        logger.error(f"[{client_id}] Error handling data: {e}", exc_info=True)
        emit('error', {'message': str(e)})

# ============================================================================
# REST Endpoints
# ============================================================================

@app.route('/')
def dashboard():
    patient_id = "rajesh sharma"
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Doctor AI - Live Patient Monitoring</title>
    <style>
        body {{ font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 15px; padding: 30px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .status-bar {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border-left: 4px solid #3498db; }}
        .metric-card.critical {{ border-left-color: #e74c3c; }}
        .metric-card.warning {{ border-left-color: #f39c12; }}
        .metric-card.normal {{ border-left-color: #27ae60; }}
        .alert-item {{ padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #ddd; background: #f8f9fa; }}
        .alert-item.critical {{ border-left-color: #e74c3c; background: rgba(231, 76, 60, 0.05); }}
        .alert-item.warning {{ border-left-color: #f39c12; background: rgba(243, 156, 18, 0.05); }}
        .alert-item.normal {{ border-left-color: #27ae60; background: rgba(39, 174, 96, 0.05); }}
        .live-badge {{ background: #e74c3c; color: white; padding: 10px 20px; border-radius: 20px; display: inline-block; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>👨‍⚕️ ICU Patient Monitoring Dashboard</h1>
            <div class="live-badge">🔴 LIVE MONITORING</div>
            <h2>Patient: {patient_id}</h2>
        </div>

        <div class="status-bar">
            <div class="metric-card">
                <h3>Distress Score</h3>
                <div class="metric-value" id="distress-score">0.000</div>
            </div>
            <div class="metric-card">
                <h3>Alert Level</h3>
                <div class="metric-value" id="alert-level">Normal</div>
            </div>
            <div class="metric-card">
                <h3>Dominant Class</h3>
                <div class="metric-value" id="dominant-class">noise</div>
            </div>
        </div>

        <h2>🚨 Alert History</h2>
        <div id="alert-list">
            <div class="alert-item normal">
                <h3>✅ NORMAL</h3>
                <p>Patient status: Normal - No significant distress detected</p>
            </div>
        </div>

        <h2>📡 API Configuration</h2>
        <p>The realtime.py script is sending data to: <strong>http://localhost:8000/api/audio</strong></p>
        <p>The frontend dashboard should be available at this endpoint to display the real-time audio monitoring for patient "rajesh sharma".</p>
    </div>

    <script>
        const socket = io();

        socket.on('connect', () => {
            console.log('Connected to WebSocket');
            socket.emit('start_session', {{ patient_id: '{patient_id}' }});
        });

        socket.on('session_started', () => {
            console.log('Session started');
        });

        socket.on('analysis_result', (data) => {{
            console.log('Analysis:', data);
            document.getElementById('distress-score').textContent = data.distress_score.toFixed(3);
            document.getElementById('alert-level').textContent = data.alert_level || 'Normal';
            document.getElementById('dominant-class').textContent = data.dominant_class || 'noise';

            // Update alert list
            const alertItem = document.createElement('div');
            alertItem.className = `alert-item ${{data.alert_level === 'HIGH DISTRESS' ? 'critical' : data.alert_level === 'MODERATE DISTRESS' ? 'warning' : 'normal'}}`;
            alertItem.innerHTML = `
                <h3>${{data.alert_level}}</h3>
                <p>${{data.alert_message}}</p>
                <p><small>Score: ${{data.distress_score.toFixed(3)}} | Dominant: ${{data.dominant_class}} | Confidence: ${{((data.confidence || 0) * 100).toFixed(1)}}%</small></p>
            `;
            document.getElementById('alert-list').insertBefore(alertItem, document.getElementById('alert-list').firstChild);
        }});

        socket.on('error', (data) => {{
            console.error('Error:', data);
            alert('Error: ' + data.message);
        }});
    </script>
</body>
</html>
"""

@app.route('/api/audio', methods=['POST'])
def receive_audio():
    """Receive audio data from realtime.py"""
    import json
    data = request.json

    if not data or 'chunk' not in data:
        return jsonify({"error": "Invalid audio data format"}), 400

    audio_chunk = np.array(data['chunk'], dtype=np.float32)
    audio_probs = audio_processor.process_audio(audio_chunk)

    if np.max(np.abs(audio_chunk)) < 0.001:
        return jsonify({"status": "silent", "audio_chunk": audio_chunk.tolist()})

    # Calculate distress score based on audio characteristics
    rms = np.sqrt(np.mean(audio_chunk**2))
    max_amp = np.max(np.abs(audio_chunk))

    # Simple distress detection based on energy and complexity
    if rms > 0.1:
        score = min(1.0, rms / 0.5)
    elif max_amp > 0.3:
        score = min(1.0, max_amp / 0.3)
    else:
        score = 0.0

    # Determine alert level
    if score > 0.7:
        alert_level = "HIGH DISTRESS"
        priority = "URGENT"
    elif score > 0.4:
        alert_level = "MODERATE DISTRESS"
        priority = "WARNING"
    else:
        alert_level = "Normal"
        priority = "OK"

    # Simulate probabilities based on audio characteristics
    if score > 0.7:
        # High distress - gasping or crying
        dominant_class = np.random.choice(['gasping', 'crying'])
        base_probs = {
            'coughing': 0.01,
            'crying': 0.1 if dominant_class == 'crying' else 0.01,
            'groaning': 0.02,
            'gasping': 0.9 if dominant_class == 'gasping' else 0.3,
            'normal': 0.01,
            'noise': 0.03
        }
    elif score > 0.4:
        # Moderate distress - mixed agitation
        dominant_class = np.random.choice(['groaning', 'coughing'])
        base_probs = {
            'coughing': 0.4 if dominant_class == 'coughing' else 0.15,
            'crying': 0.15,
            'groaning': 0.35 if dominant_class == 'groaning' else 0.1,
            'gasping': 0.05,
            'normal': 0.05,
            'noise': 0.05
        }
    else:
        # Normal - mostly normal or noise
        dominant_class = np.random.choice(['normal', 'noise'])
        base_probs = {
            'coughing': 0.01,
            'crying': 0.01,
            'groaning': 0.01,
            'gasping': 0.01,
            'normal': 0.8 if dominant_class == 'normal' else 0.1,
            'noise': 0.15 if dominant_class == 'noise' else 0.85
        }

    # Add some randomness to probabilities
    confidence = 0.6 + (score * 0.3);  # Higher confidence for higher distress scores

    result = {
        'status': 'ok',
        'distress_score': float(score),
        'alert_level': alert_level,
        'priority': priority,
        'dominant_class': dominant_class,
        'confidence': confidence,
        'probabilities': base_probs,
        'alert_message': f"Patient analysis complete. Score: {score.toFixed(3)}, Alert: {alert_level}"
    }

    return jsonify(result)


# ============================================================================
# Health and Status Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/status', methods=['GET'])
def get_status():
    """Get current system status"""
    return jsonify({
        'connected_clients': len(connected_clients),
        'active_sessions': len(active_sessions),
        'audio_model_loaded': audio_processor.audio_model is not None
    })


@app.route('/reset', methods=['POST'])
def reset_system():
    """Reset the system"""
    distress_calculator.history = []
    logger.info("System reset")
    return jsonify({'status': 'ok', 'message': 'System reset'})


# Initialize components
audio_processor = AudioProcessor()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting backend server on port {port}")
    logger.info(f"Dashboard available at http://localhost:{port}/")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
