"""
Configuration for the distress monitoring backend
"""
import os
from typing import Dict

# Server configuration
SERVER_HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.environ.get('SERVER_PORT', 5000))
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Audio configuration
AUDIO_SAMPLE_RATE = 16000  # 16 kHz
AUDIO_CHUNK_DURATION = 3.0  # seconds
AUDIO_OVERLAP = 0.5  # 50% overlap

# Distress score thresholds
DISTRESS_THRESHOLDS: Dict[str, tuple] = {
    'normal': (0.0, 0.3),
    'mild': (0.3, 0.5),
    'moderate': (0.5, 0.7),
    'severe': (0.7, 1.0)
}

# Modality weights (must sum to 1.0)
MODALITY_WEIGHTS: Dict[str, float] = {
    'audio': 0.40,
    'face': 0.30,
    'eye': 0.15,
    'posture': 0.15
}

# Alert configuration
ALERT_SUPPRESSION_DURATION = 5  # seconds
MAX_ALERT_HISTORY = 1000

# Model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_CLASSIFIER_PATH = os.path.join(BASE_DIR, 'checkpoints', 'best_model.pth')
AUDIO_DENOISER_PATH = os.path.join(BASE_DIR, 'checkpoints', 'simple_denoiser.pth')

# Logging configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FILE = os.environ.get('LOG_FILE', 'logs/backend.log')

# WebSocket configuration
WEBSOCKET_PING_INTERVAL = 25  # seconds
WEBSOCKET_PING_TIMEOUT = 60  # seconds
WEBSOCKET_BUFFER_SIZE = 1024 * 1024  # 1 MB

def get_config() -> Dict:
    """Get full configuration as dictionary"""
    return {
        'server': {
            'host': SERVER_HOST,
            'port': SERVER_PORT,
            'debug': DEBUG,
            'secret_key': SECRET_KEY
        },
        'audio': {
            'sample_rate': AUDIO_SAMPLE_RATE,
            'chunk_duration': AUDIO_CHUNK_DURATION,
            'overlap': AUDIO_OVERLAP
        },
        'distress': {
            'thresholds': DISTRESS_THRESHOLDS,
            'modality_weights': MODALITY_WEIGHTS
        },
        'alerts': {
            'suppression_duration': ALERT_SUPPRESSION_DURATION,
            'max_history': MAX_ALERT_HISTORY
        },
        'models': {
            'audio_classifier': AUDIO_CLASSIFIER_PATH,
            'audio_denoiser': AUDIO_DENOISER_PATH
        },
        'logging': {
            'level': LOG_LEVEL,
            'file': LOG_FILE
        }
    }
