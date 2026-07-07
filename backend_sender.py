import requests
import threading
import cv2
import time

ENABLE_BACKEND = True

BACKEND_URL = "http://localhost:8000/api/face-eye/features"
STREAM_URL  = "http://localhost:8000/api/stream/face"

session = requests.Session()
_last_error_log = 0.0

def set_backend_url(ip: str):
    global BACKEND_URL, STREAM_URL
    BACKEND_URL = f"http://{ip}:8000/api/face-eye/features"
    STREAM_URL  = f"http://{ip}:8000/api/stream/face"
    print(f"[Backend] Target set to http://{ip}:8000")

def send_to_backend(payload):
    global _last_error_log
    if not ENABLE_BACKEND:
        return
    try:
        response = session.post(BACKEND_URL, json=payload, timeout=3.0)
        if response.ok:
            _last_error_log = 0.0
    except Exception as e:
        now = time.time()
        if now - _last_error_log > 30.0:
            print(f"[Backend] Cannot reach {BACKEND_URL} — {e}")
            _last_error_log = now

def send_frame_to_backend(frame):
    def worker():
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            requests.post(
                STREAM_URL,
                files={"frame": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=0.2,
            )
        except Exception:
            pass
    threading.Thread(target=worker, daemon=True).start()
