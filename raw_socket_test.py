"""Raw socket test - bypasses client library"""
import sys, time, socketio

sio = socketio.Client()
results = []
errors = []

@sio.on('analysis_result')
def on_result(data):
    results.append(data)
    print(f'GOT RESULT: score={data.get("distress_score", "?")}')

@sio.on('error')
def on_error(data):
    errors.append(data)
    print(f'SERVER ERROR: {data}')

@sio.on('connect')
def on_connect():
    print('Connected!')

print('Connecting...')
sio.connect('http://localhost:5000')

print('Starting session...')
sio.emit('start_session', {'patient_id': 'raw_test'})
time.sleep(1)

# Send ONLY face data (no audio)
data = {
    'face': {'detected': True, 'emotion': 'sad', 'emotion_confidence': 0.8, 'mouth_open': 0.3}
}
print('Sending face-only data...')
sio.emit('data', data)

print('Waiting...')
time.sleep(3)
print(f'Results: {len(results)}')
print(f'Errors: {len(errors)}')
sio.emit('end_session')
sio.disconnect()
