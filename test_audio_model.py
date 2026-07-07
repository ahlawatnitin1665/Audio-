"""Test audio model loading and processing"""
import sys
import numpy as np
sys.path.insert(0, 'src')
sys.path.insert(0, 'backend')

print("Testing audio model...")

try:
    from inference import RealTimeMonitor
    print("1. RealTimeMonitor imported OK")
    
    monitor = RealTimeMonitor()
    print("2. RealTimeMonitor initialized OK")
    
    test_audio = np.random.randn(16000).astype(np.float32)
    result = monitor.monitor_chunk(test_audio)
    print("3. Audio processed OK")
    
    probs = result.get('probabilities', {})
    print("4. Probabilities:")
    for k, v in probs.items():
        print(f"   {k}: {v:.4f}")
    
    print("\nAUDIO MODEL IS WORKING!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
