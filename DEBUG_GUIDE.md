# Backend Server - Debugging Guide

## Status

Your backend server is **running** but **not sending results** to the client. This guide will help us debug it.

## What You Need To Do

### Step 1: Stop the Backend Server
Go to the terminal running `python backend/server.py` and press:
```
Ctrl+C
```

You should see:
```
^C
Keyboard interrupt received, shutting down...
```

### Step 2: Restart Backend Server (with new code)
Run:
```bash
cd C:\Users\Lenovo\OneDrive\Desktop\elc\patient_audio_monitor
python backend/server.py
```

You should see:
```
Starting server on port 5000
 * Running on http://0.0.0.0:5000
```

**KEEP THIS TERMINAL OPEN**

### Step 3: Run Test (New Terminal)
Open a **new terminal** and run:
```bash
cd C:\Users\Lenovo\OneDrive\Desktop\elc\patient_audio_monitor
python debug_test.py
```

### Step 4: Check Server Logs
Look at the **server terminal** and you should see messages like:

```
[client_id] Received data event
[client_id] Audio: True, Face: True, Eye: True, Posture: True
[client_id] Calculated distress score: 0.420
[client_id] Emitting analysis_result with score 0.420
[client_id] Response emitted successfully
```

### Step 5: Report What You See

Send me:
1. **What appears in the client terminal** (debug_test.py output)
2. **What appears in the server terminal** (python backend/server.py output)
3. **Any error messages**

---

## Expected Behavior

### If Everything Works ✅

**Client output:**
```
OK: Connected
OK: Session started
Send result: True
Results received: 1
Alerts generated: 0
RESULT: Score=0.420, Alert=mild
```

**Server output:**
```
Received data event
Calculated distress score: 0.420
Response emitted successfully
```

### If Something's Wrong ❌

**Common issues:**
- `Results received: 0` = Server not sending response
- Error messages = Look at them carefully
- Nothing in server logs = Data not reaching server

---

## Test the Core Modules First

Before testing the server, verify the core modules work:

```bash
python test_core_modules.py
```

Expected output:
```
TESTING CORE MODULES (No server needed)

1. Testing Distress Calculator...
   Distress Score: 0.420
   ...
   OK
   
2. Testing Alert Manager...
   Alert Level: moderate
   ...
   OK
```

---

## Quick Checklist

- [ ] Stopped old backend server (Ctrl+C)
- [ ] Restarted backend server with `python backend/server.py`
- [ ] Server is running and listening on port 5000
- [ ] Opened NEW terminal for tests
- [ ] Ran `python test_core_modules.py` - everything OK
- [ ] Ran `python debug_test.py` - check results

---

## Commands Reference

```bash
# Test core modules (no server needed)
python test_core_modules.py

# Test with server (needs server running)
python debug_test.py

# Check if port 5000 is in use
netstat -ano | findstr :5000

# Kill process on port 5000 (if needed)
taskkill /PID <PID> /F
```

---

## Next Steps

Once you follow these steps, share the output and we can debug further!

