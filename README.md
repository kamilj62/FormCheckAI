# 📱 FormCheck AI — Real Backend Inference

This project includes:
- Mobile app (Expo React Native) for uploading videos
- FastAPI backend that performs real exercise classification using your trained model

---

# 🧠 What the Backend Does

1. Accepts uploaded video
2. Runs MediaPipe Pose on each frame
3. Extracts 71 features per frame
4. Adds velocity → 142 features
5. Runs your trained sequence model
6. Returns:
   - exercise label
   - confidence
   - coaching notes
   - debug details

---

# ⚙️ Backend Setup

cd backend
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

You should see:
Uvicorn running on http://0.0.0.0:8000

---

# ✅ Test Backend (IMPORTANT — do this first)

## 1. Health Check (Mac browser)

http://127.0.0.1:8000/health

Expected:

{
  "status": "ok",
  "model_loaded": true,
  "labels": ["Bench Press","Deadlift","Push Press","Squat"]
}

---

## 2. Test Inference with curl

time curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "accept: application/json" \
  -F 'file=@/Users/josephkamil/Desktop/Capstone/Front Squat- Depth.mov'

---

## ✅ Example Successful Output

{
  "exercise_label": "Squat",
  "confidence": 0.63,
  "feedback": [
    "Predicted exercise: Squat.",
    "Model confidence: 63.5%",
    "This is real model inference from your uploaded video."
  ],
  "details": {
    "frames_seen": 1311,
    "frames_processed": 1267,
    "sequences_scored": 248,
    "mean_pose_visibility": 0.763,
    "class_probabilities": {
      "Bench Press": 0.0856,
      "Deadlift": 0.2252,
      "Push Press": 0.0545,
      "Squat": 0.6347
    }
  }
}

---

## 🧪 What Success Means

- /health works in browser
- curl /analyze returns JSON
- prediction looks reasonable

At this point, your ML pipeline is working.

---

# 📶 Test From Phone (Network Check)

Find your Mac IP:

ipconfig getifaddr en0

Open on phone:

http://YOUR_IP:8000/health

If it doesn’t load:
- same WiFi required
- backend not running
- firewall blocking

---

# 📱 Mobile App Setup

In mobile/App.js:

const API_BASE_URL = 'http://YOUR_IP:8000';

Then run:

cd mobile
npm install
npx expo start -c

---

# 🧪 App Test Flow

1. Tap Check Backend
2. Tap Select Video
3. Tap Analyze Form

---

# ⚠️ Known Limitations

- Slow (~1 min for long videos)
- Requires side-view full body
- Confidence may be moderate (depends on training data)

---

# 🚀 Next Improvements

## Speed
- downsample frames (every 3–5 frames)
- resize video before processing
- limit clip length

## Accuracy
- more labeled data
- better angle consistency
- class balancing

## Product Features
- detect form faults (knees, depth, back angle)
- real-time feedback
- rep counting

---

# 🎯 Current Status

Full pipeline working  
Real model inference  
Mobile → backend → response loop  
Next: optimization + coaching insights