# 📱 FormCheck AI — AI-Powered Movement Analysis & Coaching

FormCheck AI is a computer vision + deep learning platform that analyzes exercise videos, classifies movements, evaluates biomechanics, generates coaching feedback, and creates visual phase breakdowns automatically.

It acts like having an AI coach review your lifts frame-by-frame.

Built with:

- **FastAPI backend**
- **Expo React Native mobile app**
- **TensorFlow / Keras deep learning**
- **MediaPipe Pose landmark extraction**
- **OpenCV computer vision**
- **Biomechanics scoring engine**
- **Automatic overlay video generation**
- **Phase-by-phase movement visualization**

---

# Demo

Upload a lift video →

FormCheck AI returns:

✅ movement classification  
✅ rep counting  
✅ biomechanics feedback  
✅ coaching cues  
✅ rep grading  
✅ overlay analysis video  
✅ phase images  
✅ coaching zones  

---

# Features

# ✅ Exercise Classification

Automatically identifies the exercise being performed.

Supports:

## 🏋️ Strength / Olympic Lifts

- Bench Press
- Deadlift
- Back Squat
- Front Squat
- Overhead Squat
- Strict Press
- Push Press
- Split Jerk
- Thruster
- Clean
- Snatch
- Clean & Jerk

## 🤸 Gymnastics / CrossFit

- Pull-up
- Bar Muscle-up
- Ring Muscle-up

Future planned:

- Toes-to-bar
- Handstand Push-up
- Box Jump
- Rowing stroke analysis
- Running gait analysis

---

# ✅ Rep Detection

Automatically detects:

- rep start
- eccentric phase
- bottom position
- concentric phase
- lockout
- rep completion

Returns:

- total reps
- best rep
- worst rep
- average score
- consistency trend

---

# ✅ Biomechanics Analysis

Analyzes movement quality using joint tracking and movement heuristics.

Metrics include:

- knee angle
- hip angle
- torso angle
- elbow angle
- shoulder position
- valgus ratio
- lockout quality
- spinal neutrality
- bar path
- tempo / control
- overhead stability

---

# ✅ Coaching Feedback

Provides actionable cues.

Examples:

### Squat

- Drive knees out
- Keep chest tall
- Hit full depth
- Maintain heel pressure

### Deadlift

- Brace core
- Neutral spine
- Push floor away
- Finish tall with glutes

### Bench Press

- Touch lower chest
- Keep wrists stacked
- Full lockout
- Drive through feet

### Olympic Lifts

- Stay over bar
- Finish extension
- Fast elbows
- Punch overhead aggressively

---

# ✅ Phase Image Generation

Automatically creates key movement snapshots.

---

## Clean

Generates:

- setup
- first pull
- extension
- catch
- finish

---

## Split Jerk

Generates:

- setup
- dip
- drive
- catch
- recovery
- finish

---

## Push Press

Generates:

- setup
- dip
- drive
- catch
- lockout

---

## Squat

Generates:

- setup
- descent
- bottom
- ascent
- lockout

---

## Deadlift

Generates:

- setup
- pull
- mid
- finish
- lockout

---

## Pull-up

Generates:

- hang
- pull
- top
- descent
- finish

---

## Muscle-ups

Generates:

- hang
- pull
- transition
- dip
- lockout
- finish

---

# ✅ Overlay Video Rendering

FormCheck AI creates an annotated replay video showing:

- rep boundaries
- labels
- score overlays
- coaching zones
- biomechanics notes
- analysis timeline

This gives users visual feedback—not just numbers.

---

# ✅ Smart Override Engine

A biomechanics-aware rule engine improves classification beyond model predictions.

Overrides include:

- Thruster detection
- Push Press detection
- Split Jerk detection
- Pull-up detection
- Bar Muscle-up detection
- Ring Muscle-up detection
- Olympic lift routing
- Squat family routing

This dramatically improves real-world accuracy.

---

# Backend Pipeline

Video →

Frame extraction →

MediaPipe Pose →

Landmark extraction →

Feature engineering →

Velocity features →

Sequence model →

Biomechanics override →

Rep segmentation →

Movement scoring →

Feedback generation →

Phase image creation →

Overlay rendering →

JSON response

---

# Model Performance

Current production classifier:

### Movement Router v2

Performance:

| Movement | Accuracy |
|---------|----------|
| Deadlift | 98.8% |
| Push Press | 97.6% |
| Back Squat | 96.8% |
| Front Squat | 98.1% |

Confusion is minimal.

Biomechanics overrides further improve final prediction quality.

---

# Tech Stack

## Backend

- FastAPI
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- MediaPipe

## Frontend

- React Native
- Expo
- JavaScript

## ML / CV

- LSTM sequence models
- Landmark feature engineering
- Motion velocity vectors
- Heuristic override engine
- Rep segmentation logic

---

# Installation

## Backend

```bash
git clone https://github.com/YOUR_USERNAME/formcheck-ai.git

cd formcheck-ai/backend

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Run server:

```bash
uvicorn app.main:app --reload
```

Server:

```text
http://127.0.0.1:8000
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

---

## Frontend

```bash
cd frontend

npm install

npx expo start
```

Run on:

- iPhone
- Android
- simulator
- Expo Go

---

# API Example

POST:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
-H "accept: application/json" \
-F "file=@deadlift.mov"
```

Response:

```json
{
  "exercise_label": "Deadlift",
  "confidence": 0.95,
  "analysis_mode": "detailed_rep_analysis",
  "feedback": [
    "Brace core and maintain neutral spine."
  ],
  "rep_feedback": [
    {
      "rep": 1,
      "score": 8.5,
      "grade": "Good"
    }
  ],
  "overlay_video_url": "/outputs/overlay.mp4",
  "phase_images": {
    "setup": "/outputs/setup.jpg",
    "pull": "/outputs/pull.jpg",
    "finish": "/outputs/finish.jpg"
  }
}
```

---

# Vision

FormCheck AI aims to become:

**The AI movement coach for everyone**

Applications:

- strength training
- Olympic lifting
- CrossFit
- personal training
- rehab movement screening
- sports performance
- remote coaching

---

# Author

**Joseph Kamil**

Full Stack + AI Engineer

Built with deep learning, computer vision, and an obsession for biomechanics.

---