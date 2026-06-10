# 🏋️ FormCheck AI

AI-powered exercise analysis platform that uses computer vision, pose estimation, and machine learning to analyze lifting technique from video.

FormCheck AI allows athletes, coaches, and fitness enthusiasts to upload a lifting video and receive detailed feedback on movement quality, rep performance, biomechanics, and coaching recommendations.

---

# 🚀 Live Features

✅ Exercise Classification

✅ Rep Detection

✅ Rep Scoring

✅ Biomechanics Feedback

✅ Coaching Zones

✅ Set Summaries

✅ Phase Review Images

✅ Overlay Video Generation

✅ Background Overlay Processing

✅ Mobile & Web Support

---

# Demo

## Back Squat Analysis

### Input Video

![Back Squat Input](docs/examples/backsquat_input.gif)

---

### Analysis Output

```json
{
  "exercise_label": "Back Squat",
  "confidence": 0.80,
  "analysis_mode": "detailed_rep_analysis",
  "rep_feedback": [
    {
      "rep": 1,
      "score": 8.2,
      "grade": "Good",
      "issues": [
        "Depth is close, but could be slightly lower.",
        "Knees cave inward noticeably."
      ]
    }
  ]
}
```

---

### Overlay Analysis

![Overlay Video](docs/examples/backsquat_overlay.gif)

---

### Phase Review

| Setup | Descent | Bottom | Ascent | Lockout |
|--------|--------|--------|--------|--------|
| ![](docs/examples/setup.jpg) | ![](docs/examples/descent.jpg) | ![](docs/examples/bottom.jpg) | ![](docs/examples/ascent.jpg) | ![](docs/examples/lockout.jpg) |

---

# Supported Exercises

## Strength Training

- Back Squat
- Front Squat
- Overhead Squat
- Deadlift
- Bench Press
- Push Press

## Olympic Weightlifting

- Clean
- Split Jerk
- Clean & Jerk
- Snatch

---

# What FormCheck AI Analyzes

## Rep Detection

Automatically identifies:

- Rep start
- Eccentric phase
- Bottom position
- Concentric phase
- Lockout
- Rep completion

Returns:

- Total reps
- Best rep
- Worst rep
- Average score
- Consistency trend

---

## Biomechanics Feedback

### Squat Analysis

- Depth assessment
- Knee valgus detection
- Forward torso lean
- Heel rise detection

### Deadlift Analysis

- Back rounding
- Hip hinge quality
- Lockout analysis
- Bar path tracking

### Bench Press Analysis

- Range of motion
- Stability assessment
- Lockout quality

### Push Press Analysis

- Dip quality
- Timing analysis
- Overhead lockout
- Bar path tracking

---

# Visual Feedback

## Phase Review

Generates key movement snapshots.

### Squat

- Setup
- Descent
- Bottom
- Ascent
- Lockout

### Deadlift

- Setup
- Pull
- Mid-Pull
- Finish
- Lockout

### Push Press

- Setup
- Dip
- Drive
- Catch
- Lockout

### Olympic Lifts

- Setup
- First Pull
- Extension
- Catch
- Finish

---

## Overlay Video Generation

FormCheck AI generates annotated replay videos showing:

- Rep boundaries
- Rep scores
- Coaching notes
- Movement analysis
- Exercise classification

Overlay rendering is performed asynchronously using background processing jobs.

---

# Architecture

```text
Video Upload
      ↓
MediaPipe Pose
      ↓
Landmark Extraction
      ↓
Feature Engineering
      ↓
Movement Classification
      ↓
Rep Detection
      ↓
Biomechanics Analysis
      ↓
Feedback Generation
      ↓
Phase Images
      ↓
Start Overlay Job
      ↓
Background Overlay Worker
      ↓
Overlay Status Polling
      ↓
Overlay Video
```

---

# Tech Stack

## Backend

- FastAPI
- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Pandas

## Frontend

- React Native
- Expo
- expo-video

## Machine Learning

- LSTM Sequence Models
- Pose Landmark Extraction
- Feature Engineering
- Movement Routing Models
- Biomechanics Rule Engine

## Deployment

### Frontend

- Vercel

### Backend

- AWS Elastic Beanstalk
- Docker

### Inference

- TensorFlow
- MediaPipe

---

# API Endpoints

## Health Check

```http
GET /health
```

Returns:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Analyze Video

```http
POST /analyze
```

Returns:

- exercise_label
- confidence
- feedback
- rep_feedback
- set_summary
- coaching_zones
- phase_images

Example:

```json
{
  "exercise_label": "Back Squat",
  "confidence": 0.80,
  "analysis_mode": "detailed_rep_analysis"
}
```

---

## Generate Phase Review

```http
POST /generate_visuals
```

Creates:

- Phase review images
- Movement snapshots
- Exercise-specific breakdowns

---

## Start Overlay Job

```http
POST /start_overlay
```

Returns:

```json
{
  "job_id": "1d76ef7edce7",
  "status": "processing"
}
```

---

## Overlay Status

```http
GET /overlay_status/{job_id}
```

Returns:

```json
{
  "status": "ready",
  "overlay_video_url": "/outputs/overlay_ab14dd6d.mp4"
}
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/kamilj62/FormCheckAI.git

cd FormCheckAI
```

---

## Backend Setup

```bash
cd backend

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger Docs:

```text
http://127.0.0.1:8000/docs
```

---

## Mobile Setup

```bash
cd mobile

npm install

npx expo start
```

Supported Platforms:

- iPhone
- Android
- Expo Go
- Web Browser

---

# Example Response

```json
{
  "exercise_label": "Back Squat",
  "confidence": 0.80,
  "feedback": [
    "Predicted exercise: Back Squat.",
    "Model confidence: 80.0%."
  ],
  "rep_feedback": [
    {
      "rep": 1,
      "score": 8.2,
      "grade": "Good"
    }
  ],
  "set_summary": {
    "detected_reps": 1,
    "avg_rep_score": 8.2,
    "trend": "Form appears consistent across the set."
  }
}
```

---

# Current Supported Movements

## Powerlifting

- Back Squat
- Front Squat
- Overhead Squat
- Bench Press
- Deadlift

## Weightlifting

- Clean
- Split Jerk
- Clean & Jerk
- Snatch

## Pressing

- Push Press

---

# Roadmap

### Next Movements

- Strict Press
- Thruster

### Future Features

- Athlete History
- Coach Dashboard
- Team Analytics
- Performance Trends
- Movement Comparison
- Expanded CrossFit Exercise Library

---

# Author

## Joseph Kamil

AI Engineer | Full Stack Developer

University of Michigan

GitHub:
https://github.com/kamilj62

Built using computer vision, machine learning, biomechanics analysis, and modern AI tooling to help athletes move better.