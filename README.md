# 🏋️ FormCheck AI

AI-powered exercise analysis platform that uses computer vision, pose estimation, and machine learning to analyze lifting technique from video.

Users upload a video and receive:

- Exercise classification
- Rep detection
- Rep scoring
- Biomechanics feedback
- Coaching cues
- Overlay analysis videos
- Phase-by-phase movement breakdowns

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

# Features

## Exercise Recognition

### Strength Training

- Back Squat
- Front Squat
- Overhead Squat
- Deadlift
- Bench Press
- Push Press

### Olympic Weightlifting

- Clean
- Split Jerk
- Clean & Jerk
- Snatch

---

## Rep Analysis

Automatically detects:

- Rep start
- Descent / eccentric phase
- Bottom position
- Ascent / concentric phase
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

- Back rounding detection
- Hip hinge quality
- Lockout analysis
- Bar path tracking

### Bench Press Analysis

- Range of motion
- Lockout quality
- Stability assessment

### Push Press Analysis

- Dip quality
- Bar path tracking
- Overhead lockout
- Timing analysis

---

## Visual Feedback

### Overlay Video Generation

FormCheck AI generates replay videos showing:

- Rep boundaries
- Scores
- Coaching notes
- Movement analysis

### Phase Image Generation

#### Squat

- Setup
- Descent
- Bottom
- Ascent
- Lockout

#### Deadlift

- Setup
- Pull
- Mid-Pull
- Finish
- Lockout

#### Push Press

- Setup
- Dip
- Drive
- Catch
- Lockout

#### Olympic Lifts

- Setup
- First Pull
- Extension
- Catch
- Finish

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
Overlay Rendering
      ↓
JSON Response
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

## Mobile

- React Native
- Expo

## Machine Learning

- LSTM Sequence Models
- Pose Landmark Extraction
- Feature Engineering
- Movement Routing Models
- Biomechanics Rule Engine

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
- overlay_video_url
- phase_images

Example:

```json
{
  "exercise_label": "Back Squat",
  "confidence": 0.80,
  "analysis_mode": "detailed_rep_analysis",
  "rep_feedback": [
    {
      "rep": 1,
      "score": 8.2,
      "grade": "Good"
    }
  ]
}
```

---

## Generate Visuals

```http
POST /generate_visuals
```

Creates:

- Phase review images
- Movement snapshots
- Exercise-specific breakdowns

---

## Generate Overlay

```http
POST /generate_overlay
```

Creates:

- Annotated replay video
- Rep markers
- Coaching overlays
- Movement analysis timeline

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

Server:

```text
http://127.0.0.1:8000
```

Swagger Docs:

```text
http://127.0.0.1:8000/docs
```

---

## Mobile App

```bash
cd mobile

npm install

npx expo start
```

Run on:

- iPhone
- Android
- iOS Simulator
- Expo Go

---

# Example API Response

```json
{
  "exercise_label": "Back Squat",
  "confidence": 0.80,
  "analysis_mode": "detailed_rep_analysis",
  "feedback": [
    "Predicted exercise: Back Squat.",
    "Model confidence: 80.0%."
  ],
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

### Powerlifting

- Back Squat
- Front Squat
- Overhead Squat
- Bench Press
- Deadlift

### Weightlifting

- Clean
- Split Jerk
- Clean & Jerk
- Snatch

### Pressing

- Push Press

---

# Roadmap

- Strict Press
- Thruster
- Additional CrossFit movements
- Historical athlete tracking
- Coach dashboard
- Team analytics
- Performance trend analysis

---

# Author

## Joseph Kamil

AI Engineer | Full Stack Developer

University of Michigan

GitHub: https://github.com/kamilj62

Built with computer vision, machine learning, and a passion for improving movement quality through AI.