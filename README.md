# 🏋️ FormCheck AI

AI-powered exercise analysis platform that uses computer vision, pose estimation, and machine learning to analyze lifting technique from video.

FormCheck AI allows athletes, coaches, and fitness enthusiasts to upload a lifting video and receive detailed feedback on movement quality, rep performance, biomechanics, coaching recommendations, phase review images, and annotated overlay videos.

---

# Latest Release

## v1.2.0 – Olympic Overlay Release

Added:

- Detailed Clean Analysis
- Detailed Split Jerk Analysis
- Detailed Clean & Jerk Analysis
- Detailed Snatch Analysis
- Olympic Coaching Zones
- Olympic Phase Images
- Olympic Overlay Videos
- Improved Overlay Rendering
- S3 Overlay Delivery

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

✅ Strict Press Analysis

✅ Thruster Analysis

✅ Olympic Lift Analysis

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

# Current Supported Movements

## Detailed Rep Analysis

### Squat Family

- Back Squat
- Front Squat
- Overhead Squat

### Pressing

- Strict Press
- Push Press
- Thruster

### Strength

- Bench Press
- Deadlift

### Olympic Weightlifting

- Clean
- Split Jerk
- Clean & Jerk
- Snatch

All supported movements include:

- Rep Detection
- Rep Scoring
- Coaching Zones
- Set Summary
- Phase Review Images
- Overlay Video Generation

---

# Olympic Weightlifting

FormCheck AI provides detailed rep analysis for Olympic lifts.

Supported:

- Clean
- Split Jerk
- Clean & Jerk
- Snatch

Features include:

- Rep Detection
- Rep Scoring
- Coaching Zones
- Biomechanics Feedback
- Phase Review Images
- Overlay Video Generation

---

## Clean

- Rep Detection
- Rep Scoring
- First Pull Analysis
- Extension Analysis
- Turnover Analysis
- Catch Position Analysis
- Coaching Zones
- Phase Images
- Overlay Videos

---

## Split Jerk

- Rep Detection
- Rep Scoring
- Dip Analysis
- Drive Analysis
- Lockout Analysis
- Catch Position Analysis
- Coaching Zones
- Phase Images
- Overlay Videos

---

## Clean & Jerk

- Clean Analysis
- Jerk Analysis
- Rep Scoring
- Coaching Zones
- Phase Images
- Overlay Videos

---

## Snatch

- Rep Detection
- Rep Scoring
- First Pull Analysis
- Extension Analysis
- Turnover Analysis
- Overhead Catch Analysis
- Stability Analysis
- Coaching Zones
- Phase Images
- Overlay Videos

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

### Strict Press Analysis

- Knee bend detection
- Overhead lockout
- Torso positioning
- Head position at lockout

### Thruster Analysis

- Squat depth
- Torso position
- Overhead lockout
- Bar path
- Rep scoring

### Olympic Lift Analysis

- First Pull
- Extension
- Turnover
- Catch Position
- Overhead Stability
- Lockout Quality
- Bar Path Analysis
- Coaching Zones

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

### Push Press / Strict Press / Thruster

- Setup
- Dip
- Drive
- Catch
- Lockout

### Clean

- Setup
- First Pull
- Extension
- Catch
- Finish

### Split Jerk

- Setup
- Dip
- Drive
- Catch
- Lockout
- Finish

### Clean & Jerk

- Setup
- Clean Catch
- Jerk Dip
- Jerk Drive
- Jerk Catch
- Finish

### Snatch

- Setup
- First Pull
- Extension
- Catch
- Finish

---

## Overlay Video Generation

FormCheck AI generates annotated replay videos showing:

- Exercise classification
- Rep scores
- Coaching feedback
- Olympic lift phases
- Movement analysis
- Pose overlays

Overlay rendering is performed asynchronously and delivered through Amazon S3.

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
Amazon S3 Delivery
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

### Storage

- Amazon S3

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

---

## Generate Phase Review

```http
POST /generate_visuals
```

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
  "overlay_video_url": "https://..."
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

# Roadmap

## Olympic Lift Enhancements

- Multi-rep Olympic lift detection
- Advanced bar path tracking
- Power Clean analysis
- Hang Clean analysis
- Push Jerk analysis

## Platform Features

- Athlete History
- Coach Dashboard
- Team Analytics
- Performance Trends
- Movement Comparison
- Expanded CrossFit Exercise Library

---

## Smoke Test Results (v1.2.2)

| Exercise | Status |
|-----------|---------|
| Back Squat | ✅ |
| Deadlift | ✅ |
| Bench Press | ✅ |
| Push Press | ✅ |
| Clean & Jerk | ✅ |

### Verified Features

- ✅ Exercise Classification
- ✅ Rep Detection
- ✅ Rep Scoring
- ✅ Coaching Zones
- ✅ Phase Images
- ✅ Overlay Videos
- ✅ S3 Overlay Delivery
- ✅ Olympic Lift Analysis

### Production Tags

- v1.2.0-olympic-overlays
- v1.2.1-readme
- v1.2.2-smoke-test-fixes

---

# Author

## Joseph Kamil

AI Engineer | Full Stack Developer

University of Michigan

GitHub:
https://github.com/kamilj62

Built using computer vision, machine learning, biomechanics analysis, and modern AI tooling to help athletes move better.