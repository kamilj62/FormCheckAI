# 🏋️ FormCheck AI

AI-powered exercise analysis platform that uses computer vision, pose estimation, biomechanics, and machine learning to analyze lifting technique from video.

FormCheck AI allows athletes, coaches, and fitness enthusiasts to upload a lifting video and receive:

- Exercise Classification
- Rep Detection
- Rep Scoring
- Biomechanics Feedback
- Coaching Zones
- Set Summaries
- Phase Review Images
- Overlay Video Generation
- Functional Fitness Analysis
- Olympic Lift Analysis

---

# Latest Release

## v1.4.0 – Beta & Production Readiness

### Highlights

- 18 supported exercise categories
- 18/18 canonical regression benchmark passing
- Exercise classification and rep-count validation
- Beta exercise confirmation and correction workflow
- Real-world classifier feedback collection
- Versioned beta analysis records
- Improved Push Press routing and rep recovery
- Phase Review feedback collection
- HTTPS production API
- AWS Elastic Beanstalk backend deployment
- Vercel API proxy and web deployment
- React Native / Expo iOS production configuration
- TestFlight-ready build configuration

### Beta Feedback System

Each analysis receives a unique `analysis_id`.

Users can confirm or correct:

- Exercise classification
- Rep count
- Analysis helpfulness
- Phase Review accuracy

Beta records preserve both:

- `predicted_exercise`
- `confirmed_exercise`

This allows FormCheck AI to measure real-world classifier accuracy and identify movement-specific confusion patterns without overwriting the original model prediction.

### Current Validation

Canonical regression benchmark:

| Metric | Result |
|---|---:|
| Test Cases | 18 |
| Passed | 18 |
| Failed | 0 |
| Accuracy | 100% |

The canonical suite currently covers all supported movement families: squat, press, hinge, Olympic weightlifting, and functional fitness.

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

✅ Functional Fitness Analysis

✅ Olympic Lift Analysis

✅ Mobile & Web Support

---

# Demo

## Back Squat Analysis

### Input Video

![Back Squat Input](docs/examples/backsquat_input.gif)

### Overlay Analysis

![Overlay Video](docs/examples/backsquat_overlay.gif)

### Phase Review

| Setup | Descent | Bottom | Ascent | Lockout |
|--------|--------|--------|--------|--------|
| ![](docs/examples/setup.jpg) | ![](docs/examples/descent.jpg) | ![](docs/examples/bottom.jpg) | ![](docs/examples/ascent.jpg) | ![](docs/examples/lockout.jpg) |

---

# Current Supported Movements

## Squat Family

- Back Squat
- Front Squat
- Overhead Squat

## Pressing

- Strict Press
- Push Press
- Thruster

## Strength

- Bench Press
- Deadlift

## Functional Fitness

- Pull-Up
- Push-Up
- Handstand Push-Up
- Bar Muscle-Up
- Ring Muscle-Up
- Burpee

## Olympic Weightlifting

- Clean
- Split Jerk
- Clean & Jerk
- Snatch

---

# Functional Fitness Analysis

## Pull-Up

Phase Images:

- Hang
- Pull
- Top
- Descent
- Finish

## Push-Up

Phase Images:

- Setup
- Descent
- Bottom
- Ascent
- Lockout

## Handstand Push-Up

Phase Images:

- Setup
- Descent
- Bottom
- Ascent
- Lockout

## Bar Muscle-Up

Phase Images:

- Hang
- Pull
- Transition
- Dip
- Lockout
- Finish

## Ring Muscle-Up

Phase Images:

- Hang
- Pull
- Transition
- Dip
- Lockout
- Finish

## Burpee

Phase Images:

- Start
- Hands Down
- Plank
- Stand
- Finish

---

# Olympic Weightlifting

Supported:

- Clean
- Split Jerk
- Clean & Jerk
- Snatch

Features:

- Rep Detection
- Rep Scoring
- Coaching Zones
- Biomechanics Feedback
- Phase Review Images
- Overlay Video Generation

---

# What FormCheck AI Analyzes

## Squat Analysis

- Depth Assessment
- Knee Valgus Detection
- Forward Torso Lean
- Heel Rise Detection

## Deadlift Analysis

- Back Rounding
- Hip Hinge Quality
- Lockout Analysis
- Bar Path Tracking

## Bench Press Analysis

- Range of Motion
- Stability Assessment
- Lockout Quality

## Functional Fitness

- Pull Height
- Transition Quality
- Support Position
- Lockout Quality
- Body Alignment
- Stability

## Olympic Lift Analysis

- First Pull
- Extension
- Turnover
- Catch Position
- Overhead Stability
- Lockout Quality
- Bar Path Analysis

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
Base Movement Classifier
      ↓
Movement Routers
  ├── Squat Router
  ├── Olympic Router
  └── Press / Bodyweight Routing
      ↓
Biomechanics & Protection Rules
      ↓
Final Exercise Classification
      ↓
Rep Detection
      ↓
Biomechanics Analysis
      ↓
Rep Scoring & Coaching Feedback
      ↓
Phase Review Images
      ↓
Overlay Generation
      ↓
Results + Beta Feedback
```

The routing layer combines learned classifiers with biomechanics-based protections to reduce cross-family misclassification and preserve strong movement-specific signals.

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
- Expo SDK 54
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

---

# API Endpoints

## Health Check

`GET /health`

## Analyze Video

`POST /analyze`

Returns fields including:

- `analysis_id`
- `exercise_label`
- `confidence`
- `analysis_mode`
- `rep_feedback`
- `set_summary`
- `coaching_zones`
- `phase_images`
- `overlay_video_url`

## Submit Analysis Feedback

`POST /analysis_feedback`

Stores beta validation data such as:

- `predicted_exercise`
- `confirmed_exercise`
- `was_corrected`
- `helpful`
- `rep_count_correct`
- `corrected_rep_count`
- `phase_review_accurate`
- `phase_review_issue`

## Generate Overlay

`POST /generate_overlay`

---

# Installation

## Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Mobile

```bash
cd mobile
npm install
npx expo start
```

---

# Smoke Test Results (v1.3.0)

| Exercise | Status |
|-----------|---------|
| Back Squat | ✅ |
| Front Squat | ✅ |
| Overhead Squat | ✅ |
| Deadlift | ✅ |
| Bench Press | ✅ |
| Strict Press | ✅ |
| Push Press | ✅ |
| Thruster | ✅ |
| Pull-Up | ✅ |
| Push-Up | ✅ |
| Handstand Push-Up | ✅ |
| Bar Muscle-Up | ✅ |
| Ring Muscle-Up | ✅ |
| Burpee | ✅ |
| Clean | ✅ |
| Split Jerk | ✅ |
| Clean & Jerk | ✅ |
| Snatch | ✅ |

---

# Roadmap

- Chest-to-Bar Pull-Up
- Toes-to-Bar
- Kipping Pull-Up
- Butterfly Pull-Up
- Double Unders
- Rowing Analysis
- Progress Tracking
- Coach Dashboard

---

# Author

Joseph Kamil

University of Michigan

https://github.com/kamilj62
