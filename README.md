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

## v1.3.0 – Functional Fitness Release

### Added

- Pull-Up Analysis
- Push-Up Analysis
- Handstand Push-Up Analysis
- Bar Muscle-Up Analysis
- Ring Muscle-Up Analysis
- Burpee Analysis
- Front Squat Analysis
- Overhead Squat Improvements
- Functional Fitness Phase Images
- Biomechanics Classification Overrides
- Olympic Router
- Squat Router
- Improved Rep Detection
- Improved Phase Selection Logic
- Vercel + AWS Elastic Beanstalk Production Deployment

### Override Protections

- Push-Up ↔ Deadlift
- Handstand Push-Up ↔ Bench Press
- Pull-Up ↔ Olympic Lifts
- Burpee ↔ Clean
- Thruster ↔ Snatch
- Push Press ↔ Snatch

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
Movement Classifier
      ↓
Olympic Router
      ↓
Squat Router
      ↓
Biomechanics Override Engine
      ↓
Rep Detection
      ↓
Biomechanics Analysis
      ↓
Feedback Generation
      ↓
Phase Images
      ↓
Overlay Generation
      ↓
Results
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

GET /health

## Analyze Video

POST /analyze

Returns:

- exercise_label
- confidence
- feedback
- rep_feedback
- set_summary
- coaching_zones
- phase_images

## Generate Overlay

POST /generate_overlay

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
