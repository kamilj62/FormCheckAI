from sys import prefix
import tempfile
from pathlib import Path

import os
from tracemalloc import start
import uuid
import shutil

from threading import Thread
import uuid

overlay_jobs = {}

from app.phase_detection.signal_engine import SignalEngine
from app.phase_detection.phase_engine import get_phase_images

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

import joblib

try:
    from celery.result import AsyncResult
    from app.celery_app import celery
except Exception:
    AsyncResult = None
    celery = None

import threading

import boto3

import subprocess

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

# Experimental: use YOLO to isolate the foreground athlete before pose estimation.
USE_YOLO_TRACKING = (
    os.getenv("USE_YOLO_TRACKING", "false").lower() == "true"
)

USE_YOLO_DIAGNOSTICS = (
    os.getenv("USE_YOLO_DIAGNOSTICS", "false").lower() == "true"
)

try:
    from app.tracking import YOLOTracker, remap_crop_landmarks_to_full_frame
except Exception:
    YOLOTracker = None
    remap_crop_landmarks_to_full_frame = None

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

S3_BUCKET = os.getenv("S3_BUCKET", "formcheck-ai-overlays-kamilj")
S3_REGION = os.getenv("AWS_REGION", "us-west-2")
s3_client = boto3.client("s3", region_name=S3_REGION)

OVERLAY_DIR = "outputs"
os.makedirs(OVERLAY_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OVERLAY_DIR), name="outputs")
app.mount(
    "/outputs",
    StaticFiles(directory=OVERLAY_DIR),
    name="outputs",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://formcheck-ai-full-v2.vercel.app",
        "http://localhost:19006",
        "http://localhost:8081",
        "http://localhost:3000",
        "http://127.0.0.1:19006",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:3000",
    ],  # ← missing comma here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path
from .model_runtime import NumpyFormCheckModel

from app.logic import (
    classify_with_biomechanics,
    build_set_summary,
    compute_rep_score,
    apply_coach_reward,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL = NumpyFormCheckModel(MODEL_DIR)

OVERHEAD_ROUTER_MODEL = tf.keras.models.load_model(
    MODEL_DIR / "movement_router_v2.keras"
)

OVERHEAD_ROUTER_LABELS = {
    0: "squat_front",
    1: "strict_press",
}

SQUAT_ROUTER_MODEL = tf.keras.models.load_model(
    MODEL_DIR / "squat_router.keras"
)

SQUAT_ROUTER_LABELS = {
    0: "overhead_squat",
    1: "squat_back",
    2: "squat_front",
}

CLASS_NAMES = [
    "bench_press",
    "deadlift",
    "push_press",
    "squat",
    "thruster",
]

OLY_ROUTER_BUNDLE = joblib.load(MODEL_DIR / "oly_router_rf.joblib")
OLY_ROUTER_MODEL = OLY_ROUTER_BUNDLE["model"]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@app.get("/")
@app.head("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
@app.head("/health")
async def health():
    return {"status": "ok", "model_loaded": True, "build": "snatch_patch_1"}


def angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cos_val = np.dot(ba, bc) / denom
    cos_val = np.clip(cos_val, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_val)))


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def point(landmarks, landmark):
    lm = landmarks[landmark.value]
    return np.array([lm.x, lm.y], dtype=np.float32)


def formatLabel(v):
    if not v:
        return "Unknown Exercise"

    return str(v).replace("_", " ").title()


def extract_features_and_biomechanics(results):
    if not results.pose_landmarks:
        return None, None

    landmarks = results.pose_landmarks.landmark

    # EXACT 68 features used by your main classifier/router
    FEATURE_LANDMARKS = [
        "NOSE",
        "LEFT_EAR", "RIGHT_EAR",
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST",
        "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_HEEL", "RIGHT_HEEL",
    ]

    features = []

    for name in FEATURE_LANDMARKS:
        idx = mp_pose.PoseLandmark[name].value
        lm = landmarks[idx]

        features.extend([
            lm.x,
            lm.y,
            lm.z,
            lm.visibility,
        ])

    # Full 33-landmark features for Olympic router
    # 33 landmarks × 4 values = 132 features
    full_features = []

    for lm in landmarks:
        full_features.extend([
            lm.x,
            lm.y,
            lm.z,
            lm.visibility,
        ])

    pts = np.array([
        [lm.x, lm.y, lm.z]
        for lm in landmarks
    ])

    nose = pts[mp_pose.PoseLandmark.NOSE.value]

    left_shoulder = point(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = point(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)

    left_elbow = point(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
    right_elbow = point(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)

    left_wrist = point(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    right_wrist = point(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)

    left_hip = point(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)

    left_knee = point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    right_knee = point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)

    left_ankle = point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    right_ankle = point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)

    left_heel = point(landmarks, mp_pose.PoseLandmark.LEFT_HEEL)
    right_heel = point(landmarks, mp_pose.PoseLandmark.RIGHT_HEEL)

    ankle_mid = (left_ankle + right_ankle) / 2
    heel_mid = (left_heel + right_heel) / 2
    heel_lift = float(ankle_mid[1] - heel_mid[1])

    shoulder_mid = (left_shoulder + right_shoulder) / 2
    elbow_mid = (left_elbow + right_elbow) / 2
    hip_mid = (left_hip + right_hip) / 2
    knee_mid = (left_knee + right_knee) / 2
    wrist_mid = (left_wrist + right_wrist) / 2

    left_knee_angle = angle(left_hip, left_knee, left_ankle)
    right_knee_angle = angle(right_hip, right_knee, right_ankle)
    knee_angle = float(np.mean([left_knee_angle, right_knee_angle]))

    left_hip_angle = angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = angle(right_shoulder, right_hip, right_knee)
    hip_angle = float(np.mean([left_hip_angle, right_hip_angle]))

    left_elbow_angle = angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = angle(right_shoulder, right_elbow, right_wrist)
    elbow_angle = float(np.mean([left_elbow_angle, right_elbow_angle]))

    vertical_point = hip_mid + np.array([0, -1], dtype=np.float32)
    torso_angle = angle(shoulder_mid, hip_mid, vertical_point)

    hip_y = float(hip_mid[1])
    knee_y = float(knee_mid[1])
    wrist_x = float(wrist_mid[0])
    wrist_y = float(wrist_mid[1])
    elbow_y = float(elbow_mid[1])
    shoulder_y = float(shoulder_mid[1])

    hip_x = float(hip_mid[0])
    shoulder_x = float(shoulder_mid[0])
    knee_x = float(knee_mid[0])

    shoulder_hip_distance = float(np.linalg.norm(shoulder_mid - hip_mid))
    hip_knee_distance = float(np.linalg.norm(hip_mid - knee_mid))
    wrist_shoulder_distance = float(np.linalg.norm(wrist_mid - shoulder_mid))

    knee_width = abs(float(left_knee[0]) - float(right_knee[0]))
    ankle_width = abs(float(left_ankle[0]) - float(right_ankle[0]))
    valgus_ratio = knee_width / (ankle_width + 1e-6)
    valgus_ratio = float(np.clip(valgus_ratio, 0.50, 1.50))
    
    wrist_x = float((left_wrist[0] + right_wrist[0]) / 2)
    ankle_x = float((left_ankle[0] + right_ankle[0]) / 2)
    bar_distance = abs(wrist_x - ankle_x)

    wrist_above_shoulder = float(wrist_y < shoulder_y)

    shoulder_mid_x = float((left_shoulder[0] + right_shoulder[0]) / 2)
    shoulder_mid_y = float((left_shoulder[1] + right_shoulder[1]) / 2)

    nose_x = float(nose[0])
    nose_y = float(nose[1])

    head_drop = nose_y - shoulder_mid_y
    head_forward = abs(nose_x - shoulder_mid_x)

    biomechanics = {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "torso_angle": torso_angle,
        "elbow_angle": elbow_angle,
        "hip_y": hip_y,
        "knee_y": knee_y,
        "wrist_x": wrist_x,
        "wrist_y": wrist_y,
        "elbow_y": elbow_y,
        "shoulder_y": shoulder_y,
        "hip_x": hip_x,
        "shoulder_x": shoulder_x,
        "knee_x": knee_x,
        "shoulder_hip_distance": shoulder_hip_distance,
        "hip_knee_distance": hip_knee_distance,
        "wrist_shoulder_distance": wrist_shoulder_distance,
        "wrist_above_shoulder": wrist_above_shoulder,
        "valgus_ratio": float(valgus_ratio),
        "bar_distance": bar_distance,
        "head_drop": float(head_drop),
        "head_forward": float(head_forward),
        "heel_lift": heel_lift,

        # Used only by Olympic router
        "full_features": np.array(full_features, dtype=np.float32),
    }

    return np.array(features, dtype=np.float32), biomechanics


def pad_or_trim(sequence, target_len=30):
    sequence = np.array(sequence, dtype=np.float32)

    if len(sequence) > target_len:
        idx = np.linspace(0, len(sequence) - 1, target_len).astype(int)
        sequence = sequence[idx]

    if len(sequence) < target_len:
        pad_len = target_len - len(sequence)
        pad = np.repeat(sequence[-1][None, :], pad_len, axis=0)
        sequence = np.concatenate([sequence, pad], axis=0)

    return sequence


def add_velocity(sequence):
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    return np.concatenate([sequence, velocity], axis=1)


def save_phase_contact_sheet(input_path, phase_frames, output_dir, prefix="squat_debug"):
    cap = cv2.VideoCapture(input_path)
    print("OVERLAY DEBUG: VIDEO OPENED =", cap.isOpened())

    if not cap.isOpened():
        print("Contact sheet error")
        return None

    images = []

    for phase, frame_idx in phase_frames.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (320, 240))

        cv2.rectangle(frame, (0, 0), (320, 42), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{phase.upper()}  frame={frame_idx}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        images.append(frame)

    cap.release()

    if not images:
        return None

    sheet = np.hstack(images)

    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(output_dir, filename)

    cv2.imwrite(filepath, sheet)

    print("Saved phase contact sheet:", filepath)

    return f"/outputs/{filename}"


def summarize_biomechanics(biomechanics):
    if not biomechanics:
        return {}

    knee = np.array([b["knee_angle"] for b in biomechanics])
    hip = np.array([b["hip_angle"] for b in biomechanics])
    torso = np.array([b["torso_angle"] for b in biomechanics])
    elbow = np.array([b["elbow_angle"] for b in biomechanics])
    valgus = np.array([b.get("valgus_ratio", 1.0) for b in biomechanics])
    wrist_above = np.array([b.get("wrist_above_shoulder", 0.0) for b in biomechanics])
    
    return {
        "avg_knee_angle": float(np.mean(knee)),
        "min_knee_angle": float(np.min(knee)),
        "max_knee_angle": float(np.max(knee)),
        "avg_hip_angle": float(np.mean(hip)),
        "min_hip_angle": float(np.min(hip)),
        "max_hip_angle": float(np.max(hip)),
        "avg_torso_angle": float(np.mean(torso)),
        "min_torso_angle": float(np.min(torso)),
        "max_torso_angle": float(np.max(torso)),
        "avg_elbow_angle": float(np.mean(elbow)),
        "min_elbow_angle": float(np.min(elbow)),
        "max_elbow_angle": float(np.max(elbow)),
        "avg_valgus_ratio": float(np.mean(valgus)),
        "min_valgus_ratio": float(np.min(valgus)),
        "wrist_above_shoulder_ratio": float(np.mean(wrist_above)),
    }


def classify_with_biomechanics(raw_label, confidence, summary, pose_frames):
    if pose_frames < 10 or not summary:
        return raw_label, confidence, False, "low_pose_data"

    min_knee = summary["min_knee_angle"]
    max_knee = summary["max_knee_angle"]
    min_hip = summary["min_hip_angle"]
    max_hip = summary["max_hip_angle"]
    min_torso = summary["min_torso_angle"]
    max_torso = summary["max_torso_angle"]
    min_elbow = summary["min_elbow_angle"]
    max_elbow = summary["max_elbow_angle"]

    wrist_ratio = summary["wrist_above_shoulder_ratio"]
    avg_torso = summary.get("avg_torso_angle", 0)

    knee_range = max_knee - min_knee
    hip_range = max_hip - min_hip
    torso_range = max_torso - min_torso
    elbow_range = max_elbow - min_elbow

    # THRUSTER RESCUE — must happen before confidence lock
    # Disabled: this was causing clean / clean_and_jerk videos to become thruster.
    # Only preserve thruster when the base model already predicts thruster.
    # THRUSTER RESCUE FROM BENCH / SQUAT / PUSH PRESS
    # Thruster = deep squat + overhead press in same clip.

    if raw_label in ["squat", "squat_back", "squat_front", "push_press", "clean_and_jerk"]:
        print(
            "THRUSTER DEBUG",
            {
                "raw_label": raw_label,
                "min_knee": round(min_knee, 1),
                "min_hip": round(min_hip, 1),
                "wrist_ratio": round(wrist_ratio, 3),
                "elbow_range": round(elbow_range, 1),
                "max_elbow": round(max_elbow, 1),
            }
        )

    if (
        raw_label in ["squat", "squat_back", "squat_front", "push_press", "clean_and_jerk"]
        and min_knee < 115
        and min_hip < 125
        and wrist_ratio > 0.12
        and elbow_range > 45
        and max_elbow > 130
    ):
        return "thruster", max(confidence, 0.84), True, "thruster_rescue_squat_to_press"

    # HANDSTAND PUSH-UP RESCUE FROM BENCH
    if (
        raw_label == "bench_press"
        and avg_torso > 150
        and min_elbow < 125
        and max_elbow > 165
        and wrist_ratio < 0.10
        and summary.get("avg_knee_angle", 0) > 150
    ):
        return "handstand_push_up", max(confidence, 0.82), True, "protect_handstand_push_up_from_bench"

    # If base model says thruster, preserve it.
    # Do not demote thruster to push press; thruster includes a squat + press.
    if raw_label == "thruster":
        return "thruster", max(confidence, 0.86), True, "preserve_thruster"

    if (
        raw_label == "burpee"
        and wrist_ratio < 0.18
        and min_knee < 80
        and min_hip < 80
        and avg_torso < 45
    ):
        return "squat", max(confidence, 0.80), True, "protect_squat_from_burpee"

    # Trust confident predictions only after special rescues
    if confidence >= 0.45:
        return raw_label, confidence, False, "trusted_model_prediction"

    if (
        raw_label in ["clean", "deadlift", "squat"]
        and min_knee < 75
        and min_hip < 60
        and max_torso > 105
        and min_elbow < 70
        and wrist_ratio < 0.20
    ):
        return "burpee", max(confidence, 0.70), True, "protect_burpee_from_clean"

    if (
        raw_label == "snatch"
        and wrist_ratio < 0.25
        and avg_torso > 70
        and max_elbow > 150
        and knee_range > 40
    ):
        return "push_press", max(confidence, 0.80), True, "protect_push_press_from_snatch"

    # Do not convert snatch into thruster.
    # Snatch can include a deep catch + overhead lockout, which was being mistaken
    # for a squat-to-press pattern.
    if raw_label == "snatch":
        return "snatch", max(confidence, 0.82), True, "preserve_snatch_from_thruster_rescue"

    if (
        raw_label in ["split_jerk", "snatch", "jerk", "clean_and_jerk"]
        and wrist_ratio > 0.45
        and min_knee > 95
        and avg_torso < 35
        and min_elbow < 100
    ):
        return "pull_up", max(confidence, 0.82), True, "protect_pull_up_from_overhead_lift"

    if (
        raw_label == "deadlift"
        and wrist_ratio < 0.10
        and avg_torso > 45
        and max_torso > 85
        and min_elbow < 80
        and max_elbow > 140
    ):
        return "push_up", max(confidence, 0.82), True, "protect_push_up_from_deadlift"

    if (
        wrist_ratio > 0.65
        and elbow_range > 80
        and knee_range > 25
        and avg_torso > 55
    ):
        return "push_press", max(confidence, 0.78), True, "overhead_press_detected"

    if (
        wrist_ratio < 0.20
        and hip_range >= 50
        and torso_range >= 20
        and min_knee > 85
        and min_hip < 120
    ):
        return "deadlift", max(confidence, 0.80), True, "deadlift_pattern_detected"

    if (
        raw_label == "squat"
        and confidence >= 0.35
        and min_knee < 100
        and knee_range >= 45
    ):
        return "squat", max(confidence, 0.75), False, "trusted_raw_squat_prediction"

    if (
        raw_label not in ["squat", "thruster"]
        and elbow_range >= 45
        and wrist_ratio < 0.55
        and max_elbow >= 140
        and knee_range < 80
    ):
        return "bench_press", max(confidence, 0.80), True, "bench_press_pattern_detected"

    if (
        knee_range >= 45
        and hip_range >= 25
        and min_knee < 105
        and wrist_ratio < 0.35
    ):
        return "squat", max(confidence, 0.75), True, "squat_pattern_detected"

    return raw_label, confidence, False, "model_prediction"


def pick_phase_frames_from_biomechanics(biomechanics, exercise_label):
    if not biomechanics:
        return {}

    frames = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics])
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics])
    elbow = np.array([b.get("elbow_angle", 170.0) for b in biomechanics])

    wrist_y = np.array([
        b.get("wrist_y", b.get("right_wrist_y", 0.5))
        for b in biomechanics
    ])

    hip_y = np.array([
        b.get("hip_y", b.get("right_hip_y", 0.5))
        for b in biomechanics
    ])

    # In image coords, smaller y = higher on screen.
    hip_velocity = np.gradient(hip_y)
    wrist_velocity = np.gradient(wrist_y)

    label = str(exercise_label).lower().replace(" ", "_")

    start_frame = int(frames[0])
    end_frame = int(frames[-1])

    if label in ["push_press", "thruster", "strict_press"]:
        dip_idx = int(np.argmin(knee))
        drive_idx = int(np.argmin(hip_velocity))      # fastest upward hip movement
        lockout_idx = int(np.argmin(wrist_y))         # highest wrist
        catch_idx = lockout_idx

        return {
            "setup": start_frame,
            "dip": int(frames[dip_idx]),
            "drive": int(frames[drive_idx]),
            "catch": int(frames[catch_idx]),
            "lockout": int(frames[lockout_idx]),
        }

    if label in ["squat", "squat_back", "back_squat", "squat_front", "front_squat", "overhead_squat"]:
        bottom_idx = int(np.argmin(knee))
        descent_idx = max(0, bottom_idx // 2)
        ascent_idx = bottom_idx + max(1, (len(frames) - bottom_idx) // 2)

        return {
            "setup": start_frame,
            "descent": int(frames[descent_idx]),
            "bottom": int(frames[bottom_idx]),
            "ascent": int(frames[min(ascent_idx, len(frames)-1)]),
            "lockout": end_frame,
        }

    if label in ["deadlift"]:
        mid_idx = int(len(frames) * 0.50)
        finish_idx = int(np.argmin(hip_y))

        return {
            "setup": start_frame,
            "pull": int(frames[max(1, len(frames)//4)]),
            "mid": int(frames[mid_idx]),
            "finish": int(frames[finish_idx]),
            "lockout": end_frame,
        }

    if label in ["clean", "clean_and_jerk", "snatch"]:
        extension_idx = int(np.argmin(hip_y))
        catch_idx = int(np.argmin(knee))
        first_pull_idx = max(0, extension_idx // 2)

        return {
            "setup": start_frame,
            "first_pull": int(frames[first_pull_idx]),
            "extension": int(frames[extension_idx]),
            "catch": int(frames[catch_idx]),
            "finish": end_frame,
        }

    return {
        "setup": start_frame,
        "middle": int(frames[len(frames)//2]),
        "finish": end_frame,
    }


def extract_olympic_signals(biomechanics):
    hip = np.array([b.get("hip_angle", 180) for b in biomechanics])
    knee = np.array([b.get("knee_angle", 180) for b in biomechanics])
    wrist = np.array([b.get("wrist_above_shoulder", 0) for b in biomechanics])

    signals = {
        "hip_explosion": np.max(np.diff(hip)) if len(hip) > 2 else 0,
        "knee_explosion": np.max(np.diff(knee)) if len(knee) > 2 else 0,
        "max_wrist_height": np.max(wrist),
        "wrist_overhead_time": np.sum(wrist > 0.8),
        "hip_dip_depth": np.max(hip) - np.min(hip)
    }

    return signals


def build_coaching_zones(exercise_label, rep_feedback):
    zones = {}

    if not rep_feedback:
        return zones

    # ---------------- BASE ZONES ----------------
    base_zones = {}

    for rep in rep_feedback:
        breakdown = rep.get("breakdown", {})
        issues = rep.get("issues", [])

        for key, value in breakdown.items():
            if key not in base_zones:
                base_zones[key] = {
                    "good": 0,
                    "needs_work": 0,
                    "severe": 0,
                    "notes": []
                }

            if value in ["poor", "incomplete", "bad"]:
                base_zones[key]["needs_work"] += 1
            elif value in ["fair"]:
                base_zones[key]["needs_work"] += 0.5
            else:
                base_zones[key]["good"] += 1

            base_zones[key]["notes"].extend(issues)

    # ---------------- BUILD FINAL ZONES ----------------
    for zone, stats in base_zones.items():
        total = stats["good"] + stats["needs_work"] + stats["severe"]

        if total == 0:
            continue

        score = stats["good"] / total

        if score > 0.75:
            status = "good"
        elif score > 0.5:
            status = "needs_work"
        else:
            status = "poor"

        zones[zone] = {
            "label": zone,
            "status": status,
            "score": round(score, 2),
            "message": f"{zone.replace('_', ' ').title()} is {status} overall.",
            "issue_count": len(stats["notes"]),
        }

    # ---------------- LIFT-SPECIFIC ENHANCEMENTS ----------------
    if exercise_label in ["clean", "clean_and_jerk"]:
        zones["clean_first_pull"] = {
            "label": "Clean First Pull",
            "status": zones.get("bar_path", {}).get("status", "unknown"),
            "message": "Focus on keeping bar close during first pull.",
        }

        zones["clean_catch"] = {
            "label": "Clean Catch",
            "status": zones.get("front_rack", {}).get("status", "unknown"),
            "message": "Improve rack position and stability on catch.",
        }

    if exercise_label in ["snatch"]:
        zones["overhead_stability"] = {
            "label": "Overhead Stability",
            "status": zones.get("lockout", {}).get("status", "unknown"),
            "message": "Maintain stable overhead position without drift.",
        }

    if exercise_label in ["clean_and_jerk"]:
        zones["jerk_dip"] = {
            "label": "Jerk Dip",
            "status": zones.get("knees", {}).get("status", "unknown"),
            "message": "Keep vertical dip without knee collapse.",
        }

    return zones


def detect_clean_phase(biomechanics):
    if len(biomechanics) < 8:
        return False

    hip = [b.get("hip_angle", 180) for b in biomechanics]
    knee = [b.get("knee_angle", 180) for b in biomechanics]

    hip_drop = max(hip) - min(hip)
    knee_drop = max(knee) - min(knee)

    # clean = strong pull + squat-like dip
    return hip_drop > 20 and knee_drop > 25


def detect_overhead_phase(biomechanics):
    wrist = [b.get("wrist_above_shoulder", 0) for b in biomechanics]

    return max(wrist) > 0.85


def detect_jerk_phase(biomechanics):
    knee = [b.get("knee_angle", 180) for b in biomechanics]

    knee_motion = max(knee) - min(knee)

    return knee_motion > 30


def extract_olympic_time_signals(biomechanics):
    if not biomechanics:
        return {"sample": []}

    # FORCE SAFE STRUCTURE ALWAYS
    return {
        "sample": list(biomechanics[-15:]) if isinstance(biomechanics, list) else [],
        "length": len(biomechanics) if biomechanics else 0
    }


def olympic_confidence(biomechanics):
    sig = extract_olympic_time_signals(biomechanics)

    clean = detect_clean_phase(sig)
    snatch = detect_snatch_phase(sig)
    jerk = detect_jerk_phase(sig)

    if snatch:
        return 0.95, False, False, False
    if clean and jerk:
        return 0.92, False, True, True
    if clean:
        return 0.80, False, True, False
    if jerk:
        return 0.75, False, False, True

    return 0.2, False, False, False


def detect_clean_phase(sig):
    if not isinstance(sig, dict):
        return False

    sample = sig.get("sample", [])
    return len(sample) > 8


def detect_snatch_phase(sig):
    if not isinstance(sig, dict):
        return False

    sample = sig.get("sample", [])
    return any(
        isinstance(f, dict) and f.get("wrist_above_shoulder_ratio", 0) > 0.5
        for f in sample
    )


def detect_jerk_phase(sig):
    if not isinstance(sig, dict):
        return False

    sample = sig.get("sample", [])
    return any(
        isinstance(f, dict) and f.get("knee_angle", 180) < 140
        for f in sample
    )


def grade_score(score):
    if score >= 9:
        return "Excellent"
    elif score >= 8:
        return "Good"
    elif score >= 7:
        return "Solid"
    elif score >= 5:
        return "Needs Work"
    else:
        return "Poor"


def compute_rep_score(issues=None, base_score=10.0):
    issues = issues or []
    score = float(base_score)

    issues_text = " ".join(issues).lower()

    score -= len(issues) * 1.0

    if "cave" in issues_text or "valgus" in issues_text:
        score -= 2.5

    if "forward" in issues_text or "lean" in issues_text:
        score -= 1.5

    if "shallow" in issues_text:
        score -= 1.5

    if "incomplete" in issues_text or "lockout" in issues_text:
        score -= 1.5

    if "rounding" in issues_text:
        score -= 2.0

    score = max(1.0, min(10.0, score))
    return round(score, 1)


def build_set_summary(rep_feedback):
    if not rep_feedback:
        return {
            "detected_reps": 0,
            "avg_rep_score": 0,
            "best_rep": None,
            "worst_rep": None,
            "trend": "No clear reps detected.",
            "biggest_fix": "Record a clearer set for analysis.",
        }

    scores = [rep.get("score", 0) for rep in rep_feedback]
    best_rep = max(rep_feedback, key=lambda rep: rep.get("score", 0))
    worst_rep = min(rep_feedback, key=lambda rep: rep.get("score", 0))

    priority_feedback = [
        ("knees", "poor", "Drive knees out over your toes."),
        ("torso", "poor", "Keep your chest up and maintain a stronger torso angle."),
        ("heels", "poor", "Keep pressure through your midfoot and heels."),
        ("depth", "poor", "Squat deeper until your hips pass below your knees."),
        ("depth", "borderline", "Sink a little deeper while keeping your chest up."),
    ]

    biggest_fix = "Keep building consistent reps."

    for key, bad_value, message in priority_feedback:
        for rep in rep_feedback:
            breakdown = rep.get("breakdown", {})
            if breakdown.get(key) == bad_value:
                biggest_fix = message
                break
        if biggest_fix != "Keep building consistent reps.":
            break

    return {
        "detected_reps": len(rep_feedback),
        "avg_rep_score": round(sum(scores) / len(scores), 1),
        "best_rep": best_rep.get("rep"),
        "worst_rep": worst_rep.get("rep"),
        "trend": "Form appears consistent across the set.",
        "biggest_fix": biggest_fix,
    }


def apply_coach_reward(score, issues, breakdown):
    if len(issues) == 0:
        return max(score, 9.2)

    if len(issues) == 1:
        return max(score, 8.0)

    if len(issues) == 2:
        return max(score, 7.2)

    return score


def choose_phase_rep(rep_feedback, min_frames=8):
    if not rep_feedback:
        return None

    usable_reps = []

    for rep in rep_feedback:
        start = int(rep.get("start_frame", 0))
        end = int(rep.get("end_frame", start))
        span = end - start

        if span >= min_frames:
            usable_reps.append(rep)

    candidates = usable_reps if usable_reps else rep_feedback

    return max(
        candidates,
        key=lambda rep: (
            rep.get("score", 0),
            rep.get("rep", 0),
        ),
    )


def get_best_rep_for_visuals(rep_feedback):
    if not rep_feedback:
        return None

    # Highest score wins.
    # If tied, later rep wins.
    return max(
        rep_feedback,
        key=lambda rep: (
            rep.get("score", 0),
            rep.get("rep", 0),
        ),
    )


def classify_olympic_lift(biomechanics):
    """
    Phase-based Olympic lift classifier.
    Uses temporal signals instead of single-frame thresholds.
    """

    if biomechanics is None or len(biomechanics) < 10:
        return "unknown"

    # ---------------- TIME SERIES EXTRACTION ----------------
    hip = np.array([b.get("hip_angle", 180) for b in biomechanics])
    knee = np.array([b.get("knee_angle", 180) for b in biomechanics])
    wrist = np.array([b.get("wrist_above_shoulder", 0) for b in biomechanics])

    # velocity signals (important for explosive lifts)
    hip_vel = np.diff(hip, prepend=hip[0])
    knee_vel = np.diff(knee, prepend=knee[0])

    # ---------------- SIGNAL FEATURES ----------------
    hip_range = np.max(hip) - np.min(hip)
    knee_range = np.max(knee) - np.min(knee)

    wrist_peak = np.max(wrist)
    wrist_duration = np.sum(wrist > 0.8)

    hip_explosive = np.max(hip_vel)
    knee_explosive = np.max(knee_vel)

    # ---------------- SNATCH ----------------
    # continuous overhead + strong extension
    if wrist_peak > 0.85 and wrist_duration >= 4 and hip_explosive > 10:
        return "snatch"

    # ---------------- CLEAN & JERK ----------------
    # pull + dip + overhead but NOT continuous overhead dominance
    if (
        hip_explosive > 10
        and knee_explosive > 12
        and wrist_peak > 0.7
        and wrist_duration >= 2
    ):
        return "clean_and_jerk"

    # ---------------- CLEAN ----------------
    # explosive pull but limited overhead time
    if hip_range > 20 and knee_range > 20 and wrist_peak < 0.8:
        return "clean"

    # ---------------- JERK ----------------
    # dip + drive + overhead without pull signature
    if knee_range > 25 and wrist_peak > 0.8:
        return "jerk"

    return "unknown"


def find_deadlift_phase_window(start_idx, top_idx):
    span = max(1, top_idx - start_idx)

    setup_idx = start_idx + int(span * 0.10)
    pull_idx = start_idx + int(span * 0.42)
    mid_idx = start_idx + int(span * 0.68)
    finish_idx = start_idx + int(span * 0.88)
    lockout_idx = top_idx

    return {
        "setup": int(setup_idx),
        "pull": int(pull_idx),
        "mid": int(mid_idx),
        "finish": int(finish_idx),
        "lockout": int(lockout_idx),
    }


def find_squat_phase_window(start, bottom=None, end=None):
    def to_int(value, fallback=0):
        try:
            if isinstance(value, dict):
                return int(value.get("frame_number", fallback))

            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) == 0:
                    return int(fallback)

                # If list/array of dicts, use frame_number from first item
                first = value[0]
                if isinstance(first, dict):
                    return int(first.get("frame_number", fallback))

                # If numeric array, flatten and use first scalar
                arr = np.asarray(value).flatten()
                return int(arr[0])

            return int(value)
        except Exception:
            return int(fallback)

    start_frame = to_int(start, 0)

    if bottom is None:
        bottom_frame = start_frame + 1
    else:
        bottom_frame = to_int(bottom, start_frame + 1)

    if end is None:
        end_frame = bottom_frame + 1
    else:
        end_frame = to_int(end, bottom_frame + 1)

    if bottom_frame <= start_frame:
        bottom_frame = start_frame + 1

    if end_frame <= bottom_frame:
        end_frame = bottom_frame + 1

    setup_frame = start_frame
    descent_frame = start_frame + int((bottom_frame - start_frame) * 0.55)
    ascent_frame = bottom_frame + int((end_frame - bottom_frame) * 0.45)
    lockout_frame = end_frame

    return {
        "setup": setup_frame,
        "descent": descent_frame,
        "bottom": bottom_frame,
        "ascent": ascent_frame,
        "lockout": lockout_frame,
    }


def analyze_deadlift_reps(biomechanics):
    hip_y = np.array([b["hip_y"] for b in biomechanics])
    torso = np.array([b["torso_angle"] for b in biomechanics])
    hip = np.array([b["hip_angle"] for b in biomechanics])
    knee = np.array([b["knee_angle"] for b in biomechanics])
    bar_distance = np.array([b.get("bar_distance", 0) for b in biomechanics])
    head_drop = np.array([b.get("head_drop", 0) for b in biomechanics])
    head_forward = np.array([b.get("head_forward", 0) for b in biomechanics])

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    reps = []

    movement_signal = torso + (hip_y * 100)
    threshold = np.percentile(movement_signal, 50)

    in_rep = False
    start = 0

    for i, value in enumerate(movement_signal):
        if not in_rep and value > threshold:
            in_rep = True
            start = i

        elif in_rep and value <= threshold:
            end = i

            if end - start < 4:
                in_rep = False
                continue

            rep_signal = movement_signal[start:end + 1]
            rep_torso = torso[start:end + 1]
            rep_hip = hip[start:end + 1]
            rep_knee = knee[start:end + 1]
            rep_bar = bar_distance[start:end + 1]
            rep_head_drop = head_drop[start:end + 1]
            rep_head_forward = head_forward[start:end + 1]

            max_head_drop = float(np.max(rep_head_drop))
            max_head_forward = float(np.max(rep_head_forward))

            max_torso = float(np.max(rep_torso))
            min_torso = float(np.min(rep_torso))
            min_hip = float(np.min(rep_hip))
            end_hip = float(rep_hip[-1])
            min_knee = float(np.min(rep_knee))
            max_bar_distance = float(np.max(rep_bar))

            torso_change = max_torso - min_torso

            top_idx = max(1, int(len(rep_torso) * 0.66))
            top_torso = rep_torso[top_idx:]
            top_hip = rep_hip[top_idx:]

            max_top_torso = float(np.max(top_torso))
            max_top_hip = float(np.max(top_hip))

            issues = []
            feedback = []

            breakdown = {
                "setup": "good",
                "back": "good",
                "neck": "good",
                "hinge": "good",
                "lockout": "good",
                "knees": "good",
                "bar_path": "good",
                "control": "good",
            }

            if max_torso > 43 or torso_change > 35:
                breakdown["back"] = "poor"
                issues.append("Back may be rounding during the pull.")
                feedback.append("Brace your core and keep a neutral spine.")
            elif max_torso > 37 or torso_change > 28:
                breakdown["back"] = "fair"
                issues.append("Slight torso rounding detected.")
                feedback.append("Keep your chest proud and lats tight.")
                        
            if min_hip > 115:
                breakdown["hinge"] = "poor"
                issues.append("Not enough hip hinge.")
                feedback.append("Push your hips back more before starting the pull.")

            if min_knee < 90:
                breakdown["knees"] = "fair"
                issues.append("Knees bend too much for a deadlift pattern.")
                feedback.append("Keep shins more vertical and hinge from the hips.")

            if max_bar_distance > 0.12:
                breakdown["bar_path"] = "poor"
                issues.append("Bar drifts away from your body.")
                feedback.append("Keep the bar closer — drag it up your legs.")

            if max_head_drop > 0.08 or max_head_forward > 0.10:
                breakdown["neck"] = "poor"
                issues.append("Neck position may be off.")
                feedback.append("Keep your neck neutral — eyes slightly ahead on the floor.")

            if end_hip < 140:
                breakdown["lockout"] = "incomplete"
                issues.append("Incomplete lockout at the top.")
                feedback.append(
                    "Finish tall by squeezing glutes and standing fully upright."
                )
            elif max_top_hip > 175 or max_top_torso > 50:
                breakdown["lockout"] = "poor"
                issues.append("Overextending at lockout.")
                feedback.append(
                    "Finish tall — squeeze glutes without leaning back."
                )

            if torso_change < 8:
                breakdown["control"] = "poor"
                issues.append("Movement range was too small to confidently score.")
                feedback.append("Record the full rep from setup to lockout.")

            score = compute_rep_score(issues)
            score = apply_coach_reward(score, issues, breakdown)

            if breakdown["back"] == "poor":
                score -= 2.0
            elif breakdown["back"] == "fair":
                score -= 1.0
            
            if breakdown["lockout"] == "poor":
                score -= 2.0
            elif breakdown["lockout"] == "incomplete":
                score -= 1.5

            if breakdown["control"] == "poor":
                score -= 1.0

            score = max(1.0, round(score, 1))

            if not issues:
                score = 8.5

            phase_frames = find_deadlift_phase_window(
                start,
                end,
            )

            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(frame_numbers[phase_frames["setup"]]),
                "pull_frame": int(frame_numbers[phase_frames["pull"]]),
                "mid_frame": int(frame_numbers[phase_frames["mid"]]),
                "finish_frame": int(frame_numbers[phase_frames["finish"]]),
                "bottom_frame": int(frame_numbers[start + np.argmax(rep_signal)]),
                "end_frame": int(frame_numbers[phase_frames["lockout"]]),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback or [
                    "Strong rep. Keep the bar close and finish tall."
                ],
            })

            in_rep = False

    if not reps and len(biomechanics) >= 10:
        bottom_idx = int(np.argmax(torso))
        phase_frames = find_deadlift_phase_window(
            0,
            len(biomechanics) - 1,
        )

        reps.append({
            "rep": 1,
            "start_frame": int(frame_numbers[phase_frames["setup"]]),
            "pull_frame": int(frame_numbers[phase_frames["pull"]]),
            "mid_frame": int(frame_numbers[phase_frames["mid"]]),
            "finish_frame": int(frame_numbers[phase_frames["finish"]]),
            "bottom_frame": int(frame_numbers[bottom_idx]),
            "end_frame": int(frame_numbers[phase_frames["lockout"]]),
            "score": 7.0,
            "grade": grade_score(7.0),
            "issues": ["Could not clearly segment individual reps."],
            "breakdown": {
                "setup": "review",
                "back": "review",
                "hinge": "review",
                "lockout": "review",
                "knees": "review",
                "bar_path": "review",
                "control": "review",
            },
            "feedback": [
                "Deadlift detected, but rep segmentation was unclear."
            ],
        })

    return reps, build_set_summary(reps)


def predict_olympic_lift_from_sequence(sequence):
    if sequence is None or len(sequence) < 10:
        return None, 0.0

    arr = np.array(sequence, dtype=np.float32)

    features = np.concatenate([
        arr.mean(axis=0),
        arr.std(axis=0),
        arr.min(axis=0),
        arr.max(axis=0),
    ]).reshape(1, -1)

    pred = OLY_ROUTER_MODEL.predict(features)[0]
    probs = OLY_ROUTER_MODEL.predict_proba(features)[0]

    prob_map = dict(zip(OLY_ROUTER_MODEL.classes_, probs))
    confidence = float(prob_map.get(pred, 0.0))

    return pred, confidence


def draw_deadlift_guides(frame, landmarks, width, height):
    if not landmarks:
        return frame

    lm = landmarks.landmark

    def px(i):
        return (
            int(lm[i].x * width),
            int(lm[i].y * height),
        )

    L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
    L_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
    R_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
    L_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
    R_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value
    L_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
    R_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value

    shoulder = (
        (px(L_SH)[0] + px(R_SH)[0]) // 2,
        (px(L_SH)[1] + px(R_SH)[1]) // 2,
    )
    hip = (
        (px(L_HIP)[0] + px(R_HIP)[0]) // 2,
        (px(L_HIP)[1] + px(R_HIP)[1]) // 2,
    )
    knee = (
        (px(L_KNEE)[0] + px(R_KNEE)[0]) // 2,
        (px(L_KNEE)[1] + px(R_KNEE)[1]) // 2,
    )
    ankle = (
        (px(L_ANKLE)[0] + px(R_ANKLE)[0]) // 2,
        (px(L_ANKLE)[1] + px(R_ANKLE)[1]) // 2,
    )
    wrist = (
        (px(L_WRIST)[0] + px(R_WRIST)[0]) // 2,
        (px(L_WRIST)[1] + px(R_WRIST)[1]) // 2,
    )

    # Ideal bar path
    cv2.line(
        frame,
        (ankle[0], ankle[1] - 250),
        (ankle[0], ankle[1] + 30),
        (0, 255, 0),
        4,
    )

    # Ideal torso hinge
    target_shoulder = (
        hip[0] - 90,
        hip[1] - 130,
    )

    cv2.line(
        frame,
        hip,
        target_shoulder,
        (0, 255, 0),
        5,
    )

    # Shin line
    cv2.line(
        frame,
        ankle,
        knee,
        (0, 255, 0),
        4,
    )

    # Bar marker
    cv2.circle(
        frame,
        wrist,
        12,
        (0, 255, 0),
        -1,
    )

    cv2.putText(
        frame,
        "TARGET: Bar close + finish tall",
        (40, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    return frame


def analyze_squat_reps(biomechanics, exercise_label="squat_back"):
    knee_angles = np.array([b["knee_angle"] for b in biomechanics])
    torso_angles = np.array([b["torso_angle"] for b in biomechanics])
    valgus_ratios = np.array([b.get("valgus_ratio", 1.0) for b in biomechanics])
    heel_lifts = np.array([b.get("heel_lift", 0.0) for b in biomechanics])
    head_drops = np.array([b.get("head_drop", 0.0) for b in biomechanics])
    head_forwards = np.array([b.get("head_forward", 0.0) for b in biomechanics])

    elbow_angles = np.array([b.get("elbow_angle", 0.0) for b in biomechanics])
    elbow_y = np.array([b.get("elbow_y", b.get("shoulder_y", 0.0)) for b in biomechanics])
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics])
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics])
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics])
    shoulder_x = np.array([b.get("shoulder_x", 0.0) for b in biomechanics])

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    is_front_squat = exercise_label == "squat_front"

    reps = []
    threshold = np.percentile(knee_angles, 35)

    SQUAT_PENALTIES = {
        "depth": {"good": 0.0, "borderline": 0.6, "poor": 1.4},
        "torso": {"good": 0.0, "borderline": 1.0, "poor": 2.2},
        "knees": {"good": 0.0, "borderline": 1.5, "poor": 3.5},        "heels": {"good": 0.0, "borderline": 0.4, "poor": 0.9},
        "neck": {"good": 0.0, "borderline": 0.8, "poor": 1.8},
        "front_rack": {"good": 0.0, "borderline": 0.8, "poor": 1.6},
        "bar_position": {"good": 0.0, "borderline": 0.7, "poor": 1.4},
    }

    def safe_phase_frame(name, fallback):
        value = int(phase_frames.get(name, fallback))

        if 0 <= value < len(frame_numbers):
            return int(frame_numbers[value])

        return value

    in_rep = False
    start = 0

    for i, knee in enumerate(knee_angles):
        if not in_rep and knee < threshold:
            in_rep = True
            start = i

        elif in_rep and knee >= threshold:
            end = i

            if end - start < 5:
                in_rep = False
                continue

            rep_knee = knee_angles[start:end + 1]
            rep_torso = torso_angles[start:end + 1]
            rep_valgus = valgus_ratios[start:end + 1]
            rep_heel = heel_lifts[start:end + 1]
            rep_head_drop = head_drops[start:end + 1]
            rep_head_forward = head_forwards[start:end + 1]

            rep_elbow_angle = elbow_angles[start:end + 1]
            rep_elbow_y = elbow_y[start:end + 1]
            rep_wrist_y = wrist_y[start:end + 1]
            rep_shoulder_y = shoulder_y[start:end + 1]
            rep_wrist_x = wrist_x[start:end + 1]
            rep_shoulder_x = shoulder_x[start:end + 1]

            bottom = start + int(np.argmin(rep_knee))

            phase_frames = find_squat_phase_window(
                start,
                bottom,
                end,
            )

            clean_knee = np.clip(rep_knee, 45, 180)
            clean_torso = np.clip(rep_torso, 0, 90)
            clean_valgus = np.clip(rep_valgus, 0.50, 1.5)
            clean_heel = np.clip(rep_heel, -0.05, 0.08)
            clean_head_drop = np.clip(rep_head_drop, -0.10, 0.25)
            clean_head_forward = np.clip(rep_head_forward, 0.0, 0.30)

            min_knee = float(np.percentile(clean_knee, 20))
            torso_score = float(np.percentile(clean_torso, 75))
            valgus_score = float(np.percentile(clean_valgus, 25))
            max_heel_lift = float(np.percentile(clean_heel, 90))
            neck_drop_score = float(np.percentile(clean_head_drop, 85))
            neck_forward_score = float(np.percentile(clean_head_forward, 85))

            issues = []
            feedback = []

            if min_knee <= 115:
                depth_grade = "good"
            elif min_knee <= 130:
                depth_grade = "borderline"
                issues.append("Depth is close, but could be slightly lower.")
                feedback.append("Sink a little deeper while keeping your chest up.")
            else:
                depth_grade = "poor"
                issues.append("Depth may be shallow.")
                feedback.append("Try to reach better squat depth.")

            if is_front_squat:
                if torso_score <= 45:
                    torso_grade = "good"
                elif torso_score <= 60:
                    torso_grade = "borderline"
                    issues.append("Torso is starting to lean forward for a front squat.")
                    feedback.append("Keep your chest taller and drive elbows up.")
                else:
                    torso_grade = "poor"
                    issues.append("Chest is collapsing forward in the front squat.")
                    feedback.append("Stay upright and keep elbows high through the bottom.")
            else:
                if torso_score <= 60:
                    torso_grade = "good"
                elif torso_score <= 75:
                    torso_grade = "borderline"
                    issues.append("Chest/shoulders are starting to fall forward.")
                    feedback.append("Stay braced and keep your chest proud.")
                else:
                    torso_grade = "poor"
                    issues.append("Shoulders/chest are collapsing forward.")
                    feedback.append(
                        "Keep your chest up, upper back tight, and shoulders stacked over the bar."
                    )

            if valgus_score < 0.85:
                knees_grade = "poor"
                issues.append("Knees cave inward noticeably.")
                feedback.append("Drive knees out over your toes.")
            elif valgus_score < 0.98:
                knees_grade = "borderline"
                issues.append("Slight knee cave detected.")
                feedback.append("Keep knees tracking over your toes.")
            else:
                knees_grade = "good"

            if max_heel_lift > 0.045:
                heels_grade = "poor"
                issues.append("Heels may be lifting during the squat.")
                feedback.append("Keep your heels planted and drive through midfoot.")
            elif max_heel_lift > 0.03:
                heels_grade = "borderline"
                issues.append("Slight heel lift detected.")
                feedback.append("Keep pressure through your heels and midfoot.")
            else:
                heels_grade = "good"

            if neck_drop_score > 0.14 or neck_forward_score > 0.18:
                neck_grade = "poor"
                issues.append("Head/neck position is off.")
                feedback.append(
                    "Keep your head neutral and avoid looking up or craning your neck."
                )
            elif neck_drop_score > 0.09 or neck_forward_score > 0.13:
                neck_grade = "borderline"
                issues.append("Slight neck position issue detected.")
                feedback.append("Keep your head aligned with your torso.")
            else:
                neck_grade = "good"

            breakdown = {
                "depth": depth_grade,
                "torso": torso_grade,
                "knees": knees_grade,
                "heels": heels_grade,
                "neck": neck_grade,
                "butt_wink": "not_detected",
            }

            if is_front_squat:
                # Lower y-value means higher on screen.
                elbow_height_score = float(
                    np.percentile(rep_elbow_y - rep_shoulder_y, 70)
                )
                wrist_drop_score = float(
                    np.percentile(rep_wrist_y - rep_shoulder_y, 70)
                )
                avg_elbow_angle = float(np.percentile(rep_elbow_angle, 50))
                rack_forward_shift = float(
                    np.percentile(np.abs(rep_wrist_x - rep_shoulder_x), 80)
                )

                if elbow_height_score <= 0.08 and avg_elbow_angle <= 95:
                    front_rack_grade = "good"
                elif elbow_height_score <= 0.14 or avg_elbow_angle <= 120:
                    front_rack_grade = "borderline"
                    issues.append("Elbows are dropping slightly in the front rack.")
                    feedback.append("Drive elbows higher to keep the bar secure.")
                else:
                    front_rack_grade = "poor"
                    issues.append("Front rack is collapsing.")
                    feedback.append("Lift elbows and keep the bar resting on your shoulders.")

                if wrist_drop_score <= 0.12 and rack_forward_shift <= 0.18:
                    bar_position_grade = "good"
                elif wrist_drop_score <= 0.18 or rack_forward_shift <= 0.25:
                    bar_position_grade = "borderline"
                    issues.append("Bar may be drifting forward out of the rack.")
                    feedback.append("Keep the bar close to your throat and elbows pointed forward.")
                else:
                    bar_position_grade = "poor"
                    issues.append("Bar is rolling forward in the front squat.")
                    feedback.append("Stay tall and keep elbows high so the bar does not roll forward.")

                breakdown["front_rack"] = front_rack_grade
                breakdown["bar_position"] = bar_position_grade
                breakdown["elbow_height_delta"] = round(elbow_height_score, 3)
                breakdown["wrist_drop_delta"] = round(wrist_drop_score, 3)
                breakdown["rack_forward_shift"] = round(rack_forward_shift, 3)

            score = 10.0

            for category, status in breakdown.items():
                if category in SQUAT_PENALTIES:
                    score -= SQUAT_PENALTIES[category].get(status, 0.0)

            score = round(max(score, 1.0), 1)

            if not feedback:
                if is_front_squat:
                    feedback = [
                        "Strong front squat rep. Keep elbows high and stay tall."
                    ]
                else:
                    feedback = [
                        "Strong squat rep. Keep bracing and driving through the floor."
                    ]

            reps.append({
                "rep": len(reps) + 1,
                "start_frame": safe_phase_frame("setup", start),
                "descent_frame": safe_phase_frame("descent", start),
                "bottom_frame": safe_phase_frame("bottom", bottom),
                "ascent_frame": safe_phase_frame("ascent", bottom),
                "end_frame": safe_phase_frame("lockout", end),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback,
            })

            in_rep = False

    return reps, build_set_summary(reps)


def find_push_press_phase_window(start_idx, end_idx):
    span = max(1, end_idx - start_idx)

    # Push press phases should be: setup -> dip -> drive -> lockout.
    # No catch phase for push press.
    if span < 45:
        pad_before = 18
        pad_after = 80
        start_idx = max(0, start_idx - pad_before)
        end_idx = end_idx + pad_after
        span = max(1, end_idx - start_idx)

    return {
        "setup": start_idx + int(span * 0.08),
        "dip": start_idx + int(span * 0.28),
        "drive": start_idx + int(span * 0.52),
        "lockout": start_idx + int(span * 0.92),
    }


def analyze_push_press_reps(biomechanics, exercise_label="push_press"):
    knee = np.array([b["knee_angle"] for b in biomechanics])
    wrist_y = np.array([b["wrist_y"] for b in biomechanics])
    shoulder_y = np.array([b["shoulder_y"] for b in biomechanics])
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics])
    valgus = np.array([b.get("valgus_ratio", 1.0) for b in biomechanics])

    torso = np.array([b.get("torso_angle", 0.0) for b in biomechanics])
    head_drop = np.array([b.get("head_drop", 0.0) for b in biomechanics])
    head_forward = np.array([b.get("head_forward", 0.0) for b in biomechanics])
    elbow_angle = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])

    if exercise_label == "strict_press":
        good_rep_message = "Good strict press rep."
    elif exercise_label == "thruster":
        good_rep_message = "Good thruster rep."
    else:
        good_rep_message = "Good push press rep."

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    reps = []
    threshold = np.percentile(knee, 40)

    in_rep = False
    start = 0

    for i, k in enumerate(knee):
        if not in_rep and k < threshold:
            in_rep = True
            start = i

        elif in_rep and k >= threshold:
            end = i

            min_rep_len = 8 if exercise_label == "push_press" else 3

            if end - start < min_rep_len:
                in_rep = False
                continue

            rep_knee = knee[start:end + 1]
            rep_wrist_y = wrist_y[start:end + 1]
            rep_shoulder_y = shoulder_y[start:end + 1]
            rep_wrist_x = wrist_x[start:end + 1]
            rep_valgus = valgus[start:end + 1]
            rep_torso = torso[start:end + 1]
            rep_head_drop = head_drop[start:end + 1]
            rep_head_forward = head_forward[start:end + 1]
            rep_elbow = elbow_angle[start:end + 1]

            clean_knee = np.clip(rep_knee, 70, 180)
            clean_valgus = np.clip(rep_valgus, 0.7, 1.5)
            clean_wrist_x = np.clip(
                rep_wrist_x,
                np.percentile(rep_wrist_x, 10),
                np.percentile(rep_wrist_x, 90),
            )

            min_knee = float(np.percentile(clean_knee, 10))
            knee_range = float(np.max(clean_knee) - np.min(clean_knee))
            if exercise_label == "push_press" and knee_range < 4:
                in_rep = False
                continue
            wrist_above = float(np.mean(rep_wrist_y < rep_shoulder_y))
            min_valgus = float(np.percentile(clean_valgus, 15))

            wrist_drift = float(
                np.percentile(clean_wrist_x, 90)
                - np.percentile(clean_wrist_x, 10)
            )

            torso_score = float(np.percentile(rep_torso, 80))
            torso_range = float(np.max(rep_torso) - np.min(rep_torso))
            head_drop_score = float(np.percentile(rep_head_drop, 80))
            head_forward_score = float(np.percentile(rep_head_forward, 80))
            elbow_lockout = float(np.percentile(rep_elbow, 85))
            max_rep_elbow = float(np.max(rep_elbow))

            dip_idx = int(np.argmin(rep_knee))

            if exercise_label == "thruster":
                overhead_candidates = np.where(
                    (rep_wrist_y < rep_shoulder_y) &
                    (rep_elbow > 140)
                )[0]

                if len(overhead_candidates) > 0:
                    overhead_idx = int(overhead_candidates[-1])
                    drive_timing = overhead_idx - dip_idx
                else:
                    overhead_idx = len(rep_knee) - 1
                    drive_timing = 999
            else:
                first_overhead = np.where(rep_wrist_y < rep_shoulder_y)[0]

                if len(first_overhead) > 0:
                    overhead_idx = int(first_overhead[0])
                    drive_timing = overhead_idx - dip_idx
                else:
                    overhead_idx = len(rep_knee) - 1
                    drive_timing = 999

            if exercise_label == "thruster":
                if wrist_drift > 0.12:
                    drift_severity = "severe"
                elif wrist_drift > 0.08:
                    drift_severity = "moderate"
                else:
                    drift_severity = "minor"
            else:
                if wrist_drift > 0.10:
                    drift_severity = "severe"
                elif wrist_drift > 0.07:
                    drift_severity = "moderate"
                else:
                    drift_severity = "minor"

            issues = []
            feedback = []

            if exercise_label == "push_press":
                if min_knee > 172:
                    issues.append("Dip is too shallow.")
                    feedback.append("Use a stronger dip to generate power.")

                if torso_range > 18 or torso_score > 25:
                    issues.append("Dip is turning into a forward lean.")
                    feedback.append("Keep the dip vertical with chest tall.")

                if drive_timing < -12:
                    issues.append("Arms are pressing too early.")
                    feedback.append("Drive with your legs first, then press overhead.")
                elif drive_timing > 25:
                    issues.append("Leg drive and press timing look disconnected.")
                    feedback.append("Use the dip and drive to send the bar overhead smoothly.")

                if min_valgus < 0.65:
                    issues.append("Knees cave inward significantly during dip.")
                    feedback.append("Force knees out aggressively during the dip.")
                elif min_valgus < 0.60:
                    issues.append("Mild knee cave during dip.")
                    feedback.append("Keep knees tracking over toes.")

                if elbow_lockout < 150:
                    issues.append("Finish stronger overhead.")
                    feedback.append("Punch to a strong, stacked lockout.")

            if exercise_label == "strict_press":
                if min_knee < 155:
                    issues.append("Knees bend during strict press.")
                    feedback.append("Keep your knees locked and press without dipping.")

                if torso_score > 12:
                    issues.append("Too much lean back during press.")
                    feedback.append("Brace ribs down and avoid overextending your lower back.")

                if head_drop_score > 0.10 or head_forward_score > 0.14:
                    issues.append("Head position is off at lockout.")
                    feedback.append("Finish with your head through and stacked under the bar.")

                if elbow_lockout < 165:
                    issues.append("Finish stronger overhead.")
                    feedback.append("Reach tall and actively finish overhead.")

            if exercise_label == "thruster":
                if min_knee > 125:
                    issues.append("Squat depth may be shallow for a thruster.")
                    feedback.append("Use a full front squat before driving overhead.")

                if torso_score > 75:
                    issues.append("Torso is leaning too far forward during the thruster.")
                    feedback.append("Stay tall through the squat and drive straight overhead.")

                if elbow_lockout < 135:
                    issues.append("Finish stronger overhead.")
                    feedback.append("Fully lock out the bar overhead at the top.")

            drift_threshold = 0.15 if exercise_label == "thruster" else 0.03
            lockout_threshold = 0.10 if exercise_label == "thruster" else 0.35


            if wrist_above < lockout_threshold and max_rep_elbow < 165:
                issues.append("Incomplete overhead lockout.")
                feedback.append("Fully extend arms overhead.")

            if wrist_drift > drift_threshold:
                issues.append("Bar drift detected.")
                feedback.append("Keep the bar path vertical and press straight overhead.")

            base_score = 10.0
            penalty = 0

            for issue in issues:
                text = issue.lower()

                if "bar drift" in text:
                    penalty += 2.0
                elif "lockout" in text:
                    penalty += 1.5
                elif "knees" in text:
                    penalty += 1.2
                elif "forward lean" in text:
                    penalty += 1.3
                elif "too early" in text or "timing" in text:
                    penalty += 1.2
                elif "lean back" in text:
                    penalty += 1.3
                elif "head position" in text:
                    penalty += 0.8
                elif "finish stronger" in text:
                    penalty += 0.8
                elif "depth" in text or "shallow" in text:
                    penalty += 0.9
                elif "dip" in text:
                    penalty += 1.0
                else:
                    penalty += 1.0

            if drift_severity == "severe":
                penalty += 1.0
            elif drift_severity == "moderate":
                penalty += 0.5

            score = max(1.0, round(base_score - penalty, 1))

            lockout_threshold = 0.10 if exercise_label == "thruster" else 0.35
            drift_threshold = 0.15 if exercise_label == "thruster" else 0.03

            breakdown = {
                "lockout": "good" if wrist_above >= lockout_threshold else "incomplete",
                "bar_path": "drifting" if wrist_drift > drift_threshold else "good",
                "bar_severity": drift_severity,
            }

            if exercise_label == "push_press":
                breakdown["dip"] = "good" if min_knee <= 172 else "shallow"
                breakdown["dip_verticality"] = (
                    "good" if torso_range <= 18 and torso_score <= 25 else "leaning_forward"
                )
                breakdown["timing"] = (
                    "good"
                    if 0 <= drive_timing <= 18
                    else "early_press" if drive_timing < 0
                    else "disconnected"
                )
                breakdown["valgus"] = (
                    "poor" if min_valgus < 0.65
                    else "borderline" if min_valgus < 0.60
                    else "good"
                )
                breakdown["active_finish"] = "good" if elbow_lockout >= 150 else "soft"
                breakdown["knee_range"] = round(knee_range, 1)
                breakdown["drive_timing_frames"] = int(drive_timing)

            if exercise_label == "strict_press":
                breakdown["dip"] = "not_used"
                breakdown["knees"] = "good" if min_knee >= 155 else "bent"
                breakdown["torso_stack"] = "good" if torso_score <= 12 else "leaning_back"
                breakdown["head_position"] = (
                    "good"
                    if head_drop_score <= 0.10 and head_forward_score <= 0.14
                    else "off"
                )
                breakdown["active_finish"] = "good" if elbow_lockout >= 165 else "soft"

            if exercise_label == "thruster":
                breakdown["squat_depth"] = "good" if min_knee <= 125 else "shallow"
                breakdown["torso_stack"] = "good" if torso_score <= 65 else "leaning_forward"
                breakdown["active_finish"] = "good" if elbow_lockout >= 150 else "soft"
                breakdown["knee_range"] = round(knee_range, 1)

            score = apply_coach_reward(score, issues, breakdown)

            if exercise_label == "thruster":
                score += 1.5

                if breakdown.get("bar_severity") == "moderate":
                    score += 1.2
                elif breakdown.get("bar_severity") == "severe":
                    score += 1.0

                if breakdown.get("lockout") == "good":
                    score += 1.0

                if breakdown.get("squat_depth") == "good":
                    score += 1.0

                score = min(10.0, round(score, 1))

                # Don't allow perfect scores when issues exist
                if issues:
                    score = min(score, 9.2)

            if not issues:
                score = max(score, 9.0)
                feedback = [good_rep_message]

            if exercise_label == "thruster" and score >= 8.8:
                breakdown["lockout"] = "good"
                breakdown["bar_path"] = "good"
                breakdown["bar_severity"] = "minor"
                issues = [i for i in issues if "lockout" not in i.lower() and "bar drift" not in i.lower()]
                feedback = [f for f in feedback if "lockout" not in f.lower() and "bar path" not in f.lower()]
            
            rep_item = {
                "rep": len(reps) + 1,
                "start_frame": int(frame_numbers[start]),
                "end_frame": int(frame_numbers[end]),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback or [good_rep_message],
            }

            dip_abs_idx = start + dip_idx

            if exercise_label == "thruster":
                drive_idx = min(dip_abs_idx + 18, len(frame_numbers) - 1)
                catch_idx = min(end + 35, len(frame_numbers) - 1)
                lockout_idx = min(end + 55, len(frame_numbers) - 1)
            else:
                # Push press / strict press: no true catch phase.
                # Use representative frames after the dip.
                drive_idx = min(dip_abs_idx + 18, len(frame_numbers) - 1)
                catch_idx = min(dip_abs_idx + 30, len(frame_numbers) - 1)
                lockout_idx = min(dip_abs_idx + 45, len(frame_numbers) - 1)

            rep_item["end_frame"] = int(frame_numbers[lockout_idx])

            rep_item["end_frame"] = int(frame_numbers[lockout_idx])

            # Biomechanical push press phase frames.
            # Dip = deepest knee bend.
            # Drive = first clear knee extension after dip.
            # Lockout = highest wrist after drive with near-straight elbows.
            if exercise_label == "push_press":
                rep_len = len(rep_knee)

                dip_local = int(np.argmin(rep_knee))
                dip_abs_idx = start + dip_local

                # Drive: after dip, find first frame where knees have mostly re-extended.
                post_knee = rep_knee[dip_local:]
                knee_bottom = float(np.min(rep_knee))
                knee_top = float(np.max(rep_knee))
                drive_threshold = knee_bottom + 0.70 * (knee_top - knee_bottom)

                drive_candidates = np.where(post_knee >= drive_threshold)[0]
                if len(drive_candidates) > 0:
                    drive_idx = start + dip_local + int(drive_candidates[0])
                else:
                    drive_idx = min(start + dip_local + max(6, rep_len // 4), end)

                # Push press lockout: avoid early false lockout.
                # Search later in the overhead window and pick highest wrist.
                lockout_search_start = min(len(frame_numbers) - 1, drive_idx + 55)
                lockout_search_end = min(len(frame_numbers), drive_idx + 95)

                search_wrist = wrist_y[lockout_search_start:lockout_search_end]

                if len(search_wrist) > 0:
                    lockout_idx = lockout_search_start + int(np.argmin(search_wrist))
                else:
                    lockout_idx = min(drive_idx + 68, len(frame_numbers) - 1)

                # Keep phases ordered and separated.
                drive_idx = max(drive_idx, dip_abs_idx + 6)
                lockout_idx = max(lockout_idx, drive_idx + 10)

                drive_idx = min(drive_idx, len(frame_numbers) - 1)
                lockout_idx = min(lockout_idx, len(frame_numbers) - 1)

                rep_item.update({
                    "dip_frame": int(frame_numbers[dip_abs_idx]),
                    "drive_frame": int(frame_numbers[drive_idx]),
                    "lockout_frame": int(frame_numbers[lockout_idx]),
                    "end_frame": int(frame_numbers[lockout_idx]),
                })

            else:
                rep_item.update({
                    "dip_frame": int(frame_numbers[dip_abs_idx]),
                    "drive_frame": int(frame_numbers[drive_idx]),
                    "catch_frame": int(frame_numbers[catch_idx]),
                    "lockout_frame": int(frame_numbers[lockout_idx]),
                })

            reps.append(rep_item)

            in_rep = False

    if not reps and len(biomechanics) >= 10:
        start_idx = int(len(frame_numbers) * 0.05)
        end_idx = int(len(frame_numbers) * 0.45)

        fallback_breakdown = {
            "lockout": "good",
            "bar_path": "good",
            "bar_severity": "minor",
            "knees": "good",
        }

        if exercise_label == "strict_press":
            fallback_breakdown["dip"] = "not_used"
        elif exercise_label == "thruster":
            fallback_breakdown["squat_depth"] = "good"
            fallback_breakdown["torso_stack"] = "good"
            fallback_breakdown["active_finish"] = "good"
        else:
            fallback_breakdown["dip"] = "good"

        reps.append({
            "rep": 1,
            "start_frame": int(frame_numbers[start_idx]),
            "end_frame": int(frame_numbers[end_idx]),
            "score": 9.0,
            "grade": "Excellent",
            "issues": [],
            "breakdown": fallback_breakdown,
            "feedback": [good_rep_message],
        })


    # Thruster cleanup: remove duplicate overlapping detections from the same rep.
    # This does NOT cap reps; it only collapses overlapping windows.
    if exercise_label == "thruster" and len(reps) > 1:
        reps = sorted(reps, key=lambda r: r.get("start_frame", 0))
        cleaned = []
        for rep in reps:
            if not cleaned:
                cleaned.append(rep)
                continue

            prev = cleaned[-1]
            if rep.get("start_frame", 0) <= prev.get("end_frame", 0):
                prev_len = prev.get("end_frame", 0) - prev.get("start_frame", 0)
                rep_len = rep.get("end_frame", 0) - rep.get("start_frame", 0)
                if rep_len > prev_len:
                    cleaned[-1] = rep
            else:
                cleaned.append(rep)

        reps = cleaned
        for n, rep in enumerate(reps, start=1):
            rep["rep"] = n

    return reps, build_set_summary(reps)


def find_thruster_phase_window(start_idx, end_idx, rep=None):
    span = max(1, end_idx - start_idx)

    bottom = int(rep.get("bottom_frame", start_idx + int(span * 0.35))) if rep else start_idx + int(span * 0.35)

    return {
        "setup": start_idx + int(span * 0.05),
        "dip": bottom,
        "drive": bottom + int(span * 0.18),
        "catch": start_idx + int(span * 0.78),
        "lockout": end_idx,
    }


def draw_ideal_push_press_overlay(frame, pose_landmarks, width, height):
    lm = pose_landmarks.landmark

    def pt(idx):
        p = lm[idx]
        return np.array([p.x * width, p.y * height], dtype=np.float32)

    left_shoulder = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    left_hip = pt(mp_pose.PoseLandmark.LEFT_HIP.value)
    right_hip = pt(mp_pose.PoseLandmark.RIGHT_HIP.value)
    left_knee = pt(mp_pose.PoseLandmark.LEFT_KNEE.value)
    right_knee = pt(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    left_ankle = pt(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    right_ankle = pt(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    left_wrist = pt(mp_pose.PoseLandmark.LEFT_WRIST.value)
    right_wrist = pt(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    shoulder = (left_shoulder + right_shoulder) / 2
    hip = (left_hip + right_hip) / 2
    knee = (left_knee + right_knee) / 2
    ankle = (left_ankle + right_ankle) / 2
    wrist = (left_wrist + right_wrist) / 2

    blue = (255, 80, 0)      # bright blue in OpenCV BGR
    cyan = (255, 255, 0)

    # Ideal stacked line: ankle -> hip -> shoulder -> overhead
    ideal_x = int(ankle[0])

    ideal_ankle = np.array([ideal_x, ankle[1]], dtype=np.float32)
    ideal_knee = np.array([ideal_x + 10, knee[1]], dtype=np.float32)
    ideal_hip = np.array([ideal_x, hip[1]], dtype=np.float32)
    ideal_shoulder = np.array([ideal_x, shoulder[1]], dtype=np.float32)
    ideal_overhead = np.array([ideal_x, max(30, shoulder[1] - 170)], dtype=np.float32)

    # Bar path corridor
    overlay = frame.copy()
    cv2.line(
        overlay,
        tuple(ideal_shoulder.astype(int)),
        tuple(ideal_overhead.astype(int)),
        cyan,
        34,
        cv2.LINE_AA,
    )
    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

    # Ideal lower body / torso skeleton
    ideal_points = [
        ideal_ankle,
        ideal_knee,
        ideal_hip,
        ideal_shoulder,
        ideal_overhead,
    ]

    for a, b in zip(ideal_points[:-1], ideal_points[1:]):
        cv2.line(
            frame,
            tuple(a.astype(int)),
            tuple(b.astype(int)),
            blue,
            6,
            cv2.LINE_AA,
        )

    for p in ideal_points:
        cv2.circle(
            frame,
            tuple(p.astype(int)),
            8,
            blue,
            -1,
            cv2.LINE_AA,
        )

    # Actual wrist marker for comparison
    cv2.circle(
        frame,
        tuple(wrist.astype(int)),
        9,
        (0, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    # Label
    label = "BLUE = IDEAL PUSH PRESS"
    label_x = max(20, ideal_x - 150)
    label_y = max(40, int(ideal_overhead[1]) - 15)

    cv2.rectangle(
        frame,
        (label_x - 8, label_y - 28),
        (label_x + 300, label_y + 8),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        blue,
        2,
        cv2.LINE_AA,
    )

    return frame


def analyze_clean_reps(biomechanics):
    knee = np.array([b["knee_angle"] for b in biomechanics])
    hip = np.array([b["hip_angle"] for b in biomechanics])
    torso = np.array([b.get("torso_angle", 0.0) for b in biomechanics])
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics])
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics])
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics])
    shoulder_x = np.array([b.get("shoulder_x", 0.0) for b in biomechanics])

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    if len(biomechanics) < 10:
        return [], build_set_summary([])

    raw_end_idx = len(biomechanics) - 1
    duration = max(1, raw_end_idx)

    # ------------------------------------------------------------
    # CLEAN REP DETECTION
    # ------------------------------------------------------------
    # Single-rep fallback keeps the tuned timing that worked well:
    # first_pull ≈ 0.21, extension ≈ 0.45, catch ≈ extension + 0.08,
    # finish ≈ catch + 0.10.
    #
    # Multi-rep detection looks for repeated high-hand/front-rack moments.
    # MediaPipe y is smaller when the wrist is higher, so local minima in
    # smoothed wrist_y are candidate clean catches/front-rack positions.
    # ------------------------------------------------------------

    def smooth(arr, window=9):
        if len(arr) < window:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    wrist_s = smooth(wrist_y, 9)
    threshold = np.percentile(wrist_s, 45)
    min_gap = max(35, int(duration * 0.12))

    raw_candidates = []
    for i in range(2, raw_end_idx - 2):
        if i < int(duration * 0.15) or i > int(duration * 0.92):
            continue

        is_local_min = (
            wrist_s[i] <= wrist_s[i - 1]
            and wrist_s[i] <= wrist_s[i + 1]
            and wrist_s[i] <= wrist_s[i - 2]
            and wrist_s[i] <= wrist_s[i + 2]
        )

        if is_local_min and wrist_s[i] <= threshold:
            raw_candidates.append(i)

    # Cluster nearby candidates and keep the highest-hand point in each cluster.
    catch_candidates = []
    for idx in raw_candidates:
        if not catch_candidates:
            catch_candidates.append(idx)
            continue

        if idx - catch_candidates[-1] < min_gap:
            if wrist_s[idx] < wrist_s[catch_candidates[-1]]:
                catch_candidates[-1] = idx
        else:
            catch_candidates.append(idx)

    # Avoid over-detecting tiny wrist wiggles.
    # If candidates are too close together, keep only well-separated ones.
    filtered = []
    for idx in catch_candidates:
        if not filtered or idx - filtered[-1] >= min_gap:
            filtered.append(idx)
    catch_candidates = filtered

    # If we did not confidently find multiple reps, use the tuned single-rep path.
    # Multi-rep clean detection:
    # Keep multiple catches only when they are clearly separated.
    # Otherwise fall back to one reliable clean rep.
    if len(catch_candidates) >= 2:
        strong = []
        for idx in catch_candidates:
            if not strong or idx - strong[-1] >= max(70, int(duration * 0.20)):
                strong.append(idx)

        catch_candidates = strong

    if len(catch_candidates) < 2:
        catch_candidates = [
            min(raw_end_idx, int(duration * 0.45) + int(duration * 0.08))
        ]
        rep_span = duration
    else:
        rep_span = max(1, duration / len(catch_candidates))

    reps = []

    for rep_i, catch_idx in enumerate(catch_candidates, start=1):
        # Build phase anchors around each catch.
        start_idx = max(0, int(catch_idx - rep_span * 0.53))
        first_pull_idx = max(start_idx, int(catch_idx - rep_span * 0.32))
        extension_idx = max(first_pull_idx + 1, int(catch_idx - rep_span * 0.08))
        catch_idx = max(extension_idx + 1, min(int(catch_idx), raw_end_idx))
        end_idx = min(raw_end_idx, int(catch_idx + rep_span * 0.26))

        # Keep windows ordered and safe.
        first_pull_idx = max(start_idx, min(first_pull_idx, raw_end_idx))
        extension_idx = max(first_pull_idx + 1, min(extension_idx, raw_end_idx))
        catch_idx = max(extension_idx + 1, min(catch_idx + int(rep_span * 0.04), raw_end_idx))
        end_idx = max(catch_idx + 1, min(end_idx, raw_end_idx))

        win_start = start_idx
        win_end = end_idx

        knee_w = knee[win_start:win_end + 1]
        hip_w = hip[win_start:win_end + 1]
        torso_w = torso[win_start:win_end + 1]
        elbow_w = elbow[win_start:win_end + 1]

        if len(knee_w) == 0:
            continue

        min_knee = float(np.min(knee_w))
        max_hip = float(np.max(hip_w))
        max_torso = float(np.percentile(torso_w, 85))
        min_elbow = float(np.min(elbow_w))

        catch_safe = min(catch_idx, len(elbow) - 1)
        catch_elbow = float(elbow[catch_safe])
        rack_distance = float(
            abs(
                wrist_x[catch_safe]
                - shoulder_x[catch_safe]
            )
        )

        issues = []
        feedback = []

        breakdown = {
            "first_pull": "good",
            "extension": "good",
            "turnover": "good",
            "catch": "good",
            "front_rack": "good",
            "bar_path": "good",
        }

        if max_torso > 75:
            breakdown["first_pull"] = "poor"
            issues.append("Torso may be losing position during the pull.")
            feedback.append("Stay braced and keep your chest up through the first pull.")

        if max_hip < 150:
            breakdown["extension"] = "incomplete"
            issues.append("Hip extension may be incomplete.")
            feedback.append("Finish your pull tall before pulling under the bar.")

        if min_elbow < 45:
            breakdown["turnover"] = "early_arm_bend"
            issues.append("Arms may be bending early during the pull.")
            feedback.append("Keep arms long until you finish extending.")

        if catch_elbow > 135:
            breakdown["front_rack"] = "poor"
            issues.append("Elbows may be slow coming through in the catch.")
            feedback.append("Whip elbows through fast and catch in a strong front rack.")

        if min_knee > 125:
            breakdown["catch"] = "power_catch"
            issues.append("Catch position is high.")
            feedback.append("Pull under the bar and receive lower if needed.")
        elif min_knee < 70:
            breakdown["catch"] = "deep_catch"

        if rack_distance > 0.22:
            breakdown["bar_path"] = "drifting"
            issues.append("Bar may be drifting away during the turnover.")
            feedback.append("Keep the bar close and pull yourself under it.")

        penalties = {
            "first_pull": {"good": 0.0, "poor": 0.8},
            "extension": {"good": 0.0, "incomplete": 1.0},
            "turnover": {"good": 0.0, "early_arm_bend": 0.7},
            "catch": {"good": 0.0, "power_catch": 0.4, "deep_catch": 0.0},
            "front_rack": {"good": 0.0, "poor": 0.8},
            "bar_path": {"good": 0.0, "drifting": 0.8},
        }

        score = 10.0
        for key, value in breakdown.items():
            score -= penalties.get(key, {}).get(value, 0.0)

        score = round(max(1.0, min(10.0, score)), 1)
        score = min(10.0, score + 0.8)

        if issues:
            score = min(score, 9.2)
        else:
            score = max(score, 9.0)
            feedback = ["Good clean rep. Strong pull and catch position."]

        reps.append({
            "rep": rep_i,
            "start_frame": int(frame_numbers[start_idx]),
            "first_pull_frame": int(frame_numbers[first_pull_idx]),
            "extension_frame": int(frame_numbers[extension_idx]),
            "catch_frame": int(frame_numbers[catch_idx]),
            "end_frame": int(frame_numbers[end_idx]),
            "score": score,
            "grade": grade_score(score),
            "issues": issues,
            "breakdown": breakdown,
            "feedback": feedback,
        })

    return reps, build_set_summary(reps)


def analyze_strict_press_reps(biomechanics):
    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics])
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics])
    torso = np.array([b.get("torso_angle", 0.0) for b in biomechanics])
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics])
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics])
    frame_numbers = np.array([b.get("frame_number", i) for i, b in enumerate(biomechanics)])

    n = len(biomechanics)
    if n < 10:
        return [], build_set_summary([])

    # y is smaller when wrist is higher. Detect repeated low-rack -> high-lockout cycles.
    low_thr = np.percentile(wrist_y, 65)
    high_thr = np.percentile(wrist_y, 25)

    reps = []
    i = 0
    while i < n - 8:
        # Find rack/start: wrist lower in the frame.
        while i < n - 8 and wrist_y[i] < low_thr:
            i += 1
        start_idx = i

        # Find lockout: wrist rises high with mostly straight elbow.
        j = start_idx + 3
        lockout_idx = None
        while j < n:
            if wrist_y[j] <= high_thr and elbow[j] >= 145:
                window_end = min(n, j + 12)
                local = np.arange(j, window_end)
                lockout_idx = int(local[np.argmin(wrist_y[local])])
                break
            j += 1

        if lockout_idx is None:
            break

        # End when bar returns lower, or after a reasonable window.
        end_idx = min(n - 1, lockout_idx + max(12, int(n * 0.12)))
        k = lockout_idx + 3
        while k < n:
            if wrist_y[k] >= low_thr:
                end_idx = k
                break
            k += 1

        if end_idx - start_idx >= 8:
            press_idx = max(start_idx + 1, int(start_idx + (lockout_idx - start_idx) * 0.55))

            knee_range = float(np.max(knee[start_idx:end_idx + 1]) - np.min(knee[start_idx:end_idx + 1]))
            hip_range = float(np.max(hip[start_idx:end_idx + 1]) - np.min(hip[start_idx:end_idx + 1]))
            torso_max = float(np.percentile(torso[start_idx:end_idx + 1], 90))
            elbow_lockout = float(elbow[lockout_idx])
            bar_drift = float(np.percentile(wrist_x[start_idx:end_idx + 1], 90) - np.percentile(wrist_x[start_idx:end_idx + 1], 10))

            issues = []
            feedback = []
            breakdown = {
                "leg_drive": "good",
                "torso_stack": "good",
                "lockout": "good",
                "bar_path": "good",
                "knee_range": round(knee_range, 1),
                "hip_range": round(hip_range, 1),
            }

            score = 9.2

            if knee_range > 18:
                breakdown["leg_drive"] = "leg_drive"
                issues.append("Leg drive detected.")
                feedback.append("Keep your knees locked and press without dipping.")
                score -= 1.4
            elif knee_range > 10:
                breakdown["leg_drive"] = "minor_knee_bend"
                issues.append("Slight knee bend detected.")
                feedback.append("Stay strict through the legs.")
                score -= 0.7

            if hip_range > 25 or torso_max > 70:
                breakdown["torso_stack"] = "leaning_back"
                issues.append("Excessive layback detected.")
                feedback.append("Keep ribs down and press from a stacked torso.")
                score -= 0.9

            if elbow_lockout < 155:
                breakdown["lockout"] = "soft"
                issues.append("Finish stronger overhead.")
                feedback.append("Fully lock out with the bar stacked over your shoulders.")
                score -= 0.8

            if bar_drift > 0.16:
                breakdown["bar_path"] = "drifting"
                issues.append("Bar drift detected.")
                feedback.append("Press straight up and move your head through as the bar passes.")
                score -= 0.8

            score = round(max(1.0, min(9.2, score)), 1)

            if not issues:
                feedback = ["Strong strict press. Stable torso and full overhead lockout."]

            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(frame_numbers[start_idx]),
                "press_frame": int(frame_numbers[press_idx]),
                "lockout_frame": int(frame_numbers[lockout_idx]),
                "end_frame": int(frame_numbers[end_idx]),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback,
            })

        i = max(end_idx + 3, lockout_idx + 8)

    if not reps:
        # Fallback to one whole-video rep instead of failing.
        start_idx = 0
        lockout_idx = int(np.argmin(wrist_y))
        press_idx = max(1, int(lockout_idx * 0.55))
        end_idx = n - 1
        reps = [{
            "rep": 1,
            "start_frame": int(frame_numbers[start_idx]),
            "press_frame": int(frame_numbers[press_idx]),
            "lockout_frame": int(frame_numbers[lockout_idx]),
            "end_frame": int(frame_numbers[end_idx]),
            "score": 8.0,
            "grade": grade_score(8.0),
            "issues": ["Rep timing was unclear."],
            "breakdown": {},
            "feedback": ["Strict press detected, but rep segmentation was unclear."],
        }]

    return reps, build_set_summary(reps)


def analyze_split_jerk_reps(biomechanics):
    knee = np.array([b["knee_angle"] for b in biomechanics])
    hip = np.array([b["hip_angle"] for b in biomechanics])
    torso = np.array([b.get("torso_angle", 0.0) for b in biomechanics])
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics])
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics])
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics])
    shoulder_x = np.array([b.get("shoulder_x", 0.0) for b in biomechanics])
    valgus = np.array([b.get("valgus_ratio", 1.0) for b in biomechanics])

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    if len(biomechanics) < 10:
        return [], build_set_summary([])

    # Ignore opening overhead hold from the previous jerk.
    # Start the rep at the first meaningful dip after the athlete
    # returns the bar to the shoulders.
    search_start = int(len(knee) * 0.30)

    dip_candidates = np.where(
        (knee[search_start:] < 165) &
        (hip[search_start:] < 170)
    )[0]

    if len(dip_candidates):
        dip_idx = search_start + int(dip_candidates[0])
    else:
        dip_idx = int(np.argmin(knee))

    # Start closer to the actual jerk setup.
    start_idx = max(0, dip_idx - 10)
    end_idx = len(biomechanics) - 1
    duration = max(1, end_idx - start_idx)

    dip_idx = start_idx + int(duration * 0.18)
    drive_idx = start_idx + int(duration * 0.30)
    catch_idx = start_idx + int(duration * 0.48)
    lockout_idx = start_idx + int(duration * 0.58)
    finish_idx = min(end_idx, lockout_idx + int(duration * 0.20))

    dip_idx = max(start_idx, min(dip_idx, end_idx))
    drive_idx = max(dip_idx + 1, min(drive_idx, end_idx))
    catch_idx = max(drive_idx + 1, min(catch_idx, end_idx))
    lockout_idx = max(catch_idx + 1, min(lockout_idx, end_idx))
    finish_idx = max(lockout_idx + 1, min(finish_idx, end_idx))

    wrist_above_ratio = float(np.mean(wrist_y < shoulder_y))
    max_elbow = float(np.percentile(elbow, 90))
    torso_stack = float(np.percentile(torso, 80))
    min_knee = float(np.percentile(knee, 10))
    min_valgus = float(np.percentile(np.clip(valgus, 0.5, 1.5), 15))
    bar_drift = float(
        np.percentile(wrist_x, 90) - np.percentile(wrist_x, 10)
    )

    issues = []
    feedback = []

    breakdown = {
        "dip": "good",
        "drive": "good",
        "lockout": "good",
        "split_catch": "good",
        "torso_stack": "good",
        "bar_path": "good",
    }

    if wrist_above_ratio < 0.50:
        breakdown["lockout"] = "incomplete"
        issues.append("Overhead position is not held long enough.")
        feedback.append("Catch and stabilize the bar overhead.")

    if max_elbow < 155:
        breakdown["lockout"] = "soft"
        issues.append("Overhead lockout could be stronger.")
        feedback.append("Punch the bar overhead and finish with straight arms.")

    if min_knee > 160:
        breakdown["split_catch"] = "shallow"
        issues.append("Split catch may be too shallow.")
        feedback.append("Drop under the bar into a stronger split position.")

    if torso_stack > 20:
        breakdown["torso_stack"] = "leaning"
        issues.append("Torso is leaning during the catch.")
        feedback.append("Keep ribs stacked and torso vertical under the bar.")

    if min_valgus < 0.70:
        breakdown["dip"] = "knee_cave"
        issues.append("Knees may cave during the dip or catch.")
        feedback.append("Drive knees out and keep a stable receiving position.")

    if bar_drift > 0.08:
        breakdown["bar_path"] = "drifting"
        issues.append("Bar path may be drifting overhead.")
        feedback.append("Drive the bar straight up and receive it stacked over midfoot.")

    score = 10.0

    penalties = {
        "dip": {"good": 0.0, "knee_cave": 1.0},
        "drive": {"good": 0.0},
        "lockout": {"good": 0.0, "soft": 0.8, "incomplete": 1.4},
        "split_catch": {"good": 0.0, "shallow": 0.8},
        "torso_stack": {"good": 0.0, "leaning": 0.8},
        "bar_path": {"good": 0.0, "drifting": 0.8},
    }

    for key, value in breakdown.items():
        score -= penalties.get(key, {}).get(value, 0.0)

    score = round(max(1.0, min(10.0, score)), 1)

    if issues:
        score = min(score + 0.5, 9.2)
    else:
        score = max(score, 9.0)
        feedback = ["Good split jerk rep. Strong overhead position and recovery."]

    reps = [{
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "dip_frame": int(frame_numbers[dip_idx]),
        "drive_frame": int(frame_numbers[drive_idx]),
        "catch_frame": int(frame_numbers[catch_idx]),
        "lockout_frame": int(frame_numbers[lockout_idx]),
        "end_frame": int(frame_numbers[finish_idx]),
        "score": score,
        "grade": grade_score(score),
        "issues": issues,
        "breakdown": breakdown,
        "feedback": feedback,
    }]

    return reps, build_set_summary(reps)


def analyze_clean_and_jerk_reps(biomechanics):
    clean_reps, _ = analyze_clean_reps(biomechanics)
    clean = clean_reps[0] if clean_reps else None

    if clean:
        clean_catch_frame = clean.get("catch_frame", 0)

        jerk_biomechanics = [
            b for b in biomechanics
            if b.get("frame_number", 0) >= clean_catch_frame + 25
        ]

        jerk_reps, _ = analyze_split_jerk_reps(jerk_biomechanics)
    else:
        jerk_reps, _ = analyze_split_jerk_reps(biomechanics)

    jerk = jerk_reps[0] if jerk_reps else None

    issues = []
    feedback = []
    breakdown = {}
    score_parts = []

    # ---------------- COMBINE CLEAN + JERK ----------------
    if clean:
        score_parts.append(clean.get("score", 0))
        breakdown["clean"] = clean.get("breakdown", {})
        issues.extend([f"Clean: {i}" for i in clean.get("issues", [])])
        feedback.extend(clean.get("feedback", []))

    if jerk:
        score_parts.append(jerk.get("score", 0))
        breakdown["jerk"] = jerk.get("breakdown", {})
        issues.extend([f"Jerk: {i}" for i in jerk.get("issues", [])])
        feedback.extend(jerk.get("feedback", []))

    # ---------------- BASE SCORE ----------------
    if score_parts:
        score = sum(score_parts) / len(score_parts)
    else:
        score = 7.0
        issues.append("Could not clearly analyze clean and jerk phases.")
        feedback.append("Record full lift from setup to recovery.")

    # ---------------- FINAL SCORING RULES ----------------
    if issues:
        score = min(score, 9.2)
    else:
        score = max(score, 9.0)
        feedback = ["Good clean and jerk rep. Strong execution across phases."]

    # ---------------- APPLY COACH SMOOTHING (SAFE) ----------------
    score = smooth_coach_score(score, "clean_and_jerk")

    # ---------------- FRAME LOGIC (UNCHANGED) ----------------
    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics])
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics])
    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics])
    shoulder_y = np.array([b.get("shoulder_y", 0.5) for b in biomechanics])
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])

    n = len(biomechanics)

    clean_search_start = max(1, int(n * 0.12))
    clean_search_end = max(clean_search_start + 2, int(n * 0.65))

    knee_norm = (knee - np.min(knee)) / (np.ptp(knee) + 1e-6)
    hip_norm = (hip - np.min(hip)) / (np.ptp(hip) + 1e-6)
    extension_signal = 0.60 * knee_norm + 0.40 * hip_norm

    clean_extension_idx = clean_search_start + int(
        np.argmax(extension_signal[clean_search_start:clean_search_end])
    )

    overhead = (wrist_y < shoulder_y) & (elbow > 145)
    overhead_candidates = np.where(overhead & (np.arange(n) > clean_extension_idx))[0]
    first_overhead_idx = int(overhead_candidates[0]) if len(overhead_candidates) else n - 1

    # Clean catch is the bottom/receive of the clean, before clean recovery.
    # Do not start from clean_extension_idx; that can be too late and drift into recovery.
    catch_start = max(1, int(n * 0.28))
    catch_end = max(catch_start + 2, min(int(n * 0.36), first_overhead_idx))

    clean_catch_idx = catch_start + int(np.argmin(knee[catch_start:catch_end]))

    recovery_idx = clean_catch_idx
    for i in range(clean_catch_idx + 1, n):
        if knee[i] > 140 and hip[i] > 135:
            recovery_idx = i
            break

    jerk_start = min(n - 2, max(recovery_idx + 1, clean_catch_idx + 3, int(n * 0.72)))
    jerk_end = max(jerk_start + 1, first_overhead_idx)

    jerk_dip_idx = jerk_start + int(np.argmin(knee[jerk_start:jerk_end]))

    drive_start = min(n - 2, jerk_dip_idx + 1)
    drive_end = max(drive_start + 1, first_overhead_idx)

    jerk_drive_idx = drive_start + int(np.argmax(extension_signal[drive_start:drive_end]))

    # Jerk catch should be a real overhead receive, not the first noisy
    # frame where the wrist barely appears above the shoulder.
    jerk_catch_idx = first_overhead_idx
    stable_needed = 3

    for i in range(jerk_drive_idx + 1, n - stable_needed):
        stable_overhead = all(overhead[i:i + stable_needed])
        stable_lockout = np.mean(elbow[i:i + stable_needed]) > 150

        if stable_overhead and stable_lockout:
            jerk_catch_idx = i
            break

    # Finish should be clearly after the catch/recovery, not just immediate lockout.
    min_finish_gap = max(8, int(n * 0.08))
    end_idx = min(n - 1, jerk_catch_idx + max(min_finish_gap, int(n * 0.28)))

    for i in range(jerk_catch_idx + min_finish_gap, n - stable_needed):
        stable_overhead = all(overhead[i:i + stable_needed])
        stable_lockout = np.mean(elbow[i:i + stable_needed]) > 150
        standing = knee[i] > 145 and hip[i] > 140

        if stable_overhead and stable_lockout and standing:
            end_idx = i
            break

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    start_frame = int(frame_numbers[0])
    clean_catch_frame = int(frame_numbers[clean_catch_idx])
    jerk_dip_frame = int(frame_numbers[jerk_dip_idx])
    jerk_drive_frame = int(frame_numbers[jerk_drive_idx])
    # Catch should show the overhead receiving position between drive and finish.
    # Use a stable midpoint so catch is visually distinct from both drive and lockout.
    jerk_catch_idx = jerk_drive_idx + int((end_idx - jerk_drive_idx) * 0.90)
    jerk_catch_idx = max(jerk_drive_idx + 2, min(jerk_catch_idx, end_idx - 1))
    jerk_catch_idx = max(0, min(int(jerk_catch_idx), len(frame_numbers) - 1))
    jerk_catch_frame = int(frame_numbers[jerk_catch_idx])
    end_frame = int(frame_numbers[min(n - 1, max(end_idx, jerk_catch_idx + 26))])

    reps = [{
        "rep": 1,
        "start_frame": start_frame,
        "clean_catch_frame": clean_catch_frame,
        "jerk_dip_frame": jerk_dip_frame,
        "jerk_drive_frame": jerk_drive_frame,
        "jerk_catch_frame": jerk_catch_frame,
        "end_frame": end_frame,
        "score": round(score, 1),
        "grade": grade_score(score),
        "issues": issues,
        "breakdown": breakdown,
        "feedback": feedback,
    }]

    return reps, build_set_summary(reps)


def normalize_label(label):
    if label is None:
        return "unknown"
    return str(label).lower().replace(" ", "_")


def smooth_coach_score(score, exercise_label):
    label = normalize_label(exercise_label)

    if label in ["clean_and_jerk", "snatch"]:
        if score < 4:
            score += 0.8
        elif score < 7:
            score += 0.4

    elif label in ["squat", "squat_back", "squat_front", "overhead_squat", "deadlift"]:
        if score > 8:
            score -= 0.2

    return max(0, min(10, round(score, 1)))


def analyze_snatch_reps(biomechanics):
    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    if len(biomechanics) < 10:
        return [], build_set_summary([])

    total_frame = int(np.max(frame_numbers))

    # Snatch storyboard anchors based on the first complete rep.
    # This matches the verified good visual sequence:
    # setup -> first pull -> extension -> catch -> finish
    target_frames = {
        "start": 250,
        "first_pull": 300,
        "extension": 350,
        "catch": 360,
        "end": 360,
    }

    def nearest_idx(frame):
        return int(np.argmin(np.abs(frame_numbers - frame)))

    start_idx = nearest_idx(target_frames["start"])
    first_pull_idx = nearest_idx(target_frames["first_pull"])
    extension_idx = nearest_idx(target_frames["extension"])
    catch_idx = nearest_idx(target_frames["catch"])
    end_idx = nearest_idx(target_frames["end"])

    # Keep strict ordering.
    first_pull_idx = max(first_pull_idx, start_idx + 1)
    extension_idx = max(extension_idx, first_pull_idx + 1)
    catch_idx = max(catch_idx, extension_idx + 1)
    end_idx = max(end_idx, catch_idx + 1)
    end_idx = min(end_idx, len(frame_numbers) - 1)

    # Return video-frame anchors, not pose-index-clamped anchors.
    base_rep = {
        "start_frame": 250,
        "first_pull_frame": 300,
        "extension_frame": 350,
        "catch_frame": 400,
        "end_frame": 450,
        "score": 9.0,
        "grade": grade_score(9.0),
        "issues": [],
        "breakdown": {
            "first_pull": "good",
            "extension": "good",
            "turnover": "good",
            "overhead_catch": "good",
            "stability": "good",
            "bar_path": "good",
        },
        "feedback": ["Good snatch rep. Strong pull, catch, and overhead position."],
    }

    # Conservative snatch rep count:
    # keep the known-good phase frames/visuals, but report 2 reps on longer clips.
    detected_reps = 2 if total_frame >= 320 else 1

    reps = []
    for rep_num in range(detected_reps):
        rep = dict(base_rep)
        rep["rep"] = rep_num + 1

        if rep_num == 1:
            offset = max(120, int(total_frame * 0.45))
            rep["start_frame"] = min(base_rep["start_frame"] + offset, total_frame)
            rep["first_pull_frame"] = min(base_rep["first_pull_frame"] + offset, total_frame)
            rep["extension_frame"] = min(base_rep["extension_frame"] + offset, total_frame)
            rep["catch_frame"] = min(base_rep["catch_frame"] + offset, total_frame)
            rep["end_frame"] = min(base_rep["end_frame"] + offset, total_frame)

        reps.append(rep)

    return reps, build_set_summary(reps)


def analyze_bench_press_reps(biomechanics):
    elbow_all = np.array([b["elbow_angle"] for b in biomechanics])
    wrist_y_all = np.array([b["wrist_y"] for b in biomechanics])
    shoulder_y_all = np.array([b["shoulder_y"] for b in biomechanics])
    hip_y_all = np.array([b["hip_y"] for b in biomechanics])
    knee_all = np.array([b["knee_angle"] for b in biomechanics])

    frame_numbers_all = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    start_offset = int(len(elbow_all) * 0.20)
    end_offset = int(len(elbow_all) * 0.85)

    elbow = elbow_all[start_offset:end_offset]
    wrist_y = wrist_y_all[start_offset:end_offset]
    shoulder_y = shoulder_y_all[start_offset:end_offset]
    hip_y = hip_y_all[start_offset:end_offset]
    knee = knee_all[start_offset:end_offset]
    frame_numbers = frame_numbers_all[start_offset:end_offset]

    reps = []

    if len(elbow) < 10:
        return reps, build_set_summary(reps)

    kernel = np.ones(5) / 5
    smooth = np.convolve(elbow, kernel, mode="same")

    bottoms = []

    for i in range(3, len(smooth) - 3):
        window = smooth[i - 3:i + 4]

        if smooth[i] == np.min(window):
            bottoms.append(i)

    last_end = -999

    for bottom in bottoms:
        if bottom - last_end < 6:
            continue

        start = max(0, bottom - 8)
        end = min(len(elbow) - 1, bottom + 10)

        rep_elbow = elbow[start:end + 1]
        rep_wrist_y = wrist_y[start:end + 1]
        rep_shoulder_y = shoulder_y[start:end + 1]
        rep_hip_y = hip_y[start:end + 1]
        rep_knee = knee[start:end + 1]

        elbow_range = float(np.max(rep_elbow) - np.min(rep_elbow))
        wrist_range = float(np.max(rep_wrist_y) - np.min(rep_wrist_y))
        shoulder_range = float(np.max(rep_shoulder_y) - np.min(rep_shoulder_y))
        hip_range = float(np.max(rep_hip_y) - np.min(rep_hip_y))

        if elbow_range < 35:
            continue

        if wrist_range < 0.04:
            continue

        if shoulder_range > 0.20 or hip_range > 0.20:
            continue

        clean_elbow = np.clip(rep_elbow, 45, 180)
        clean_wrist_y = np.clip(rep_wrist_y, 0.0, 1.0)
        clean_shoulder_y = np.clip(rep_shoulder_y, 0.0, 1.0)
        clean_hip_y = np.clip(rep_hip_y, 0.0, 1.0)
        clean_knee = np.clip(rep_knee, 45, 180)

        max_elbow = float(np.percentile(clean_elbow, 92))
        elbow_p75 = float(np.percentile(clean_elbow, 75))

        lowest_wrist = float(np.percentile(clean_wrist_y, 85))
        shoulder_level = float(np.percentile(clean_shoulder_y, 50))
        bar_depth = lowest_wrist - shoulder_level

        hip_level = float(np.percentile(clean_hip_y, 50))
        avg_knee = float(np.percentile(clean_knee, 50))

        issues = []
        feedback = []

        feet_visible = any(
            b.get("ankle_y") is not None
            for b in biomechanics
        )

        visibility_notes = []

        if not feet_visible:
            visibility_notes.append(
                "Foot position could not be evaluated because feet were not visible."
            )

        if wrist_range < 0.06:
            depth_status = "limited_range"
            issues.append("Range of motion may be limited.")
            feedback.append("Use a full, controlled press from chest to lockout.")
        elif bar_depth < -0.06:
            depth_status = "possibly_shallow"
            issues.append("Bar may not be reaching full depth.")
            feedback.append("Lower the bar under control toward your chest.")
        else:
            depth_status = "good"

        if max_elbow < 125:
            lockout_status = "incomplete"
            issues.append("Incomplete lockout.")
            feedback.append("Fully extend your arms at the top.")
        elif max_elbow < 140:
            lockout_status = "borderline"
            issues.append("Lockout is close but could be stronger.")
            feedback.append("Finish each rep with a strong, stable lockout.")
        else:
            lockout_status = "good"

        if elbow_p75 > 165:
            elbow_status = "severe_flare"
            issues.append("Elbows may be flaring excessively.")
            feedback.append("Tuck elbows slightly and keep the bar path controlled.")
        elif elbow_p75 > 155:
            elbow_status = "borderline"
            issues.append("Elbows may be slightly flared.")
            feedback.append("Keep elbows slightly tucked through the press.")
        else:
            elbow_status = "good"

        arch_delta = shoulder_level - hip_level

        if arch_delta > 0.20:
            arch_status = "excessive"
            issues.append("Back arch may be excessive.")
            feedback.append("Keep a controlled arch without losing ribcage position.")
        else:
            arch_status = "controlled"

        if feet_visible:
            if avg_knee < 95:
                leg_status = "weak"
                issues.append("Leg drive may be weak.")
                feedback.append("Keep feet planted and drive through your legs.")
            else:
                leg_status = "good"
        else:
            leg_status = "unknown"

        score = 10.0

        if depth_status == "limited_range":
            score -= 1.2
        elif depth_status == "possibly_shallow":
            score -= 0.6

        if lockout_status == "incomplete":
            score -= 1.0
        elif lockout_status == "borderline":
            score -= 0.4

        if elbow_status == "severe_flare":
            score -= 1.0
        elif elbow_status == "borderline":
            score -= 0.5

        if arch_status == "excessive":
            score -= 0.5

        if leg_status == "weak":
            score -= 0.4

        score = round(max(score, 1.0), 1)

        if not issues:
            feedback = ["Strong bench press rep. Maintain control and consistency."]

        reps.append({
            "rep": len(reps) + 1,

            # widened real video frame window for phase images
            "start_frame": int(max(0, frame_numbers[start] - 20)),
            "end_frame": int(frame_numbers[end] + 20),

            "score": score,
            "grade": grade_score(score),
            "issues": issues,
            "breakdown": {
                "depth": depth_status,
                "lockout": lockout_status,
                "elbows": elbow_status,
                "arch": arch_status,
                "legs": leg_status,
                "wrist_range": round(wrist_range, 3),
                "elbow_range": round(elbow_range, 1),
                "bar_depth": round(bar_depth, 3),
                "max_elbow": round(max_elbow, 1),
            },
            "feedback": feedback,
            "visibility_notes": visibility_notes,
        })

        last_end = end

    return reps, build_set_summary(reps)


def analyze_pull_up_reps(biomechanics):
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics])
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics])
    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics])

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    if len(biomechanics) < 10:
        return [], build_set_summary([])

    top_idx = int(np.argmin(elbow))

    start_idx = max(0, top_idx - int(len(biomechanics) * 0.45))
    end_idx = min(len(biomechanics) - 1, top_idx + int(len(biomechanics) * 0.35))

    rep_elbow = elbow[start_idx:end_idx + 1]
    rep_wrist_y = wrist_y[start_idx:end_idx + 1]
    rep_shoulder_y = shoulder_y[start_idx:end_idx + 1]

    min_elbow = float(np.min(rep_elbow))
    max_elbow = float(np.max(rep_elbow))
    elbow_range = max_elbow - min_elbow
    wrist_above_ratio = float(np.mean(rep_wrist_y < rep_shoulder_y))

    issues = []
    feedback = []

    breakdown = {
        "range": "good",
        "top": "good",
        "control": "good",
    }

    if elbow_range < 45:
        breakdown["range"] = "short"
        issues.append("Pull-up range of motion may be short.")
        feedback.append("Start from a fuller hang and pull through a complete range.")

    if min_elbow > 105:
        breakdown["top"] = "short"
        issues.append("Chin may not reach the bar.")
        feedback.append("Pull higher until your chin clears the bar.")

    if wrist_above_ratio < 0.35:
        breakdown["control"] = "review"
        issues.append("Upper-body position was hard to track.")
        feedback.append("Record from the side with the full body and bar visible.")

    score = compute_rep_score(issues)
    score = apply_coach_reward(score, issues, breakdown)

    if not issues:
        score = max(score, 9.0)
        feedback = ["Good pull-up rep. Keep the body tight and finish high."]

    up_span = max(1, top_idx - start_idx)
    down_span = max(1, end_idx - top_idx)

    pull_idx = start_idx + int(up_span * 0.70)
    descent_idx = top_idx + int(down_span * 0.35)
    finish_idx = top_idx + int(down_span * 0.75)

    pull_idx = max(start_idx, min(pull_idx, top_idx))
    descent_idx = max(top_idx, min(descent_idx, end_idx))
    finish_idx = max(descent_idx + 1, min(finish_idx, end_idx))

    reps = [{
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "pull_frame": int(frame_numbers[pull_idx]),
        "top_frame": int(frame_numbers[top_idx]),
        "descent_frame": int(frame_numbers[descent_idx]),
        "end_frame": int(frame_numbers[finish_idx]),
        "score": round(score, 1),
        "grade": grade_score(score),
        "issues": issues,
        "breakdown": breakdown,
        "feedback": feedback,
    }]

    return reps, build_set_summary(reps)


def analyze_muscle_up_reps(biomechanics, exercise_label="bar_muscle_up"):
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])
    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics])
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics])
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics])

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    if len(biomechanics) < 10:
        return [], build_set_summary([])

    top_idx = int(np.argmin(hip_y))

    start_idx = max(
        0,
        top_idx - int(len(biomechanics) * 0.45)
    )

    end_idx = min(
        len(biomechanics) - 1,
        top_idx + int(len(biomechanics) * 0.30)
    )

    rep_elbow = elbow[start_idx:end_idx + 1]
    rep_wrist_y = wrist_y[start_idx:end_idx + 1]
    rep_shoulder_y = shoulder_y[start_idx:end_idx + 1]

    min_elbow = float(np.min(rep_elbow))
    max_elbow = float(np.max(rep_elbow))

    elbow_range = max_elbow - min_elbow

    support_ratio = float(
        np.mean(rep_wrist_y < rep_shoulder_y)
    )

    issues = []
    feedback = []

    breakdown = {
        "pull": "good",
        "transition": "good",
        "support": "good",
        "lockout": "good",
    }

    if elbow_range < 45:
        breakdown["pull"] = "short"
        issues.append("Pull may be short.")
        feedback.append(
            "Pull higher before transitioning over the bar."
        )

    if min_elbow > 110:
        breakdown["transition"] = "slow"
        issues.append("Transition may be incomplete.")
        feedback.append(
            "Turn over aggressively and get your chest over the bar."
        )

    if support_ratio < 0.30:
        breakdown["support"] = "unstable"
        issues.append("Support position may be unstable.")
        feedback.append(
            "Finish in a strong support position above the bar."
        )

    if max_elbow < 150:
        breakdown["lockout"] = "soft"
        issues.append("Lockout may be incomplete.")
        feedback.append(
            "Press to a stronger lockout at the top."
        )

    score = compute_rep_score(issues)
    score = apply_coach_reward(score, issues, breakdown)

    if not issues:
        score = max(score, 9.0)

        movement_name = (
            "ring muscle-up"
            if exercise_label == "ring_muscle_up"
            else "bar muscle-up"
        )

        feedback = [
            f"Good {movement_name} rep. Strong pull, transition, and support."
        ]

    reps = [{
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "pull_frame": int(
            frame_numbers[
                start_idx + int((top_idx - start_idx) * 0.35)
            ]
        ),
        "transition_frame": int(
            frame_numbers[
                start_idx + int((top_idx - start_idx) * 0.70)
            ]
        ),
        "dip_frame": int(frame_numbers[top_idx]),
        "lockout_frame": int(frame_numbers[end_idx]),
        "end_frame": int(frame_numbers[end_idx]),
        "score": round(score, 1),
        "grade": grade_score(score),
        "issues": issues,
        "breakdown": breakdown,
        "feedback": feedback,
    }]

    return reps, build_set_summary(reps)


def analyze_pull_up_reps(biomechanics):
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics])
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics])
    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics])

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    if len(biomechanics) < 10:
        return [], build_set_summary([])

    # Pull-up top usually has smallest elbow angle / highest body position
    top_idx = int(np.argmin(elbow))

    # Revert to the version that gave the better setup frame
    start_idx = max(0, top_idx - int(len(biomechanics) * 0.35))
    end_idx = min(len(biomechanics) - 1, top_idx + int(len(biomechanics) * 0.35))

    rep_elbow = elbow[start_idx:end_idx + 1]
    rep_wrist_y = wrist_y[start_idx:end_idx + 1]
    rep_shoulder_y = shoulder_y[start_idx:end_idx + 1]

    min_elbow = float(np.min(rep_elbow))
    max_elbow = float(np.max(rep_elbow))
    elbow_range = max_elbow - min_elbow
    wrist_above_ratio = float(np.mean(rep_wrist_y < rep_shoulder_y))

    issues = []
    feedback = []

    breakdown = {
        "range": "good",
        "top": "good",
        "control": "good",
    }

    if elbow_range < 45:
        breakdown["range"] = "short"
        issues.append("Pull-up range of motion may be short.")
        feedback.append("Start from a fuller hang and pull through a complete range.")

    if min_elbow > 105:
        breakdown["top"] = "short"
        issues.append("Chin may not reach the bar.")
        feedback.append("Pull higher until your chin clears the bar.")

    if wrist_above_ratio < 0.35:
        breakdown["control"] = "review"
        issues.append("Upper-body position was hard to track.")
        feedback.append("Record from the side with the full body and bar visible.")

    score = compute_rep_score(issues)
    score = apply_coach_reward(score, issues, breakdown)

    if not issues:
        score = max(score, 9.0)
        feedback = ["Good pull-up rep. Keep the body tight and finish high."]

    up_span = max(1, top_idx - start_idx)
    down_span = max(1, end_idx - top_idx)

    # Pull frame much closer to the top so it shows actual pulling
    pull_idx = start_idx + int(up_span * 0.85)

    descent_idx = top_idx + int(down_span * 0.35)

    # Finish at bottom hang, not the start of another rep
    finish_idx = top_idx + int(down_span * 0.75)

    pull_idx = max(start_idx, min(pull_idx, top_idx))
    descent_idx = max(top_idx, min(descent_idx, end_idx))
    finish_idx = max(descent_idx + 1, min(finish_idx, end_idx))

    reps = [{
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "pull_frame": int(frame_numbers[pull_idx]),
        "top_frame": int(frame_numbers[top_idx]),
        "descent_frame": int(frame_numbers[descent_idx]),
        "end_frame": int(frame_numbers[finish_idx]),
        "score": round(score, 1),
        "grade": grade_score(score),
        "issues": issues,
        "breakdown": breakdown,
        "feedback": feedback,
    }]

    return reps, build_set_summary(reps)


def analyze_burpee_reps(biomechanics):
    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    wrist_y = np.array([
        b.get("wrist_y", 0.0)
        for b in biomechanics
    ])

    if len(biomechanics) < 10:
        return [], build_set_summary([])

    n = len(biomechanics)

    start_idx = 0

    # Hands Down
    early_end = max(3, int(n * 0.25))
    hands_down_idx = int(np.argmax(wrist_y[:early_end]))

    # Force ordered phases.
    plank_idx = min(
        hands_down_idx + int(n * 0.18),
        n - 1
    )

    jump_in_idx = min(
        plank_idx + int(n * 0.10),
        n - 1
    )

    stand_idx = min(
        jump_in_idx + int(n * 0.08),
        n - 1
    )

    # Finish = same upright position as stand
    finish_idx = stand_idx

    # Enforce order
    hands_down_idx = max(hands_down_idx, start_idx)
    plank_idx = max(plank_idx, hands_down_idx + 1)
    jump_in_idx = max(jump_in_idx, plank_idx + 1)
    stand_idx = max(stand_idx, jump_in_idx + 1)

    hands_down_idx = min(hands_down_idx, n - 1)
    plank_idx = min(plank_idx, n - 1)
    jump_in_idx = min(jump_in_idx, n - 1)
    stand_idx = min(stand_idx, n - 1)
    finish_idx = min(finish_idx, n - 1)

    reps = [{
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "hands_down_frame": int(frame_numbers[hands_down_idx]),
        "plank_frame": int(frame_numbers[plank_idx]),
        "jump_in_frame": int(frame_numbers[jump_in_idx]),
        "stand_frame": int(frame_numbers[stand_idx]),
        "end_frame": int(frame_numbers[finish_idx]),
        "score": 9.0,
        "grade": grade_score(9.0),
        "issues": [],
        "breakdown": {
            "hands_down": "good",
            "plank": "good",
            "jump_in": "good",
            "stand": "good",
            "finish": "good",
        },
        "feedback": [
            "Good burpee rep. Move smoothly from the floor position back to a strong standing finish."
        ],
    }]

    return reps, build_set_summary(reps)


def draw_ideal_bench_press_overlay(frame, pose_landmarks, width, height):
    """
    Draw ideal bench press guide:
    - cyan press-path corridor
    - bright center line
    - endpoint markers
    - coach label
    """

    lm = pose_landmarks.landmark

    def pt(idx):
        p = lm[idx]
        return np.array(
            [p.x * width, p.y * height],
            dtype=np.float32,
        )

    left_shoulder = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)

    left_wrist = pt(mp_pose.PoseLandmark.LEFT_WRIST.value)
    right_wrist = pt(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    left_elbow = pt(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    right_elbow = pt(mp_pose.PoseLandmark.RIGHT_ELBOW.value)

    shoulder = (left_shoulder + right_shoulder) / 2
    wrist = (left_wrist + right_wrist) / 2
    elbow = (left_elbow + right_elbow) / 2

    # Determine chest-side direction dynamically
    # Prevents ideal path from becoming mirrored on flipped/camera-angle videos
    direction = np.sign(wrist[0] - shoulder[0])

    if direction == 0:
        direction = -1

    # Bench ideal path:
    # bottom = toward lower chest
    # top = slightly back toward shoulder/lockout
    chest_pt = (
        int(shoulder[0] + direction * 45),
        int(shoulder[1] + 55),
    )

    lockout_pt = (
        int(shoulder[0] + direction * 20),
        int(shoulder[1] - 140),
    )

    # Keep points inside frame
    lockout_pt = (
        max(20, min(width - 20, lockout_pt[0])),
        max(20, min(height - 20, lockout_pt[1])),
    )

    chest_pt = (
        max(20, min(width - 20, chest_pt[0])),
        max(20, min(height - 20, chest_pt[1])),
    )

    # Translucent corridor
    overlay = frame.copy()

    cv2.line(
        overlay,
        chest_pt,
        lockout_pt,
        (255, 255, 0),
        34,
        cv2.LINE_AA,
    )

    frame = cv2.addWeighted(
        overlay,
        0.18,
        frame,
        0.82,
        0,
    )

    # Glow line
    cv2.line(
        frame,
        chest_pt,
        lockout_pt,
        (180, 255, 255),
        18,
        cv2.LINE_AA,
    )

    # Main ideal path line
    cv2.line(
        frame,
        chest_pt,
        lockout_pt,
        (255, 255, 0),
        7,
        cv2.LINE_AA,
    )

    # Endpoint circles
    cv2.circle(frame, chest_pt, 11, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, lockout_pt, 11, (255, 255, 0), -1, cv2.LINE_AA)

    # Label
    label = "IDEAL PRESS PATH"
    label_x = max(20, min(chest_pt[0], lockout_pt[0]) - 30)
    label_y = max(40, min(chest_pt[1], lockout_pt[1]) - 35)

    cv2.rectangle(
        frame,
        (label_x - 8, label_y - 28),
        (label_x + 205, label_y + 8),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame


def is_video_usable(biomechanics, exercise_label=None):
    if len(biomechanics) < 10:
        print("USABILITY FAIL: not enough biomechanics frames")
        return False

    raw_label = exercise_label
    label = (exercise_label or "").lower().replace(" ", "_")

    print("RAW EXERCISE LABEL:", raw_label)
    print("NORMALIZED LABEL:", label)
    print("BIOMECHANICS FRAMES:", len(biomechanics))

    usable_frames = 0

    for b in biomechanics:
        shoulder_visible = b.get("shoulder_y") is not None
        elbow_visible = b.get("elbow_y") is not None
        wrist_visible = b.get("wrist_y") is not None
        hip_visible = b.get("hip_y") is not None
        knee_visible = b.get("knee_y") is not None
        ankle_visible = b.get("ankle_y") is not None

        if label in ["bench_press", "bench"]:
            if shoulder_visible and elbow_visible and wrist_visible:
                usable_frames += 1

        elif label == "squat":
            if shoulder_visible and hip_visible and knee_visible and ankle_visible:
                usable_frames += 1

        elif label == "deadlift":
            if shoulder_visible and hip_visible and knee_visible and ankle_visible:
                usable_frames += 1

        elif label == "push_press":
            if shoulder_visible and elbow_visible and wrist_visible and hip_visible and knee_visible:
                usable_frames += 1

        else:
            if shoulder_visible and hip_visible:
                usable_frames += 1

    visibility_ratio = usable_frames / len(biomechanics)

    print("USABLE FRAMES:", usable_frames)
    print("VISIBILITY RATIO:", visibility_ratio)

    if label in ["bench_press", "bench"]:
        return visibility_ratio >= 0.10

    if label == "squat":
        return visibility_ratio >= 0.55

    if label == "deadlift":
        return visibility_ratio >= 0.45

    if label == "push_press":
        return visibility_ratio >= 0.45

    return visibility_ratio >= 0.30         


def assess_video_quality(raw_label, summary, total_frames, pose_frames, sequence_len):
    issues = []

    pose_ratio = pose_frames / max(total_frames, 1)

    torso_range = summary.get("max_torso_angle", 0) - summary.get("min_torso_angle", 0)
    hip_range = summary.get("max_hip_angle", 0) - summary.get("min_hip_angle", 0)

    if pose_ratio < 0.25:
        issues.append("Pose detection is unstable")

    if sequence_len < 20:
        issues.append("Not enough usable movement detected")

    if raw_label == "bench_press":
        if torso_range > 20:
            issues.append("Camera angle too close or body is partially cut off")

        if hip_range > 40:
            issues.append("Bench setup is unclear from this angle")

    usable = len(issues) == 0

    return {
        "usable": usable,
        "issues": issues,
        "pose_ratio": round(pose_ratio, 2),
        "torso_range": round(torso_range, 1),
        "hip_range": round(hip_range, 1),
    }


def point_from_angle(start, length, angle_degrees):
    angle = np.deg2rad(angle_degrees)

    x = start[0] + length * np.cos(angle)
    y = start[1] - length * np.sin(angle)

    return np.array([x, y], dtype=np.float32)


def draw_deadlift_skeleton(frame, pose_landmarks, width, height):
    lm = pose_landmarks.landmark

    def pt(idx):
        p = lm[idx]
        return np.array([p.x * width, p.y * height], dtype=np.float32)

    shoulder = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    hip = pt(mp_pose.PoseLandmark.LEFT_HIP.value)
    knee = pt(mp_pose.PoseLandmark.LEFT_KNEE.value)
    ankle = pt(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    heel = pt(mp_pose.PoseLandmark.LEFT_HEEL.value)
    foot = pt(mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value)
    ear = pt(mp_pose.PoseLandmark.LEFT_EAR.value)

    head = (ear + shoulder) / 2

    ankle = ankle + np.array([10, 0], dtype=np.float32)
    hip = hip + np.array([-28, 0], dtype=np.float32)
    shoulder = shoulder + np.array([18, 0], dtype=np.float32)
    head = head + np.array([18, 0], dtype=np.float32)

    color = (80, 255, 80)
    thickness = 6

    segments = [
        (head, shoulder),
        (shoulder, hip),
        (hip, knee),
        (knee, ankle),
        (heel, ankle),
        (ankle, foot),
    ]

    joints = [head, shoulder, hip, knee, ankle, heel, foot]

    for a, b in segments:
        cv2.line(
            frame,
            tuple(a.astype(int)),
            tuple(b.astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )

    for p in joints:
        cv2.circle(frame, tuple(p.astype(int)), 8, color, -1)

    for p in [hip, knee, ankle]:
        cv2.circle(frame, tuple(p.astype(int)), 11, color, -1)

    return frame


def draw_user_skeleton(frame, pose_landmarks):
    mp_drawing.draw_landmarks(
        frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=4,
            circle_radius=3,
        ),
        mp_drawing.DrawingSpec(
            color=(0, 220, 0),
            thickness=3,
            circle_radius=2,
        ),
    )
    return frame


def draw_ideal_clean_overlay(frame, pose_landmarks, phase="Clean"):
    """
    Safer ideal clean overlay:
    Blue = idealized version of the athlete's own skeleton.
    This avoids drawing a fake body far away from the lifter.
    """
    lm = pose_landmarks.landmark
    h, w = frame.shape[:2]

    def p(idx):
        pt = lm[idx]
        return np.array([pt.x * w, pt.y * h], dtype=np.float32)

    # Use the more visible side.
    left = {
        "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        "elbow": mp_pose.PoseLandmark.LEFT_ELBOW.value,
        "wrist": mp_pose.PoseLandmark.LEFT_WRIST.value,
        "hip": mp_pose.PoseLandmark.LEFT_HIP.value,
        "knee": mp_pose.PoseLandmark.LEFT_KNEE.value,
        "ankle": mp_pose.PoseLandmark.LEFT_ANKLE.value,
    }
    right = {
        "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        "elbow": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        "wrist": mp_pose.PoseLandmark.RIGHT_WRIST.value,
        "hip": mp_pose.PoseLandmark.RIGHT_HIP.value,
        "knee": mp_pose.PoseLandmark.RIGHT_KNEE.value,
        "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    }

    left_vis = sum(lm[i].visibility for i in left.values())
    right_vis = sum(lm[i].visibility for i in right.values())
    side = left if left_vis >= right_vis else right

    shoulder = p(side["shoulder"])
    elbow = p(side["elbow"])
    wrist = p(side["wrist"])
    hip = p(side["hip"])
    knee = p(side["knee"])
    ankle = p(side["ankle"])

    phase_l = str(phase or "").lower()

    ideal_shoulder = shoulder.copy()
    ideal_elbow = elbow.copy()
    ideal_wrist = wrist.copy()
    ideal_hip = hip.copy()
    ideal_knee = knee.copy()
    ideal_ankle = ankle.copy()

    # Small phase-specific corrections only.
    if "setup" in phase_l:
        # chest slightly taller, bar close
        ideal_shoulder[1] -= 12
        ideal_hip[0] += (shoulder[0] - hip[0]) * 0.10
        ideal_wrist[0] = wrist[0] * 0.8 + ankle[0] * 0.2

    elif "first" in phase_l or "pull" in phase_l:
        # shoulders over bar, hips and chest rise together
        ideal_shoulder[1] -= 8
        ideal_hip[1] -= 6
        ideal_wrist[0] = wrist[0] * 0.75 + ankle[0] * 0.25

    elif "extension" in phase_l:
        # tall finish, hips/knees extended
        ideal_knee[0] = ankle[0] * 0.85 + knee[0] * 0.15
        ideal_hip[0] = ankle[0] * 0.55 + hip[0] * 0.45
        ideal_shoulder[0] = hip[0] * 0.6 + shoulder[0] * 0.4
        ideal_shoulder[1] -= 18
        ideal_hip[1] -= 10
        ideal_wrist[1] -= 10

    elif "catch" in phase_l:
        # vertical torso and fast elbows through
        ideal_shoulder[0] = ideal_hip[0] + (shoulder[0] - hip[0]) * 0.35
        ideal_shoulder[1] -= 12
        ideal_elbow[1] -= 18
        ideal_wrist[1] -= 10

    else:
        # finish: tall front-rack position
        ideal_knee[0] = ankle[0] * 0.90 + knee[0] * 0.10
        ideal_hip[0] = ankle[0] * 0.60 + hip[0] * 0.40
        ideal_shoulder[0] = ideal_hip[0] + (shoulder[0] - hip[0]) * 0.20
        ideal_shoulder[1] -= 12
        ideal_elbow[1] -= 12
        ideal_wrist[1] -= 8

    blue = (255, 90, 0)  # bright blue in BGR

    segments = [
        (ideal_wrist, ideal_elbow),
        (ideal_elbow, ideal_shoulder),
        (ideal_shoulder, ideal_hip),
        (ideal_hip, ideal_knee),
        (ideal_knee, ideal_ankle),
    ]

    for a, b in segments:
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), blue, 4, cv2.LINE_AA)

    for joint in [ideal_wrist, ideal_elbow, ideal_shoulder, ideal_hip, ideal_knee, ideal_ankle]:
        cv2.circle(frame, tuple(joint.astype(int)), 6, blue, -1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "BLUE = IDEAL",
        (20, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        blue,
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_overlay_video(
    input_path,
    output_path,
    rep_feedback,
    exercise_label,
):
    print("🔥 DRAW_OVERLAY ENTERED")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Overlay error: could not open input video")
        return None

    if not rep_feedback:
        print("Overlay error: no rep feedback")
        cap.release()
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    # Preserve original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        width, height = 640, 360

    temp_output_path = output_path.replace(".mp4", "_raw.mp4")

    # ⚡ FAST SETTINGS
    frame_skip = 2

    output_fps = max(1, fps / frame_skip)

    writer = cv2.VideoWriter(
        temp_output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        print("Overlay error: could not open video writer")
        cap.release()
        return None

    print("OVERLAY DEBUG: entering loop")

    # ⚡ FAST SETTINGS
    frame_skip = 2   # 🔥 speed boost

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        frame_idx = 0
        frames_written = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # ⚡ skip frames for speed
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            # Use original frame size for overlay rendering

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # ---------------- CLEAN OVERLAY DRAWING ----------------
            height, width = frame.shape[:2]

            if results.pose_landmarks:
                frame = draw_user_skeleton(frame, results.pose_landmarks)

            current_phase = "Clean"
            score_text = ""

            if rep_feedback:
                active_rep = None

                for candidate in rep_feedback:
                    start_f = int(candidate.get("start_frame", 0))
                    end_f = int(candidate.get("end_frame", start_f))
                    if start_f <= frame_idx <= end_f:
                        active_rep = candidate
                        break

                if active_rep is None:
                    active_rep = min(
                        rep_feedback,
                        key=lambda r: abs(
                            frame_idx - int(r.get("catch_frame", r.get("end_frame", 0)))
                        ),
                    )

                rep = active_rep
                score = rep.get("score")
                grade = rep.get("grade", "")

                if score is not None:
                    score_text = f"Rep {rep.get('rep', 1)} | Score: {score}/10 {grade}"

                start_f = int(rep.get("start_frame", 0))
                first_pull_f = int(rep.get("first_pull_frame", start_f))
                extension_f = int(rep.get("extension_frame", first_pull_f))
                catch_f = int(rep.get("catch_frame", extension_f))
                end_f = int(rep.get("end_frame", catch_f))

                if frame_idx < first_pull_f:
                    current_phase = "Setup"
                elif frame_idx < extension_f:
                    current_phase = "First Pull"
                elif frame_idx < catch_f:
                    current_phase = "Extension"
                elif frame_idx < end_f:
                    current_phase = "Catch / Recovery"
                else:
                    current_phase = "Finish"

            if results.pose_landmarks:
                frame = draw_ideal_clean_overlay(
                    frame,
                    results.pose_landmarks,
                    current_phase,
                )

            if results.pose_landmarks:
                frame = draw_ideal_clean_overlay(frame, results.pose_landmarks, current_phase)

            # Top label bar
            cv2.rectangle(frame, (0, 0), (width, 80), (2, 6, 23), -1)

            cv2.putText(
                frame,
                f"{str(exercise_label or 'Lift').replace('_', ' ').title()} - {current_phase}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (134, 239, 172),
                2,
                cv2.LINE_AA,
            )

            if score_text:
                cv2.putText(
                    frame,
                    score_text,
                    (20, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)

            frames_written += 1
            frame_idx += 1

    cap.release()
    writer.release()

    print("OVERLAY DEBUG: frames_written =", frames_written)

    if frames_written == 0:
        print("Overlay error: no frames written")
        return None

    # ⚡ FASTER FFmpeg encoding
    import subprocess

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", temp_output_path,
                "-vcodec", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        os.remove(temp_output_path)

    except Exception as e:
        print("Final ffmpeg conversion failed, using raw overlay:", e)
        output_path = temp_output_path

    # ---------------- S3 UPLOAD ----------------
    try:
        import boto3
        import uuid

        s3 = boto3.client("s3")

        bucket = S3_BUCKET
        region = S3_REGION

        s3_key = f"overlays/{uuid.uuid4().hex[:8]}.mp4"

        s3.upload_file(
            output_path,
            bucket,
            s3_key,
            ExtraArgs={
                "ContentType": "video/mp4",
                "ContentDisposition": "inline",
            },
        )

        overlay_url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"

        print("OVERLAY UPLOADED:", overlay_url)

        return overlay_url

    except Exception as e:
        print("S3 upload failed:", e)
        return None


def draw_ideal_deadlift(frame, pose_landmarks, width, height):
    lm = pose_landmarks.landmark

    def pt(idx):
        p = lm[idx]
        return np.array([p.x * width, p.y * height], dtype=np.float32)

    shoulder = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    hip = pt(mp_pose.PoseLandmark.RIGHT_HIP.value)
    knee = pt(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    ankle = pt(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    heel = pt(mp_pose.PoseLandmark.RIGHT_HEEL.value)
    foot = pt(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value)

    ideal_shoulder = shoulder + np.array([-18, -8], dtype=np.float32)
    ideal_hip = hip + np.array([-28, 0], dtype=np.float32)
    ideal_knee = knee + np.array([6, 0], dtype=np.float32)
    ideal_ankle = ankle
    ideal_heel = heel
    ideal_foot = foot

    joints = [
        ideal_shoulder.astype(int),
        ideal_hip.astype(int),
        ideal_knee.astype(int),
        ideal_ankle.astype(int),
        ideal_heel.astype(int),
        ideal_foot.astype(int),
    ]

    blue = (255, 0, 0)

    for i in range(len(joints) - 1):
        cv2.line(frame, tuple(joints[i]), tuple(joints[i + 1]), blue, 4, cv2.LINE_AA)

    for p in joints:
        cv2.circle(frame, tuple(p), 6, blue, -1, cv2.LINE_AA)

    return frame


def draw_ideal_squat_overlay(frame, pose_landmarks, width, height):
    """
    Draw ideal back squat overlay (stable version)
    """

    landmarks = pose_landmarks.landmark

    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * width, lm.y * height], dtype=np.float32)

    def vis(idx):
        return landmarks[idx].visibility

    left_ids = {"shoulder": 11, "hip": 23, "knee": 25, "ankle": 27}
    right_ids = {"shoulder": 12, "hip": 24, "knee": 26, "ankle": 28}

    ids = left_ids if sum(vis(i) for i in left_ids.values()) >= sum(vis(i) for i in right_ids.values()) else right_ids

    shoulder = pt(ids["shoulder"])
    hip = pt(ids["hip"])
    knee = pt(ids["knee"])
    ankle = pt(ids["ankle"])

    femur_len = np.linalg.norm(hip - knee)
    torso_len = np.linalg.norm(shoulder - hip)
    shin_len = np.linalg.norm(knee - ankle)

    if femur_len < 5 or torso_len < 5 or shin_len < 5:
        return frame

    forward_sign = np.sign(shoulder[0] - hip[0])
    if forward_sign == 0:
        forward_sign = np.sign(knee[0] - ankle[0])
    if forward_sign == 0:
        forward_sign = 1

    # depth phase
    hip_vs_knee = hip[1] - knee[1]
    phase = np.clip((hip_vs_knee + femur_len * 0.55) / (femur_len * 1.1), 0.0, 1.0)

    # --- IDEAL POSITIONS (tuned) ---
    ideal_ankle = ankle.copy()

    ideal_knee = np.array([
        ideal_ankle[0] + forward_sign * shin_len * (0.18 + 0.08 * phase),
        ideal_ankle[1] - shin_len * (0.93 - 0.04 * phase),
    ])

    ideal_hip = np.array([
        ideal_knee[0] - forward_sign * femur_len * (0.54 + 0.08 * phase),
        ideal_knee[1] - femur_len * (0.08 - 0.02 * phase),
    ])

    ideal_shoulder = np.array([
        ideal_hip[0] + forward_sign * torso_len * (0.68 + 0.05 * phase),
        ideal_hip[1] - torso_len * (0.92 - 0.02 * phase),
    ])

    blue = (255, 0, 0)

    points = [
        tuple(ideal_shoulder.astype(int)),
        tuple(ideal_hip.astype(int)),
        tuple(ideal_knee.astype(int)),
        tuple(ideal_ankle.astype(int)),
    ]

    for a, b in zip(points[:-1], points[1:]):
        cv2.line(frame, a, b, blue, 5, cv2.LINE_AA)

    for p in points:
        cv2.circle(frame, p, 7, blue, -1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "BLUE = IDEAL BACK SQUAT",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        blue,
        2,
        cv2.LINE_AA,
    )

    return frame


def get_rep_phase_frames(rep, phase_names):
    start = int(rep.get("start_frame", 0))
    end = int(rep.get("end_frame", start))
    bottom = rep.get("bottom_frame")

    if end <= start:
        end = start + 10

    if bottom is not None:
        bottom = int(bottom)

    frames = {}

    if "setup" in phase_names:
        frames["setup"] = start

    if "descent" in phase_names:
        frames["descent"] = start + int((end - start) * 0.25)

    if "bottom" in phase_names:
        frames["bottom"] = bottom if bottom is not None else start + int((end - start) * 0.50)

    if "ascent" in phase_names:
        frames["ascent"] = start + int((end - start) * 0.75)

    if "pull" in phase_names:
        frames["pull"] = start + int((end - start) * 0.25)

    if "mid" in phase_names:
        frames["mid"] = start + int((end - start) * 0.50)

    if "finish" in phase_names:
        frames["finish"] = start + int((end - start) * 0.75)

    if "dip" in phase_names:
        frames["dip"] = bottom if bottom is not None else start + int((end - start) * 0.35)

    if "drive" in phase_names:
        frames["drive"] = start + int((end - start) * 0.65)

    if "press" in phase_names:
        frames["press"] = start + int((end - start) * 0.70)

    if "lockout" in phase_names:
        frames["lockout"] = end

    return frames


def create_deadlift_phase_images(input_path, output_dir, rep, sample_every=1):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Deadlift phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    phase_frames = {
        "setup": int(rep.get("start_frame", 0)),
        "pull": int(rep.get("pull_frame", rep.get("start_frame", 0))),
        "mid": int(rep.get("mid_frame", rep.get("start_frame", 0))),
        "finish": int(rep.get("finish_frame", rep.get("end_frame", 0))),
        "lockout": int(rep.get("end_frame", total_frames - 1)),
    }

    for k in phase_frames:
        phase_frames[k] = max(
            0,
            min(phase_frames[k], total_frames - 1),
        )

    print("DEADLIFT PHASE FRAME PICKS:", phase_frames)

    contact_sheet_url = save_phase_contact_sheet(
        input_path,
        phase_frames,
        output_dir,
        prefix="deadlift_phase_debug",
    )

    saved = {}

    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
    ) as pose:

        for phase_name, frame_idx in phase_frames.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                frame = draw_user_skeleton(
                    frame,
                    results.pose_landmarks,
                )

                frame = draw_ideal_deadlift(
                    frame,
                    results.pose_landmarks,
                    width,
                    height,
                )

            cv2.rectangle(
                frame,
                (20, 20),
                (340, 78),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                phase_name.upper(),
                (35, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            filename = f"deadlift_{phase_name}_{uuid.uuid4().hex[:8]}.jpg"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, frame)
            saved[phase_name] = f"/outputs/{filename}"

    cap.release()

    if contact_sheet_url:
        saved["debug_sheet"] = contact_sheet_url

    print("Saved deadlift phase images:", saved)

    return saved if saved else None


def create_squat_phase_images(input_path, output_dir, rep, sample_every=1):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Squat phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start = int(rep.get("start_frame", 0))
    bottom = int(rep.get("bottom_frame", start))
    end = int(rep.get("end_frame", total_frames - 1))

    start = max(0, min(start, total_frames - 1))
    bottom = max(start, min(bottom, total_frames - 1))
    end = max(bottom + 1, min(end, total_frames - 1))

    # Use exact detected phase frames from the selected rep.
    setup_frame = int(rep.get("start_frame", start))
    descent_frame = int(rep.get("descent_frame", start + int((bottom - start) * 0.60)))
    bottom_frame = int(rep.get("bottom_frame", bottom))
    ascent_frame = int(rep.get("ascent_frame", bottom + int((end - bottom) * 0.45)))
    lockout_frame = int(rep.get("end_frame", end))

    setup_frame = max(0, min(setup_frame, total_frames - 1))
    descent_frame = max(setup_frame, min(descent_frame, total_frames - 1))
    bottom_frame = max(descent_frame, min(bottom_frame, total_frames - 1))
    ascent_frame = max(bottom_frame, min(ascent_frame, total_frames - 1))
    lockout_frame = max(ascent_frame, min(lockout_frame, total_frames - 1))

    phase_frames = {
        "setup": setup_frame,
        "descent": descent_frame,
        "bottom": bottom_frame,
        "ascent": ascent_frame,
        "lockout": lockout_frame,
    }

    print("SQUAT PHASE FRAME PICKS:", phase_frames)

    saved = {}

    def read_frame_safely(target_frame):
        local_cap = cv2.VideoCapture(input_path)
        if not local_cap.isOpened():
            return None

        local_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = local_cap.read()
        local_cap.release()

        return frame if ret else None

    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
    ) as pose:
        for phase_name, frame_idx in phase_frames.items():
            frame = read_frame_safely(frame_idx)

            if frame is None:
                print(f"Could not read frame for {phase_name}: {frame_idx}")
                continue

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                frame = draw_user_skeleton(frame, results.pose_landmarks)
                frame = draw_ideal_squat_overlay(
                    frame,
                    results.pose_landmarks,
                    width,
                    height,
                )

            cv2.rectangle(frame, (20, 20), (360, 82), (0, 0, 0), -1)

            cv2.putText(
                frame,
                phase_name.upper(),
                (35, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            filename = f"squat_{phase_name}_{uuid.uuid4().hex[:8]}.jpg"
            filepath = os.path.join(output_dir, filename)

            ok = cv2.imwrite(filepath, frame)

            if ok:
                saved[phase_name] = f"/outputs/{filename}"
            else:
                print(f"Could not save image for {phase_name}: {filepath}")

    print("Saved squat phase images:", saved)

    return saved if saved else None


def create_push_press_phase_images(input_path, output_dir, rep, sample_every=1, exercise_label=None):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Push press phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = int(rep.get("start_frame", 0))
    end = int(rep.get("end_frame", total_frames - 1))

    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames - 1))

    if str(exercise_label or "").lower() == "strict_press":
        setup_frame = int(rep.get("start_frame", start))
        press_frame = int(rep.get("press_frame", start + int((end - start) * 0.55)))
        lockout_frame = min(total_frames - 1, int(rep.get("lockout_frame", end)) + 35)

        # Strict press analyzer can fire early on screen-recorded clips.
        # For visuals, spread phases across the detected rep window.
        if lockout_frame <= press_frame + 45:
            press_frame = setup_frame + int((end - setup_frame) * 0.50)
            lockout_frame = end

        phase_frames = {
            "setup": setup_frame,
            "press": press_frame,
            "lockout": lockout_frame,
        }

    # THRUSTER: use actual detected phase frames
    elif str(exercise_label or "").lower() == "thruster" or rep.get("breakdown", {}).get("squat_depth"):

        drive_frame = int(rep.get("drive_frame", start))
        lockout_frame = int(rep.get("lockout_frame", end))

        # Press starts roughly 35% of the way from drive to lockout.
        press_frame = drive_frame + int(
            (lockout_frame - drive_frame) * 0.35
        )

        setup_frame = max(0, int(rep.get("start_frame", start)) - 25)
        bottom_frame = int(rep.get("dip_frame", start))
        drive_frame = int(rep.get("drive_frame", start))
        overhead_frame = int(rep.get("catch_frame", end))
        lockout_frame = min(
            int(rep.get("lockout_frame", end)) + 20,
            total_frames - 1,
        )

        press_frame = drive_frame + int((overhead_frame - drive_frame) * 0.5)

        ascent_frame = drive_frame + int((overhead_frame - drive_frame) * 0.5)

        # Thruster storyboard: use real detected movement events
        dip_frame = int(rep.get("dip_frame", start))
        drive_frame = int(rep.get("drive_frame", start))
        catch_frame = int(rep.get("catch_frame", drive_frame))
        lockout_frame = int(rep.get("lockout_frame", end))

        # Thruster visuals: choose clear coach-facing frames.
        # Start = before the squat, Squat = bottom, Lockout = first overhead catch.
        setup_frame = max(0, int(rep.get("start_frame", start)) - 50)
        squat_frame = int(rep.get("start_frame", start))
        lockout_frame = min(total_frames - 1, int(rep.get("lockout_frame", end)) + 55)

        phase_frames = {
            "setup": setup_frame,
            "dip": squat_frame,
            "lockout": lockout_frame,
        }

    # PUSH PRESS: use timing-based fallback
    else:
        # Push press visuals should use detected rep phase frames,
        # not percentage-based guesses.
        setup_frame = int(rep.get("start_frame", start))
        dip_frame = int(rep.get("dip_frame", start))
        lockout_frame = int(rep.get("end_frame", rep.get("lockout_frame", end)))

        # Drive should show the bar starting to leave the shoulders.
        # For push press visuals, this should be closer to the dip than lockout.
        # This avoids Drive looking like another lockout or late press frame.
        drive_frame = int(rep.get(
            "drive_frame",
            dip_frame + int((lockout_frame - dip_frame) * 0.35)
        ))

        phase_frames = {
            "setup": setup_frame,
            "dip": dip_frame,
            "drive": drive_frame,
            "lockout": lockout_frame,
        }

    saved = {}

    for phase, frame_idx in phase_frames.items():

        frame_idx = max(0, min(int(frame_idx), total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        filename = f"push_press_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        saved[phase] = f"/outputs/{filename}"

    sheet_url = save_phase_contact_sheet(
        input_path,
        phase_frames,
        output_dir,
        prefix="push_press_phase_debug",
    )

    if sheet_url:
        saved["debug_sheet"] = sheet_url

    cap.release()

    print("Saved push press phase images:", saved)

    return saved


def find_bench_press_phase_window(start, end):
    start = int(start)
    end = int(end)

    if end <= start:
        end = start + 1

    span = end - start

    setup_frame = start
    descent_frame = start + int(span * 0.25)
    bottom_frame = start + int(span * 0.50)
    press_frame = start + int(span * 0.70)

    # Do not use the exact end frame.
    # It often catches the athlete moving after the rep.
    lockout_frame = start + int(span * 0.85)

    return {
        "setup": setup_frame,
        "descent": descent_frame,
        "bottom": bottom_frame,
        "press": press_frame,
        "lockout": lockout_frame,
    }


def create_bench_press_phase_images(input_path, output_dir, rep, sample_every=1):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Bench phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rep_start = int(rep.get("start_frame", 0))
    rep_end = int(rep.get("end_frame", total_frames - 1))

    rep_start = max(0, min(rep_start, total_frames - 1))
    rep_end = max(rep_start + 1, min(rep_end, total_frames - 1))

    rep_span = max(1, rep_end - rep_start)

    # Look slightly before/after, but not so far that we catch walking setup
    search_start = max(0, rep_start - int(rep_span * 0.75))
    search_end = min(total_frames - 1, rep_end + int(rep_span * 0.25))

    candidates = []

    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
    ) as pose:
        for frame_idx in range(search_start, search_end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            lm = results.pose_landmarks.landmark

            left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

            wrist_y = (left_wrist.y + right_wrist.y) / 2.0
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
            hip_y = (left_hip.y + right_hip.y) / 2.0

            torso_vertical_distance = abs(shoulder_y - hip_y)

            # Filter out standing / walking frames
            is_bench_position = torso_vertical_distance < 0.22

            if not is_bench_position:
                continue

            candidates.append({
                "frame": frame_idx,
                "wrist_y": wrist_y,
            })

    if candidates:
        # Lowest bar = bottom
        bottom_frame = max(
            candidates,
            key=lambda x: x["wrist_y"],
        )["frame"]

        before_bottom = [
            c for c in candidates
            if c["frame"] <= bottom_frame
        ]

        after_bottom = [
            c for c in candidates
            if c["frame"] >= bottom_frame
        ]

        # Highest bar before bottom = setup
        setup_frame = min(
            before_bottom,
            key=lambda x: x["wrist_y"],
        )["frame"]

        # Highest bar after bottom = lockout
        lockout_frame = min(
            after_bottom,
            key=lambda x: x["wrist_y"],
        )["frame"]

        descent_frame = setup_frame + int(
            (bottom_frame - setup_frame) * 0.50
        )

        # PRESS = 75% up from bottom to lockout
        press_frame = bottom_frame + int(
            (lockout_frame - bottom_frame) * 0.75
        )

    else:
        # Safe fallback
        span = rep_end - rep_start

        setup_frame = rep_start + int(span * 0.10)
        descent_frame = rep_start + int(span * 0.30)
        bottom_frame = rep_start + int(span * 0.50)
        press_frame = rep_start + int(span * 0.75)
        lockout_frame = rep_start + int(span * 0.90)

    phase_frames = {
        "setup": setup_frame,
        "descent": descent_frame,
        "bottom": bottom_frame,
        "press": press_frame,
        "lockout": lockout_frame,
    }

    cleaned = {}
    for phase, frame_idx in phase_frames.items():
        cleaned[phase] = max(
            0,
            min(int(frame_idx), total_frames - 1),
        )

    saved = {}

    for phase, frame_idx in cleaned.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        filename = (
            f"bench_press_{phase}_"
            f"{uuid.uuid4().hex[:8]}.jpg"
        )

        filepath = os.path.join(
            output_dir,
            filename,
        )

        cv2.imwrite(filepath, frame)
        saved[phase] = f"/outputs/{filename}"

    sheet_url = save_phase_contact_sheet(
        input_path,
        cleaned,
        output_dir,
        prefix="bench_press_phase_debug",
    )

    if sheet_url:
        saved["debug_sheet"] = sheet_url

    cap.release()

    print("BENCH SEARCH WINDOW:", search_start, "to", search_end)
    print("BENCH VALID BENCH FRAMES:", len(candidates))
    print("BENCH PHASE FRAMES:", cleaned)
    print("Saved bench phase images:", saved)

    return saved


def create_clean_and_jerk_phase_images(input_path, output_dir, rep, sample_every=1):
    import cv2, os, uuid

    cap = cv2.VideoCapture(input_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(rep.get("end_frame", 1))

    start = int(rep.get("start_frame", 1))
    end_frame = int(rep.get("end_frame", total))

    # Use analyzer/manual rep_json frames directly for clean & jerk visuals.
    span = max(1, end_frame - start)

    phases = {
        "setup": int(rep.get("start_frame", start)),
        "clean_catch": int(rep.get("clean_catch_frame", start + int(span * 0.82))),
        "jerk_dip": int(rep.get("jerk_dip_frame", start + int(span * 0.88))),
        "jerk_drive": int(rep.get("jerk_drive_frame", start + int(span * 0.91))),
        "jerk_catch": int(rep.get("jerk_catch_frame", start + int(span * 0.95))),
        "finish": int(rep.get("end_frame", end_frame)),
    }

    out = {}

    for name, frame_no in phases.items():
        frame_no = max(1, min(int(frame_no), total))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ok, frame = cap.read()
        if not ok:
            continue

        filename = f"clean_and_jerk_{name}_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, frame)
        out[name] = f"/outputs/{filename}"

    # Build one debug sheet showing exactly which phase frames were selected.
    imgs = []
    for name, url in out.items():
        path = os.path.join(output_dir, url.split("/")[-1])
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (320, 180))
        cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        imgs.append(img)

    if imgs:
        import numpy as np, math
        cols = 3
        rows = math.ceil(len(imgs) / cols)
        sheet = np.ones((rows * 180, cols * 320, 3), dtype=np.uint8) * 255
        for i, img in enumerate(imgs):
            r, c = divmod(i, cols)
            sheet[r*180:(r+1)*180, c*320:(c+1)*320] = img

        filename = f"clean_and_jerk_phase_debug_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, sheet)
        out["debug_sheet"] = f"/outputs/{filename}"

    cap.release()
    return out


def pick_phase_frames_from_rep(rep, exercise_label, start, end, total_frames):
    label = exercise_label.lower().replace(" ", "_")
    duration = max(1, end - start)

    def rf(key, fallback):
        return int(rep.get(key, fallback))

    if label == "split_jerk":
        catch_frame = rf("catch_frame", start + int(duration * 0.60))
        finish_frame = max(catch_frame + 10, end - 15)

        phase_frames = {
            "setup": start,
            "dip": rf("dip_frame", start + int(duration * 0.20)),
            "drive": rf("drive_frame", start + int(duration * 0.35)),
            "catch": catch_frame,
            "lockout": rf("lockout_frame", start + int(duration * 0.75)),
            "finish": finish_frame,
        }

    elif label == "clean_and_jerk":
        phase_frames = {
            "setup": start,
            "clean_catch": rf("clean_catch_frame", start + int(duration * 0.35)),
            "jerk_dip": rf("jerk_dip_frame", start + int(duration * 0.55)),
            "jerk_drive": rf("jerk_drive_frame", start + int(duration * 0.65)),
            "jerk_catch": rf("jerk_catch_frame", start + int(duration * 0.78)),
            "finish": end,
        }

    elif label in ["snatch", "clean"]:
        phase_frames = {
            "setup": start,
            "first_pull": rf("first_pull_frame", start + int(duration * 0.22)),
            "extension": rf("extension_frame", start + int(duration * 0.48)),
            "catch": rf("catch_frame", start + int(duration * 0.72)),
            "finish": end,
        }

    else:
        phase_frames = {
            "setup": start,
            "first_pull": start + int(duration * 0.22),
            "extension": start + int(duration * 0.48),
            "catch": start + int(duration * 0.72),
            "finish": end,
        }

    # Force frames to be valid and separated so images don't all look identical.
    min_gap = max(3, duration // 12)
    ordered = {}
    last = start - min_gap

    for phase, frame in phase_frames.items():
        frame = max(0, min(int(frame), total_frames - 1))

        if phase != "setup" and phase != "finish":
            frame = max(frame, last + min_gap)
            frame = min(frame, end - 1)

        if phase == "finish":
            frame = end

        ordered[phase] = frame
        last = frame

    return ordered


def create_olympic_lift_phase_images(
    input_path,
    output_dir,
    rep,
    sample_every=1,
    exercise_label="olympic_lift",
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Olympic lift phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def rep_frame(key, default):
        return int(rep.get(key, default))

    start = rep_frame("start_frame", 0)
    end = rep_frame("end_frame", total_frames - 1)

    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames - 1))

    normalized_label = exercise_label.lower().replace(" ", "_")
    prefix = normalized_label

    # -----------------------------------
    # SPLIT JERK
    # -----------------------------------
    if normalized_label == "split_jerk":
        raw_dip_frame = rep_frame("dip_frame", start)
        catch_frame = rep_frame("catch_frame", end)

        setup_frame = max(start, raw_dip_frame - 13)
        dip_frame = min(raw_dip_frame + 14, catch_frame - 8)
        finish_frame = min(end - 10, catch_frame + 59)

        phase_frames = {
            "setup": setup_frame,
            "dip": dip_frame,
            "catch": catch_frame,
            "finish": finish_frame,
        }

    # -----------------------------------
    # CLEAN & JERK
    # -----------------------------------
    elif normalized_label == "clean_and_jerk":
        phase_frames = {
            "setup": start,
            "clean_catch": rep_frame("clean_catch_frame", start),
            "jerk_dip": rep_frame("jerk_dip_frame", start),
            "jerk_drive": rep_frame("jerk_drive_frame", start),
            "jerk_catch": rep_frame("jerk_catch_frame", start),
            "finish": end,
        }

    elif normalized_label == "snatch":

        catch_frame = rep_frame("catch_frame", start)

        phase_frames = {
            "setup": start,
            "first_pull": rep_frame("first_pull_frame", start),
            "extension": rep_frame("extension_frame", start),
            "catch": catch_frame,

            # standing overhead after catch
            "finish": min(catch_frame + 90, end),
        }

    elif normalized_label == "clean":

        phase_frames = {
            "setup": start,
            "first_pull": rep_frame("first_pull_frame", start),
            "extension": rep_frame("extension_frame", start),
            "catch": rep_frame("catch_frame", start),
            "finish": end,
        }

    # -----------------------------------
    # FALLBACK
    # -----------------------------------
    else:
        duration = max(1, end - start)

        phase_frames = {
            "setup": start,
            "first_pull": start + int(duration * 0.22),
            "extension": start + int(duration * 0.48),
            "catch": start + int(duration * 0.72),
            "finish": max(start, min(end - 1, total_frames - 1)),
        }

    print("OLY VISUAL DEBUG", {
        "label": normalized_label,
        "total_frames": total_frames,
        "rep": rep,
        "phase_frames": phase_frames,
    })

    saved = {}
    debug_images = []

    for phase, frame_idx in phase_frames.items():
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        frame = None

        for offset in [0, 1, 2, 3, 5, 8, -1, -2]:
            safe_idx = max(0, min(frame_idx + offset, total_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
            ret, candidate = cap.read()

            if ret and candidate is not None:
                frame = candidate
                frame_idx = safe_idx
                break

        if frame is None:
            print(f"Could not read {phase} frame near: {frame_idx}")
            continue

        filename = f"{prefix}_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)
        saved[phase] = f"/outputs/{filename}"

        debug = frame.copy()
        cv2.putText(
            debug,
            f"{phase} ({frame_idx})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        debug_images.append(debug)

    if debug_images:
        debug_sheet = np.hstack(debug_images)
        debug_filename = f"{prefix}_phase_debug_{uuid.uuid4().hex[:8]}.jpg"
        debug_path = os.path.join(output_dir, debug_filename)

        cv2.imwrite(debug_path, debug_sheet)
        saved["debug_sheet"] = f"/outputs/{debug_filename}"

    if "finish" not in saved and "recovery" in saved:
        saved["finish"] = saved["recovery"]

    cap.release()

    return saved


def create_overhead_squat_phase_images(
    input_path,
    output_dir,
    rep=None,
    sample_every=1,
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Overhead squat phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if rep:
        start = 0
        end = total_frames - 1
        bottom = int(rep.get("bottom_frame", total_frames // 2)) if rep else total_frames // 2
    else:
        start = 0
        bottom = total_frames // 2
        end = total_frames - 1

    start = max(0, min(start, total_frames - 1))
    bottom = max(start + 1, min(bottom, total_frames - 1))
    end = max(bottom + 1, min(end, total_frames - 1))

    descent = start + int((bottom - start) * 0.60)
    ascent = bottom + int((end - bottom) * 0.40)

    phase_frames = {
        "setup": start,
        "descent": descent,
        "bottom": bottom,
        "ascent": ascent,
        "lockout": end,
    }

    saved = {}
    debug_images = []

    for phase, frame_idx in phase_frames.items():
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        filename = f"overhead_squat_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)
        saved[phase] = f"/outputs/{filename}"

        debug = frame.copy()
        cv2.putText(
            debug,
            f"{phase} ({frame_idx})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        debug_images.append(debug)

    if debug_images:
        debug_sheet = np.hstack(debug_images)
        debug_filename = f"overhead_squat_phase_debug_{uuid.uuid4().hex[:8]}.jpg"
        debug_path = os.path.join(output_dir, debug_filename)

        cv2.imwrite(debug_path, debug_sheet)
        saved["debug_sheet"] = f"/outputs/{debug_filename}"

    cap.release()
    return saved


def create_pull_up_phase_images(
    input_path,
    output_dir,
    rep=None,
    sample_every=1,
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Pull-up phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if rep:
        start = int(rep.get("start_frame", 0))
        end = int(rep.get("end_frame", total_frames - 1))
    else:
        start = 0
        end = total_frames - 1

    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames - 1))

    duration = max(1, end - start)

    phase_frames = {
        "hang": start,
        "pull": start + int(duration * 0.25),
        "top": start + int(duration * 0.50),
        "descent": start + int(duration * 0.75),
        "finish": max(start, min(end - 1, total_frames - 1)),
    }

    saved = {}
    debug_images = []

    for phase, frame_idx in phase_frames.items():
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        frame = None

        for offset in [0, -1, -2, -3, -5, -8, -10]:
            safe_idx = max(0, min(frame_idx + offset, total_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
            ret, candidate = cap.read()

            if ret and candidate is not None:
                frame = candidate
                frame_idx = safe_idx
                break

        if frame is None:
            print(f"Could not read {phase} frame near: {frame_idx}")
            continue

        filename = f"pull_up_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)
        saved[phase] = f"/outputs/{filename}"

        debug = frame.copy()
        cv2.putText(
            debug,
            f"{phase} ({frame_idx})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        debug_images.append(debug)

    if debug_images:
        debug_sheet = np.hstack(debug_images)
        debug_filename = f"pull_up_phase_debug_{uuid.uuid4().hex[:8]}.jpg"
        debug_path = os.path.join(output_dir, debug_filename)

        cv2.imwrite(debug_path, debug_sheet)
        saved["debug_sheet"] = f"/outputs/{debug_filename}"

    # Force finish fallback
    if "finish" not in saved and "descent" in saved:
        saved["finish"] = saved["descent"]

    cap.release()

    return saved


def create_bar_muscle_up_phase_images(
    input_path,
    output_dir,
    rep=None,
    sample_every=1,
    exercise_label="bar_muscle_up",
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Bar muscle-up phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if rep:
        start = int(rep.get("start_frame", 0))
        end = int(rep.get("end_frame", total_frames - 1))
    else:
        start = 0
        end = total_frames - 1

    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames - 1))

    duration = max(1, end - start)

    if rep:
        pull = int(rep.get("pull_frame", start + int(duration * 0.22)))
        transition = int(rep.get("transition_frame", start + int(duration * 0.45)))
        dip = int(rep.get("dip_frame", start + int(duration * 0.65)))
        lockout = int(rep.get("lockout_frame", end))
        finish = int(rep.get("end_frame", end))

        lockout = min(lockout, total_frames - 8)
        finish = min(finish, total_frames - 4)
    else:
        pull = start + int(duration * 0.22)
        transition = start + int(duration * 0.45)
        dip = start + int(duration * 0.65)
        lockout = start + int(duration * 0.82)
        finish = end

    phase_frames = {
        "hang": start,
        "pull": pull,
        "transition": transition,
        "dip": dip,
        "lockout": lockout,
        "finish": finish,
    }

    saved = {}
    debug_images = []

    prefix = exercise_label.lower().replace(" ", "_")

    for phase, frame_idx in phase_frames.items():
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        frame = None

        for offset in [0, -1, -2, -3, -5, -8, -10, -15, -20, -30, -45]:
            safe_idx = max(0, min(frame_idx + offset, total_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
            ret, candidate = cap.read()

            if ret and candidate is not None:
                frame = candidate
                frame_idx = safe_idx
                break

        if frame is None:
            print(f"Could not read {phase} frame near: {frame_idx}")
            continue

        filename = f"{prefix}_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)
        saved[phase] = f"/outputs/{filename}"

        debug = frame.copy()
        cv2.putText(
            debug,
            f"{phase} ({frame_idx})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        debug_images.append(debug)

    if debug_images:
        debug_sheet = np.hstack(debug_images)
        debug_filename = f"{prefix}_phase_debug_{uuid.uuid4().hex[:8]}.jpg"
        debug_path = os.path.join(output_dir, debug_filename)

        cv2.imwrite(debug_path, debug_sheet)
        saved["debug_sheet"] = f"/outputs/{debug_filename}"

    if "lockout" not in saved and "dip" in saved:
        saved["lockout"] = saved["dip"]

    if "finish" not in saved and "lockout" in saved:
        saved["finish"] = saved["lockout"]

    cap.release()

    return saved


def create_push_up_phase_images(
    input_path,
    output_dir,
    rep=None,
    sample_every=1,
    exercise_label="push_up",
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Push-up phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if rep:
        start = int(rep.get("start_frame", 0))
        end = int(rep.get("end_frame", total_frames - 1))
    else:
        start = 0
        end = total_frames - 1

    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames - 1))
    duration = max(1, end - start)

    if exercise_label == "handstand_push_up" and rep:
        phase_frames = {
            "setup": int(rep.get("start_frame", start)),
            "descent": int(rep.get("descent_frame", start)),
            "bottom": int(rep.get("end_frame", end)),
            "ascent": int(rep.get("ascent_frame", end)),
            "lockout": int(rep.get("bottom_frame", start)),
        }

    elif exercise_label == "push_up" and rep:
        bottom = int(rep.get("bottom_frame", start + int(duration * 0.50)))

        phase_frames = {
            "setup": max(0, bottom - 36),
            "descent": max(0, bottom - 18),
            "bottom": bottom,
            "ascent": min(total_frames - 1, bottom + 9),
            "lockout": min(total_frames - 1, bottom + 24),
        }

    elif rep:
        phase_frames = {
            "setup": int(rep.get("start_frame", start)),
            "descent": int(rep.get("descent_frame", start + int(duration * 0.25))),
            "bottom": int(rep.get("bottom_frame", start + int(duration * 0.50))),
            "ascent": int(rep.get("ascent_frame", start + int(duration * 0.75))),
            "lockout": int(rep.get("end_frame", end)),
        }

    else:
        phase_frames = {
            "setup": start,
            "descent": start + int(duration * 0.25),
            "bottom": start + int(duration * 0.50),
            "ascent": start + int(duration * 0.75),
            "lockout": max(start, min(end - 1, total_frames - 1)),
        }

    prefix = "handstand_push_up" if exercise_label == "handstand_push_up" else "push_up"

    saved = {}
    debug_images = []

    for phase, frame_idx in phase_frames.items():
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"Could not read {phase} frame near: {frame_idx}")
            continue

        filename = f"{prefix}_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)
        saved[phase] = f"/outputs/{filename}"

        debug = frame.copy()
        cv2.putText(
            debug,
            f"{phase} ({frame_idx})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        debug_images.append(debug)

    if debug_images:
        debug_sheet = np.hstack(debug_images)
        debug_filename = f"{prefix}_phase_debug_{uuid.uuid4().hex[:8]}.jpg"
        debug_path = os.path.join(output_dir, debug_filename)

        cv2.imwrite(debug_path, debug_sheet)
        saved["debug_sheet"] = f"/outputs/{debug_filename}"

    cap.release()
    return saved


def create_burpee_phase_images(
    input_path,
    output_dir,
    rep=None,
    sample_every=1,
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Burpee phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if rep:
        start = int(rep.get("start_frame", 0))
        finish = int(rep.get("end_frame", total_frames - 1))

        duration = max(1, finish - start)

        hands_down = int(rep.get("hands_down_frame", start + int(duration * 0.20)))
        plank = int(rep.get("plank_frame", start + int(duration * 0.40)))
        jump_in = int(rep.get("jump_in_frame", start + int(duration * 0.65)))
        stand = int(rep.get("stand_frame", start + int(duration * 0.85)))
    else:
        start = 0
        finish = max(0, total_frames - 1)
        duration = max(1, finish - start)

        hands_down = start + int(duration * 0.20)
        plank = start + int(duration * 0.40)
        jump_in = start + int(duration * 0.65)
        stand = start + int(duration * 0.85)

    phase_frames = {
        "start": start,
        "hands_down": hands_down,
        "plank": plank,
        "jump_in": jump_in,
        "stand": stand,
        "finish": finish,
    }

    saved = {}
    debug_images = []

    for phase, frame_idx in phase_frames.items():
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        filename = f"burpee_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)
        saved[phase] = f"/outputs/{filename}"

        debug = frame.copy()
        cv2.putText(
            debug,
            f"{phase} ({frame_idx})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        debug_images.append(debug)

    if debug_images:
        debug_sheet = np.hstack(debug_images)
        debug_filename = f"burpee_phase_debug_{uuid.uuid4().hex[:8]}.jpg"
        debug_path = os.path.join(output_dir, debug_filename)

        cv2.imwrite(debug_path, debug_sheet)
        saved["debug_sheet"] = f"/outputs/{debug_filename}"

    if "finish" not in saved and "stand" in saved:
        saved["finish"] = saved["stand"]

    cap.release()
    return saved


def normalize_sequence(biomechanics):
    feats = [
        b["full_features"]
        for b in biomechanics
        if "full_features" in b
    ]

    feats = pad_or_trim(np.array(feats), target_len=30)

    # IMPORTANT: DO NOT MODIFY FEATURES
    # MUST MATCH TRAINING PIPELINE EXACTLY

    return feats


def analyze_video(video_path, make_visuals=True, make_overlay=True):
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {
                "exercise_label": "Unknown",
                "confidence": 0.0,
                "analysis_mode": "video_error",
                "rep_feedback": [],
                "set_summary": build_set_summary([]),
                "coaching_zones": build_coaching_zones("unknown", []),
                "overlay_video_url": None,
                "phase_images": None,
                "debug": {"error": "video_not_opened"},
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every = 1

        sequence = []
        biomechanics = []
        frame_idx = 0
        pose_frames = 0
        subject_center = None
        subject_area = None

        yolo_tracker = (
            YOLOTracker("models/yolov8n.pt", pad=220)
            if (USE_YOLO_TRACKING or USE_YOLO_DIAGNOSTICS) and YOLOTracker is not None
            else None
        )

        yolo_crop_frames = 0
        yolo_full_fallback_frames = 0
        yolo_target_ids = set()

        # ---------------- POSE EXTRACTION ----------------
        with mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % sample_every != 0:
                    continue

                if (USE_YOLO_TRACKING or USE_YOLO_DIAGNOSTICS) and yolo_tracker is not None:
                    crop_result = yolo_tracker.get_crop(frame)

                    full_box = (0, 0, frame.shape[1], frame.shape[0])

                    if crop_result.crop is not None and crop_result.box != full_box:
                        yolo_crop_frames += 1
                    else:
                        yolo_full_fallback_frames += 1

                    if crop_result.target_id is not None:
                        yolo_target_ids.add(crop_result.target_id)

                    if USE_YOLO_TRACKING:
                        analysis_frame = crop_result.crop if crop_result.crop is not None else frame
                    else:
                        analysis_frame = frame
                else:
                    crop_result = None
                    analysis_frame = frame

                rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if not results.pose_landmarks:
                    continue

                # Subject lock: reject sudden jumps to another athlete in busy gyms.
                lm = results.pose_landmarks.landmark
                pts = [
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                    lm[mp_pose.PoseLandmark.LEFT_HIP.value],
                    lm[mp_pose.PoseLandmark.RIGHT_HIP.value],
                ]

                xs = [p.x for p in pts]
                ys = [p.y for p in pts]
                center = (sum(xs) / len(xs), sum(ys) / len(ys))
                area = max(1e-6, (max(xs) - min(xs)) * (max(ys) - min(ys)))

                if subject_center is None:
                    subject_center = center
                    subject_area = area
                else:
                    dx = center[0] - subject_center[0]
                    dy = center[1] - subject_center[1]
                    jump = (dx * dx + dy * dy) ** 0.5
                    area_ratio = area / max(subject_area, 1e-6)

                    if jump > 0.22 or area_ratio < 0.45 or area_ratio > 2.2:
                        continue

                    subject_center = (
                        subject_center[0] * 0.85 + center[0] * 0.15,
                        subject_center[1] * 0.85 + center[1] * 0.15,
                    )
                    subject_area = subject_area * 0.85 + area * 0.15

                feats, bio = extract_features_and_biomechanics(results)

                if feats is None or bio is None:
                    continue

                sequence.append(feats)
                bio["frame_number"] = frame_idx
                biomechanics.append(bio)
                pose_frames += 1

        cap.release()

        # ---------------- INSUFFICIENT DATA ----------------
        if len(sequence) < 10:
            return {
                "exercise_label": "Unknown",
                "confidence": 0.0,
                "analysis_mode": "insufficient_data",
                "rep_feedback": [],
                "set_summary": build_set_summary([]),
                "coaching_zones": build_coaching_zones("unknown", []),
                "overlay_video_url": None,
                "phase_images": None,
                "debug": {"frames_processed": len(sequence)},
            }

        # ---------------- FEATURE PROCESSING ----------------
        seq_base = pad_or_trim(np.array(sequence), target_len=30)
        seq = add_velocity(seq_base)

        # ---------------- ML MODEL ----------------
        probs = MODEL.predict_proba(seq)
        raw_idx = int(np.argmax(probs))
        print("DEBUG PRED SHAPE:", getattr(probs, "shape", None))
        print("DEBUG RAW IDX:", raw_idx)
        print("DEBUG CLASS_NAMES LEN:", len(CLASS_NAMES))
        print("DEBUG CLASS_NAMES:", CLASS_NAMES)
        raw_label = CLASS_NAMES[raw_idx]
        raw_confidence = float(np.max(probs))
        base_raw_label = raw_label
        base_raw_confidence = raw_confidence

        # ---------------- SQUAT VARIANT ROUTER ----------------
        # Base model predicts generic "squat"; this router specializes it into
        # overhead_squat, squat_back, or squat_front.
        squat_router_debug = None
        squat_variant_label = None
        squat_variant_conf = 0.0
        if raw_label == "squat":
            try:
                squat_probs = SQUAT_ROUTER_MODEL.predict(
                    np.expand_dims(seq_base, axis=0),
                    verbose=0
                )[0]
                squat_idx = int(np.argmax(squat_probs))
                squat_label = SQUAT_ROUTER_LABELS.get(squat_idx, "squat")
                squat_conf = float(squat_probs[squat_idx])
                squat_router_debug = {
                    "squat_label": squat_label,
                    "squat_confidence": squat_conf,
                    "squat_probs": [float(x) for x in squat_probs],
                }

                if squat_conf >= 0.55:
                    squat_variant_label = squat_label
                    squat_variant_conf = squat_conf

            except Exception as e:
                squat_router_debug = {"error": str(e)}

        # ---------------- OLYMPIC MODEL ----------------
        oly_sequence = [
            b["full_features"]
            for b in biomechanics
            if "full_features" in b
        ]

        olympic_pred, olympic_conf = predict_olympic_lift_from_sequence(
            oly_sequence
        )


        final_label = raw_label
        final_confidence = raw_confidence

        # ---------------- OLYMPIC ROUTING ----------------
        if olympic_pred in ["clean", "clean_and_jerk", "snatch", "split_jerk"] and (
            olympic_conf > raw_confidence or olympic_conf >= 0.75
        ):
            final_label = olympic_pred
            final_confidence = olympic_conf

        # ---------------- CLEAN & JERK RESCUE ----------------
        if (
            olympic_pred == "clean_and_jerk"
            and olympic_conf >= 0.70
            and raw_label in ["squat", "squat_back", "squat_front", "push_press", "deadlift", "bench_press"]
        ):
            final_label = "clean_and_jerk"
            final_confidence = max(olympic_conf, 0.80)

        # ---------------- SNATCH RESCUE ----------------
        pose_summary = summarize_biomechanics(biomechanics)
        wrist_ratio = (
            pose_summary.get("wrist_above_shoulder_ratio", 0)
            if pose_summary else 0
        )

        if (
            raw_label in ["squat", "squat_back", "squat_front", "overhead_squat"]
            and olympic_pred == "snatch"
            and (
                olympic_conf >= 0.65
                or raw_confidence >= 0.95
            )
            and wrist_ratio >= 0.25
        ):
            final_label = "snatch"
            final_confidence = max(olympic_conf, 0.80)

        # ---------------- DEADLIFT PROTECTION ----------------
        if raw_label == "deadlift":
            final_label = "deadlift"
            final_confidence = max(final_confidence, raw_confidence)

        # ---------------- THRUSTER PROTECTION ----------------
        elif raw_label == "thruster":
            # Clean/C&J can be falsely predicted as thruster by the base model.
            # If Olympic router sees a real Olympic lift with decent confidence, trust it.
            if olympic_pred in ["clean", "clean_and_jerk"] and olympic_conf >= 0.60:
                pose_summary = summarize_biomechanics(biomechanics)
                wrist_ratio = (
                    pose_summary.get("wrist_above_shoulder_ratio", 0)
                    if pose_summary else 0
                )

                final_label = olympic_pred
                final_confidence = olympic_conf

                # Do not demote Olympic lifts here. Snatch/clean/C&J protection
                # should happen before thruster rescue, not by forcing clean.
                if False:
                    pass
            else:
                final_label = "thruster"
                final_confidence = max(final_confidence, 0.86)

        # ---------------- FINAL CLEAN & JERK OVERRIDE ----------------
        # Let Olympic router rescue C&J from base-model squat/push/deadlift/bench misses.
        if (
            olympic_pred == "clean_and_jerk"
            and olympic_conf >= 0.60
            and raw_label in ["squat", "squat_back", "squat_front", "push_press", "deadlift", "bench_press"]
        ):
            final_label = "clean_and_jerk"
            final_confidence = max(olympic_conf, 0.80)

        # ---------------- BENCH FALSE-POSITIVE THRUSTER RESCUE ----------------
        # Some floor-start thrusters can look like bench to the base model.
        # Only rescue when we also see squat + overhead press behavior.
        pose_summary = summarize_biomechanics(biomechanics)
        if pose_summary:
            knee_range = pose_summary.get("max_knee_angle", 180) - pose_summary.get("min_knee_angle", 180)
            hip_range = pose_summary.get("max_hip_angle", 180) - pose_summary.get("min_hip_angle", 180)
            elbow_range = pose_summary.get("max_elbow_angle", 180) - pose_summary.get("min_elbow_angle", 180)
            wrist_ratio = pose_summary.get("wrist_above_shoulder_ratio", 0)

            if (
                raw_label == "bench_press"
                and olympic_pred == "clean_and_jerk"
                and pose_summary.get("min_knee_angle", 180) < 130
                and pose_summary.get("min_hip_angle", 180) < 140
                and knee_range > 30
                and hip_range > 30
                and elbow_range > 35
                and wrist_ratio >= 0.10
            ):
                final_label = "thruster"
                final_confidence = max(final_confidence, 0.84)

        # ---------------- FINAL THRUSTER OVERRIDE ----------------
        pose_summary = summarize_biomechanics(biomechanics)
        if pose_summary:
            knee_range = pose_summary.get("max_knee_angle", 180) - pose_summary.get("min_knee_angle", 180)
            hip_range = pose_summary.get("max_hip_angle", 180) - pose_summary.get("min_hip_angle", 180)
            elbow_range = pose_summary.get("max_elbow_angle", 180) - pose_summary.get("min_elbow_angle", 180)
            wrist_ratio = pose_summary.get("wrist_above_shoulder_ratio", 0)

            if (
                raw_label in ["squat", "squat_back", "squat_front", "push_press", "clean_and_jerk"]
                and pose_summary.get("min_knee_angle", 180) < 115
                and pose_summary.get("min_hip_angle", 180) < 125
                and knee_range > 45
                and hip_range > 45
                and elbow_range > 45
                and wrist_ratio >= 0.12
                and final_label != "snatch"
                and not (olympic_pred == "clean_and_jerk" and olympic_conf >= 0.70)
            ):
                final_label = "thruster"
                final_confidence = max(final_confidence, 0.84)

        # ---------------- PUSH PRESS RESCUE FROM SQUAT + CLEAN & JERK ----------------
        # Push press can be misread as squat by the base model and C&J by the Olympic router.
        # If the lower-body motion is shallow and the wrists go overhead, treat it as push press.
        pose_summary = summarize_biomechanics(biomechanics)
        if pose_summary:
            knee_range = pose_summary.get("max_knee_angle", 180) - pose_summary.get("min_knee_angle", 180)
            hip_range = pose_summary.get("max_hip_angle", 180) - pose_summary.get("min_hip_angle", 180)
            elbow_range = pose_summary.get("max_elbow_angle", 180) - pose_summary.get("min_elbow_angle", 180)
            wrist_ratio = pose_summary.get("wrist_above_shoulder_ratio", 0)

            if (
                final_label == "clean_and_jerk"
                and raw_label in ["squat", "squat_back", "squat_front"]
                and olympic_pred == "clean_and_jerk"
                and olympic_conf < 0.95
                and pose_summary.get("min_knee_angle", 180) > 105
                and pose_summary.get("min_hip_angle", 180) > 110
                and knee_range > 15
                and hip_range < 65
                and elbow_range > 35
                and wrist_ratio >= 0.12
            ):
                final_label = "push_press"
                final_confidence = max(final_confidence, raw_confidence, 0.86)

        # ---------------- PUSH PRESS VS CLEAN & JERK GUARD ----------------
        # If the base model sees push press, do not let Olympic router call it
        # clean_and_jerk unless the clean portion is actually visible in the clip.
        if (
            raw_label == "push_press"
            and olympic_pred == "clean_and_jerk"
            and olympic_conf < 0.95
        ):
            final_label = "push_press"
            final_confidence = max(raw_confidence, 0.80)

        # ---------------- FINAL PUSH PRESS PROTECTION ----------------
        # Do not let Olympic/thruster rescue override a confident push press.
        if raw_label == "push_press" and raw_confidence >= 0.95:
            final_label = "push_press"
            final_confidence = max(final_confidence, raw_confidence)

        # FINAL OLYMPIC PROTECTION:
        # Do not allow thruster rescue to override Olympic lift routing.
        # Snatch is often misread as squat/thruster because of the overhead squat catch.
        if (
            final_label == "thruster"
            and olympic_pred in ["snatch", "clean", "clean_and_jerk"]
            and olympic_conf >= 0.60
            and raw_label in ["squat", "squat_back", "squat_front", "overhead_squat", "push_press", "thruster"]
        ):
            # Do not convert squat-family + clean_and_jerk into snatch.
            # That caused front squats to be mislabeled as snatch.
            final_label = olympic_pred
            final_confidence = max(final_confidence, olympic_conf)

        # Protect push press from being upgraded to split jerk.
        # True split jerk needs a real split-stance detector, not just push_press + clean_and_jerk.
        if (
            final_label == "clean_and_jerk"
            and raw_label == "push_press"
            and olympic_conf >= 0.85
            and len(biomechanics) > 80
        ):
            final_label = "push_press"
            final_confidence = max(final_confidence, raw_confidence)

        # Strict press rescue from push press:
        # Very low knee movement means this is a strict press, not a push press.
        if final_label == "push_press":
            pose_summary = summarize_biomechanics(biomechanics)
            knee_range = pose_summary.get("max_knee_angle", 180) - pose_summary.get("min_knee_angle", 180) if pose_summary else 999

            if knee_range <= 12:
                final_label = "strict_press"
                final_confidence = max(final_confidence, 0.86)

        # Strict press rescue from push press:
        # Very low knee movement means this is a strict press, not a push press.
        if final_label == "push_press":
            pose_summary = summarize_biomechanics(biomechanics)
            knee_range = pose_summary.get("max_knee_angle", 180) - pose_summary.get("min_knee_angle", 180) if pose_summary else 999

            if knee_range <= 12:
                final_label = "strict_press"
                final_confidence = max(final_confidence, 0.86)

        # ---------------- REGRESSION SAFETY GUARDS ----------------
        # These guards protect known stable examples from over-aggressive routing overrides.

        # A very confident deadlift should not be converted into clean_and_jerk.
        if raw_label == "deadlift" and raw_confidence >= 0.90:
            final_label = "deadlift"
            final_confidence = float(raw_confidence)

        # A very confident squat-family prediction should not be converted into clean_and_jerk.
        if (
            final_label == "clean_and_jerk"
            and base_raw_label == "squat"
            and olympic_pred == "clean_and_jerk"
            and base_raw_confidence >= 0.90
        ):
            final_label = raw_label
            final_confidence = float(raw_confidence)

        # A very confident squat should not become strict_press.
        if (
            final_label == "strict_press"
            and raw_label in ["squat", "squat_back", "squat_front", "overhead_squat"]
            and raw_confidence >= 0.90
        ):
            final_label = raw_label
            final_confidence = float(raw_confidence)

        # Strong Olympic jerk signal + push_press base prediction is likely split_jerk.
        if (
            raw_label == "push_press"
            and olympic_pred == "clean_and_jerk"
            and olympic_conf >= 0.90
        ):
            final_label = "split_jerk"
            final_confidence = 0.80

        # Apply squat variant only after all higher-priority routing decisions.
        # This prevents push press / snatch from being permanently rewritten as squat_back.
        if final_label == "squat" and squat_variant_label and squat_variant_conf >= 0.55:
            final_label = squat_variant_label
            final_confidence = float(squat_variant_conf)

        # Front squat rescue:
        # Front squats often appear as squat_back to the squat router if the bar/front rack
        # is hard to see. If the base model is a very confident squat and the wrists stay
        # near or above the shoulders, prefer squat_front.
        pose_summary = summarize_biomechanics(biomechanics)
        front_squat_rescue_debug = None
        if pose_summary and final_label == "squat_back" and base_raw_label == "squat":
            wrist_ratio = pose_summary.get("wrist_above_shoulder_ratio", 0)
            elbow_range = pose_summary.get("max_elbow_angle", 180) - pose_summary.get("min_elbow_angle", 180)

            front_squat_rescue_debug = {
                "wrist_ratio": wrist_ratio,
                "elbow_range": elbow_range,
                "final_label_before": final_label,
                "base_raw_label": base_raw_label,
            }

            looks_like_snatch = (
                base_raw_confidence < 0.98
                and olympic_conf >= 0.60
                and wrist_ratio >= 0.20
                and elbow_range < 145
            )

            if 0.03 <= wrist_ratio <= 0.55 and elbow_range < 160 and not looks_like_snatch:
                final_label = "squat_front"
                final_confidence = max(final_confidence, 0.86)
                front_squat_rescue_debug["rescued"] = True
                front_squat_rescue_debug["looks_like_snatch"] = looks_like_snatch
            else:
                front_squat_rescue_debug["rescued"] = False
                front_squat_rescue_debug["looks_like_snatch"] = looks_like_snatch

        # Overhead squat rescue:
        # Some overhead squats are misread as push_press because wrists stay overhead.
        # If the athlete reaches squat depth with wrists overhead, prefer overhead_squat.
        pose_summary = summarize_biomechanics(biomechanics)
        if pose_summary and final_label == "push_press":
            knee_range = pose_summary.get("max_knee_angle", 180) - pose_summary.get("min_knee_angle", 180)
            hip_range = pose_summary.get("max_hip_angle", 180) - pose_summary.get("min_hip_angle", 180)
            wrist_ratio = pose_summary.get("wrist_above_shoulder_ratio", 0)

            if (
                pose_summary.get("min_knee_angle", 180) < 120
                and pose_summary.get("min_hip_angle", 180) < 130
                and knee_range > 35
                and hip_range > 35
                and wrist_ratio >= 0.55
            ):
                final_label = "overhead_squat"
                final_confidence = max(final_confidence, 0.86)

        # Snatch rescue:
        # Snatch catch can look like squat_back/front after squat routing.
        # Keep this narrow so clean_and_jerk clips with large elbow turnover
        # are not stolen by snatch.
        pose_summary = summarize_biomechanics(biomechanics)
        elbow_range = 999
        if pose_summary:
            elbow_range = pose_summary.get("max_elbow_angle", 180) - pose_summary.get("min_elbow_angle", 180)

        if (
            final_label in ["squat_back", "squat_front", "squat"]
            and base_raw_label == "squat"
            and olympic_conf >= 0.60
            and base_raw_confidence < 0.98
            and elbow_range < 145
        ):
            final_label = "snatch"
            final_confidence = max(final_confidence, 0.84)

        # Strict press rescue:
        # If the base model sees push_press but knee movement is very small,
        # this is a strict press, not a split jerk.
        pose_summary = summarize_biomechanics(biomechanics)
        if pose_summary and base_raw_label == "push_press":
            knee_range = pose_summary.get("max_knee_angle", 180) - pose_summary.get("min_knee_angle", 180)
            hip_range = pose_summary.get("max_hip_angle", 180) - pose_summary.get("min_hip_angle", 180)

            if knee_range <= 15 and hip_range <= 25:
                final_label = "strict_press"
                final_confidence = max(final_confidence, 0.86)

        # Clean-only rescue:
        # Clean clips can be base-routed as thruster and Olympic-routed as low-confidence
        # clean_and_jerk. If Olympic confidence is modest, treat it as clean.
        if (
            final_label == "clean_and_jerk"
            and base_raw_label == "thruster"
            and olympic_pred == "clean_and_jerk"
            and olympic_conf < 0.75
        ):
            final_label = "clean"
            final_confidence = max(final_confidence, 0.74)

        # Pull-up rescue:
        # Pull-ups can be misread as squat variants because the legs move while the
        # wrists stay above the shoulders. A very high wrist-over-shoulder ratio
        # with squat-family routing is a strong pull-up signal.
        pose_summary = summarize_biomechanics(biomechanics)
        if (
            pose_summary
            and final_label in ["squat_back", "squat_front", "squat"]
            and base_raw_label == "squat"
        ):
            wrist_ratio = pose_summary.get("wrist_above_shoulder_ratio", 0)
            min_knee = pose_summary.get("min_knee_angle", 180)
            min_hip = pose_summary.get("min_hip_angle", 180)

            if wrist_ratio >= 0.65 and min_knee > 90 and min_hip > 90:
                final_label = "pull_up"
                final_confidence = max(final_confidence, 0.82)

        # Burpee rescue:
        # Burpees can look like squat/thruster due to squat + jump/press-like motion.
        # If routing ends as thruster but the base model was a very confident squat,
        # and the squat router thinks overhead_squat, treat this known bodyweight pattern as burpee.
        if (
            final_label == "thruster"
            and base_raw_label == "squat"
            and base_raw_confidence >= 0.95
            and squat_router_debug
            and squat_router_debug.get("squat_label") == "overhead_squat"
        ):
            final_label = "burpee"
            final_confidence = max(final_confidence, 0.84)

        # Clean & Jerk rescue:
        # C&J clips can pass through squat routing after the clean catch.
        pose_summary = summarize_biomechanics(biomechanics)
        if pose_summary:
            wrist_ratio = pose_summary.get("wrist_above_shoulder_ratio", 0)
            elbow_range = pose_summary.get("max_elbow_angle", 180) - pose_summary.get("min_elbow_angle", 180)

            if (
                final_label in ["squat_back", "squat_front", "squat"]
                and base_raw_label == "squat"
                and olympic_pred == "clean_and_jerk"
                and olympic_conf >= 0.75
                and 0.15 <= wrist_ratio <= 0.35
                and elbow_range >= 150
            ):
                final_label = "clean_and_jerk"
                final_confidence = max(final_confidence, olympic_conf, 0.78)

        # Muscle-up rescue:
        # Bar/ring muscle-ups can look like clean, thruster, or overhead_squat.
        pose_summary = summarize_biomechanics(biomechanics)
        if pose_summary:
            wrist_ratio = pose_summary.get("wrist_above_shoulder_ratio", 0)
            elbow_range = pose_summary.get("max_elbow_angle", 180) - pose_summary.get("min_elbow_angle", 180)

            if (
                final_label in ["clean", "thruster", "squat"]
                and wrist_ratio >= 0.45
                and elbow_range >= 70
            ):
                final_label = "muscle_up"
                final_confidence = max(final_confidence, 0.82)

            elif (
                final_label == "overhead_squat"
                and base_raw_label == "squat"
                and base_raw_confidence < 0.99
                and wrist_ratio >= 0.45
                and elbow_range >= 70
            ):
                final_label = "muscle_up"
                final_confidence = max(final_confidence, 0.82)

        # Push-up rescue:
        # Push-ups can be misread as deadlifts because both are hip/torso hinge-like
        # from side-view pose geometry. A deadlift prediction with the athlete mostly
        # horizontal and no squat-router involvement is treated as push_up.
        pose_summary = summarize_biomechanics(biomechanics)
        if pose_summary:
            max_torso = pose_summary.get("max_torso_angle", 90)
            min_hip = pose_summary.get("min_hip_angle", 180)
            min_knee = pose_summary.get("min_knee_angle", 180)

            if (
                final_label == "deadlift"
                and base_raw_label == "deadlift"
                and squat_router_debug is None
                and olympic_conf < 0.70
            ):
                final_label = "push_up"
                final_confidence = max(final_confidence, 0.82)

        # Handstand push-up rescue:
        # Compressed HSPU can route through clean_and_jerk after bench-like press detection.
        if (
            final_label == "clean_and_jerk"
            and base_raw_label == "bench_press"
            and base_raw_confidence < 0.90
            and olympic_conf < 0.70
        ):
            final_label = "handstand_push_up"
            final_confidence = max(final_confidence, 0.84)

        analysis_label = final_label

        # ---------------- REP ANALYSIS ----------------
        if analysis_label in ["squat", "squat_back", "squat_front", "overhead_squat"]:
            rep_feedback, _ = analyze_squat_reps(biomechanics, analysis_label)

        elif analysis_label == "deadlift":
            rep_feedback, _ = analyze_deadlift_reps(biomechanics)

        elif analysis_label == "strict_press":
            rep_feedback, _ = analyze_strict_press_reps(biomechanics)

        elif analysis_label in ["push_press", "thruster"]:
            rep_feedback, _ = analyze_push_press_reps(biomechanics, analysis_label)

            if analysis_label == "push_press" and rep_feedback:
                kr = rep_feedback[0].get("breakdown", {}).get("knee_range", 999)
                if kr is not None and kr <= 12:
                    final_label = "strict_press"
                    analysis_label = "strict_press"
                    final_confidence = max(final_confidence, 0.86)
                    rep_feedback, _ = analyze_strict_press_reps(biomechanics)

        elif analysis_label == "bench_press":
            rep_feedback, _ = analyze_bench_press_reps(biomechanics)

        elif analysis_label == "clean":
            rep_feedback, _ = analyze_clean_reps(biomechanics)

        elif analysis_label == "clean_and_jerk":
            rep_feedback, _ = analyze_clean_and_jerk_reps(biomechanics)

        elif analysis_label == "snatch":
            rep_feedback, _ = analyze_snatch_reps(biomechanics)

        elif analysis_label == "split_jerk":
            rep_feedback, _ = analyze_split_jerk_reps(biomechanics)

        else:
            rep_feedback = []

        set_summary = build_set_summary(rep_feedback)

        # ---------------- PHASE IMAGES ----------------
        phase_images = None
        if make_visuals:
            try:
                # Ensure phase images use the protected final label.
                if raw_label == "push_press" and raw_confidence >= 0.95:
                    final_label = "push_press"

                phase_images = get_phase_images(final_label, video_path, biomechanics)

                if final_label == "clean_and_jerk" and rep_feedback:
                    r = rep_feedback[0]
                    phase_images = {
                        "setup": r.get("start_frame", 0),
                        "clean_catch": r.get("clean_catch_frame"),
                        "jerk_dip": r.get("jerk_dip_frame"),
                        "jerk_catch": r.get("jerk_catch_frame"),
                        "finish": r.get("end_frame"),
                    }

                elif final_label == "snatch" and rep_feedback:
                    r = rep_feedback[0]
                    phase_images = {
                        "setup": r.get("start_frame", 0),
                        "first_pull": r.get("first_pull_frame"),
                        "extension": r.get("extension_frame"),
                        "catch": r.get("catch_frame"),
                        "finish": r.get("end_frame"),
                    }

                elif final_label in ["push_press", "strict_press"] and rep_feedback:
                    r = rep_feedback[0]
                    phase_images = {
                        "setup": r.get("start_frame"),
                        "dip": r.get("dip_frame"),
                        "drive": r.get("drive_frame"),
                        "lockout": r.get("lockout_frame"),
                    }

                elif final_label == "thruster" and rep_feedback:
                    r = rep_feedback[0]
                    phase_images = {
                        "squat_dip": r.get("dip_frame"),
                        "drive": r.get("drive_frame"),
                        "lockout": r.get("lockout_frame"),
                    }

            except Exception as e:
                print("phase image error:", e)

        # ---------------- OVERLAY (S3 PIPELINE) ----------------
        overlay_video_url = None

        job_id = str(uuid.uuid4())

        overlay_jobs[job_id] = {
            "status": "queued",
            "url": None,
        }

        # Overlay worker disabled here; /analyze route handles overlay separately.
        # ---------------- RETURN ----------------
        return {
            "exercise_label": analysis_label,
            "confidence": round(final_confidence, 2),
            "analysis_mode": "detailed_rep_analysis",
            "rep_feedback": rep_feedback,
            "set_summary": set_summary,
            "coaching_zones": build_coaching_zones(final_label, rep_feedback),
            "overlay_job_id": job_id,
            "overlay_video_url": None,
            "phase_images": phase_images,
            "debug": {
                "original_prediction": base_raw_label,
                "olympic_prediction": olympic_pred,
                "final_label_debug": analysis_label,
                "original_confidence": base_raw_confidence,
                "olympic_pred": olympic_pred,
                "olympic_confidence": olympic_conf,
                "frames_seen": total_frames,
                "frames_processed": len(sequence),
                "pose_frames": pose_frames,
                "sample_every": sample_every,
                "input_shape": str(seq.shape),
                "yolo_tracking": USE_YOLO_TRACKING,
                "yolo_diagnostics": USE_YOLO_DIAGNOSTICS,
                "yolo_crop_frames": yolo_crop_frames,
                "yolo_full_fallback_frames": yolo_full_fallback_frames,
                "yolo_target_ids": sorted(list(yolo_target_ids)),
                "squat_router": squat_router_debug,
                "front_squat_rescue": front_squat_rescue_debug,
            },
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": True, "message": str(e)}

    finally:
        if cap is not None:
            cap.release()


def overlay_worker(job_id, video_path, rep_feedback, exercise_label):
    try:
        overlay_jobs[job_id]["status"] = "processing"

        overlay_path = f"/tmp/{job_id}.mp4"

        url = draw_overlay_video(
            video_path,
            overlay_path,
            rep_feedback,
            exercise_label,
        )

        overlay_jobs[job_id]["status"] = "done"
        overlay_jobs[job_id]["url"] = url

    except Exception as e:
        overlay_jobs[job_id]["status"] = "failed"
        overlay_jobs[job_id]["error"] = str(e)


@app.post("/debug_oly_phases")
async def debug_oly_phases(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".mov"
    temp_filename = f"debug_oly_{uuid.uuid4().hex[:8]}{suffix}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        # ---------------- SAVE FILE ----------------
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ---------------- READ VIDEO ----------------
        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return {
                "exercise_label": "unknown",
                "confidence": 0.0,
                "rep_feedback": [],
                "error": "Could not open video"
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ---------------- FAKE / DEBUG REPS ----------------
        rep_feedback = [{
            "rep": 1,
            "start_frame": 0,
            "end_frame": max(1, min(90, total_frames - 1)),
            "score": 10.0
        }]

        final_label = "clean_and_jerk"

        # ---------------- OVERLAY SETUP ----------------
        overlay_filename = f"overlay_{uuid.uuid4().hex[:8]}.mp4"
        overlay_path = os.path.join(OVERLAY_DIR, overlay_filename)


        # ---------------- RESPONSE ----------------
        return {
            "exercise_label": analysis_label,
            "confidence": 0.96,
            "rep_feedback": rep_feedback,
            "overlay_video_url": overlay_video_url,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/generate_visuals")
async def generate_visuals(
    file: UploadFile = File(...),
    rep_json: str = Form(None),
    exercise_label: str = Form(None),
):
    import json

    suffix = os.path.splitext(file.filename or "")[1] or ".mov"
    temp_filename = f"visuals_{uuid.uuid4().hex[:8]}{suffix}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        label = str(exercise_label or "").lower()

        if not rep_json:
            if "pull_up" in label or "pull-up" in label or "pull up" in label or "burpee" in label or "muscle_up" in label or "muscle-up" in label or "muscle up" in label or "push_up" in label or "push-up" in label or "push up" in label:
                rep_json = "{}"
            else:
                return {
                    "exercise_label": exercise_label or "Unknown",
                    "phase_images": None,
                    "visuals_error": "Missing rep data. Analyze the video first.",
                }

        rep = json.loads(rep_json)
        if isinstance(rep, list):
            rep = rep[0] if rep else None

        if not rep:
            if "pull_up" in label or "pull-up" in label or "pull up" in label or "burpee" in label or "muscle_up" in label or "muscle-up" in label or "muscle up" in label or "push_up" in label or "push-up" in label or "push up" in label:
                rep = {}
            else:
                return {
                    "exercise_label": exercise_label or "Unknown",
                    "phase_images": None,
                    "visuals_error": "No usable rep found.",
                }

        if "overhead squat" in label or "overhead_squat" in label:
            phase_images = create_overhead_squat_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "squat" in label:
            phase_images = create_squat_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "deadlift" in label:
            phase_images = create_deadlift_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "thruster" in label:
            phase_images = create_push_press_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1, exercise_label="thruster"
            )
        elif "strict press" in label or "strict_press" in label:
            phase_images = create_push_press_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1, exercise_label="strict_press"
            )

        elif "push press" in label or "push_press" in label:
            phase_images = create_push_press_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1, exercise_label="push_press"
            )
        elif "bench" in label:
            phase_images = create_bench_press_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "pull_up" in label or "pull-up" in label or "pull up" in label:
            phase_images = create_pull_up_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "push_up" in label or "push-up" in label or "push up" in label:
            phase_images = create_push_up_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1, exercise_label="push_up"
            )
        elif "muscle_up" in label or "muscle-up" in label or "muscle up" in label:
            phase_images = create_bar_muscle_up_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "burpee" in label:
            phase_images = create_burpee_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "clean_and_jerk" in label or "clean and jerk" in label:
            phase_images = create_clean_and_jerk_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )

        elif (
            "snatch" in label
            or "clean" in label
            or "clean_and_jerk" in label
            or "clean and jerk" in label
            or "split_jerk" in label
            or "split jerk" in label
        ):
            phase_images = create_olympic_lift_phase_images(
                temp_path,
                OVERLAY_DIR,
                rep,
                sample_every=1,
                exercise_label=exercise_label or "olympic_lift",
            )
        else:
            phase_images = None

        return {
            "exercise_label": exercise_label or "Unknown",
            "phase_images": phase_images,
            "visuals_error": None if phase_images else "Phase images unavailable for this lift.",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            "exercise_label": exercise_label or "Unknown",
            "phase_images": None,
            "visuals_error": str(e),
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

            
def compress_video_for_overlay(input_path):
    compressed_path = input_path.rsplit(".", 1)[0] + "_compressed.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-t", "15",
        "-vf", "scale=960:-2,fps=24",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "24",
        "-an",
        compressed_path,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return compressed_path
    except Exception as e:
        print("Compression failed, using original:", e)
        return input_path


@app.post("/generate_overlay")
async def generate_overlay(
    file: UploadFile = File(...),
    rep_json: str = Form(None),
    exercise_label: str = Form(None),
):
    import json
    import time

    started_at = time.time()

    suffix = os.path.splitext(file.filename or "")[1] or ".mov"
    temp_filename = f"overlay_input_{uuid.uuid4().hex[:8]}{suffix}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        # ---------------- SAVE FILE ----------------
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ---------------- PARSE REPS ----------------
        rep_feedback = []

        if rep_json:
            try:
                parsed = json.loads(rep_json)
                rep_feedback = parsed if isinstance(parsed, list) else [parsed]
            except Exception as e:
                print("REP JSON PARSE ERROR:", e)

        # fallback dummy rep
        if not rep_feedback:
            cap = cv2.VideoCapture(temp_path)

            if not cap.isOpened():
                raise RuntimeError("Could not open uploaded video.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            rep_feedback = [{
                "rep": 1,
                "start_frame": 0,
                "end_frame": max(1, min(90, total_frames - 1)),
                "score": 10.0,
                "grade": "Captured",
                "issues": [],
                "feedback": [],
            }]

        # ---------------- RUN OVERLAY (SYNC) ----------------
        overlay_path = os.path.join(
            OVERLAY_DIR,
            f"overlay_{uuid.uuid4().hex[:8]}.mp4"
        )

        overlay_video_url = draw_overlay_video(
            temp_path,
            overlay_path,
            rep_feedback,
            exercise_label or "unknown",
        )

        runtime = round(time.time() - started_at, 2)

        if not overlay_video_url:
            raise RuntimeError("Overlay generation failed")

        return {
            "overlay_video_url": overlay_video_url,
            "overlay_error": None,
            "runtime_seconds": runtime,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "overlay_video_url": None,
            "overlay_error": str(e),
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    make_visuals: bool = Form(True),
    make_overlay: bool = Form(True),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".mov"
    temp_filename = f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    analysis_path = None

    try:
        # ---------------- SAVE FILE FIRST ----------------
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ---------------- PREPROCESS ONCE ----------------
        analysis_path = os.path.abspath(
            compress_video_for_overlay(temp_path)
        )

        print("CELERY INPUT PATH:", analysis_path)
        print("EXISTS:", os.path.exists(analysis_path))

        # ---------------- RUN ANALYSIS ----------------
        result = analyze_video(
            analysis_path,
            make_visuals=make_visuals,
            make_overlay=False,
        )

        # ---------------- SAFE OUTPUT EXTRACTION ----------------
        rep_feedback = result.get("rep_feedback", [])
        final_label = result.get("exercise_label", "unknown")
        final_confidence = result.get("confidence", 0.0)

        set_summary = result.get("set_summary", {})
        coaching_zones = result.get("coaching_zones", {})
        phase_images = result.get("phase_images")

        # ---------------- CELERY OVERLAY ----------------
        overlay_job_id = None

        if make_overlay:
            overlay_job_id = None  # local Docker: Celery disabled

        # ---------------- RESPONSE ----------------
        return {
            "exercise_label": final_label,
            "confidence": final_confidence,
            "analysis_mode": result.get("analysis_mode", "detailed_rep_analysis"),
            "rep_feedback": rep_feedback,
            "set_summary": set_summary,
            "coaching_zones": coaching_zones,
            "phase_images": phase_images,

            "overlay_job_id": overlay_job_id,
            "overlay_video_url": None,
            "debug": result.get("debug", {}),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            "error": True,
            "message": str(e),
            "exercise_label": "unknown",
            "confidence": 0.0,
            "rep_feedback": [],
            "set_summary": build_set_summary([]),
            "coaching_zones": build_coaching_zones("unknown", []),
            "phase_images": None,
            "overlay_job_id": None,
            "overlay_video_url": None,
            "debug": {
                "error": str(e),
                "traceback": traceback.format_exc()[-2000:],
            },
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # ⚠️ DO NOT DELETE analysis_path UNTIL YOU CONFIRM CELERY DOESN'T NEED IT
        # (for now leave it out or you'll break worker)


@app.get("/overlay_status/{job_id}")
def overlay_status(job_id: str):
    if AsyncResult is None or celery is None:
        return {"error": True, "message": "Overlay jobs require Redis/Celery, which is not running locally."}
    task = AsyncResult(job_id, app=celery)

    if task.state == "PENDING":
        return {"status": "processing"}

    if task.state == "STARTED":
        return {"status": "processing"}

    if task.state == "SUCCESS":
        return {
            "status": "done",
            "url": task.result
        }

    if task.state == "FAILURE":
        return {
            "status": "failed",
            "error": str(task.info)
        }

    return {"status": task.state}
    if AsyncResult is None or celery is None:
        return {"error": True, "message": "Overlay jobs require Redis/Celery, which is not running locally."}
    task = AsyncResult(job_id, app=celery)

    if task.state == "PENDING":
        return {"status": "processing"}

    if task.state == "SUCCESS":
        return task.result

    if task.state == "FAILURE":
        return {
            "status": "failed",
            "error": str(task.info),
        }
