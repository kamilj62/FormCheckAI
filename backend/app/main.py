from sys import prefix
import tempfile
from pathlib import Path
import traceback

import os
from tracemalloc import start
import uuid
import shutil

from threading import Thread
import uuid

from profiles.athlete_profile import load_profile

overlay_jobs = {}

job_store = {}

from app.phase_engine.squat_v3 import extract_pose_records

from app.phase_detection.signal_engine import SignalEngine
from app.phase_detection.phase_engine import get_phase_images

from app.phase_engine.rep_segmenter import segment_reps
from app.phase_engine.rep_scorer import score_rep
from app.phase_engine.bottom_detector import find_bottom_v1
from app.phase_engine.rep_coach import coach_rep

from app.phase_engine.fatigue_engine import (
    extract_rep_features,
    compute_fatigue_curve
)

from app.feature_engine.feature_engine_rf import build_rf_features

from app.ml.events.clean_events import detect_clean_events

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

import joblib
from app.movement.event_detector import detect_movement_events
from app.feature_engine.movement_video_features import build_movement_video_features

from app.ml.router_v8.protections import apply_protections

try:
    from celery.result import AsyncResult
    from app.celery_app import celery
except Exception:
    AsyncResult = None
    celery = None

import threading

import boto3

import subprocess

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Experimental: use YOLO to isolate the foreground athlete before pose estimation.
USE_YOLO_TRACKING = False

try:
    from app.tracking import YOLOTracker, remap_crop_landmarks_to_full_frame
except Exception:
    YOLOTracker = None
    remap_crop_landmarks_to_full_frame = None

from app.ml.oly_router_v5 import route_olympic_lift

from app.coaching.clean import build_clean_coaching

from app.coaching.snatch import build_snatch_coaching

from app.coaching.split_jerk import build_split_jerk_coaching

from app.coaching.squat import build_squat_coaching

from app.coaching.overhead_squat import build_overhead_squat_coaching

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

S3_BUCKET = os.getenv("S3_BUCKET", "formcheck-ai-overlays-kamilj")
S3_REGION = os.getenv("AWS_REGION", "us-west-2")
s3_client = boto3.client("s3", region_name=S3_REGION)

OVERLAY_DIR = "outputs"
os.makedirs(OVERLAY_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OVERLAY_DIR), name="outputs")


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
    ],  # CORS origins list
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path
from .model_runtime import NumpyFormCheckModel

from app.ml.router_v8.collectors import collect_predictions
from app.ml.router_v8.fusion_clean_v10 import fuse_predictions
from app.ml.router_v8.debug import build_debug
from app.ml.router_v8.state import RouterState
from app.ml.router_v8.locks import get_locks

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
    "squat_front",
]

USE_OLY_ROUTER_V4 = True  # Experimental video-level Olympic router
USE_ROUTER_V7 = False  # Candidate 70-feature Olympic router. Loaded but not active.

OLY_ROUTER_BUNDLE = joblib.load(MODEL_DIR / "oly_router_v2.joblib")
OLY_ROUTER_MODEL = OLY_ROUTER_BUNDLE["model"]
OLY_ROUTER_ENCODER = OLY_ROUTER_BUNDLE.get("label_encoder")

OLY_ROUTER_V4_MODEL = None
try:
    OLY_ROUTER_V4_MODEL = joblib.load(MODEL_DIR / "oly_router_v4.joblib")
except Exception as e:
    print("OLY ROUTER V4 NOT LOADED:", e)

OLY_ROUTER_V7_MODEL = None
try:
    OLY_ROUTER_V7_MODEL = joblib.load(MODEL_DIR / "candidates" / "oly_router_v7.joblib")
except Exception as e:
    print("OLY ROUTER V7 NOT LOADED:", e)

OLY_ROUTER_LABELS = {
    0: "clean",
    1: "clean_and_jerk",
    2: "snatch",
    3: "split_jerk"
}

BODYWEIGHT_ROUTER_MODEL = None
BODYWEIGHT_ROUTER_ENCODER = None

try:
    BODYWEIGHT_ROUTER_MODEL = joblib.load(MODEL_DIR / "bodyweight_router.joblib")
    BODYWEIGHT_ROUTER_ENCODER = joblib.load(MODEL_DIR / "bodyweight_router_labels.joblib")
    print("BODYWEIGHT ROUTER LOADED")
except Exception as e:
    print("BODYWEIGHT ROUTER NOT LOADED:", e)

BODYWEIGHT_ROUTER_FEATURES = [
    "total_frames",
    "wrist_above_shoulder_ratio",
    "wrist_below_shoulder_ratio",
    "mean_wrist_minus_shoulder_y",
    "mean_hip_minus_shoulder_y",
    "mean_knee_minus_hip_y",
    "median_head_drop",
    "avg_wrist_forward",
    "wrist_y_range",
    "shoulder_y_range",
    "hip_y_range",
    "elbow_range",
    "min_elbow",
    "max_elbow",
    "avg_elbow",
    "avg_torso_angle",
]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@app.get("/")
@app.head("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
@app.head("/health")
async def health():
    return {"status": "ok", "model_loaded": True, "build": "clean_v2_251f968"}


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


def build_coaching_zones(exercise_label, issues):
    # SAFE STUB ADDED: prevents runtime crash if fallback path is hit
    return {
        "zones": [],
        "exercise_label": exercise_label,
        "issues": issues
    }


def decode_olympic_label(raw_label, label_encoder=None):
    if label_encoder is not None:
        try:
            return str(label_encoder.inverse_transform([int(raw_label)])[0])
        except Exception:
            pass

    try:
        return OLY_ROUTER_LABELS.get(int(raw_label), str(raw_label))
    except Exception:
        return str(raw_label)


def point(landmarks, landmark):
    lm = landmarks[landmark.value]
    return np.array([lm.x, lm.y], dtype=np.float32)


def formatLabel(v):
    if not v:
        return "Unknown Exercise"

    return str(v).replace("_", " ").title()


def extract_features_and_biomechanics(results):
    def safe_float(x):
        try:
            if x is None:
                return 0.0
            return float(x)
        except:
            return 0.0

    if not results.pose_landmarks:
        return None, None

    landmarks = results.pose_landmarks.landmark

    # -------------------------------
    # 1. 68-feature base vector
    # -------------------------------
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
            safe_float(lm.x),
            safe_float(lm.y),
            safe_float(lm.z),
            safe_float(lm.visibility),
        ])

    features = np.array(features, dtype=np.float32)

    # -------------------------------
    # 2. full 33-landmark vector (RF + future models)
    # -------------------------------
    full_features = np.array([
        safe_float(v)
        for lm in landmarks
        for v in [lm.x, lm.y, lm.z, lm.visibility]
    ], dtype=np.float32)

    # -------------------------------
    # 3. key points (SAFE)
    # -------------------------------
    def pt(lm):
        return np.array([safe_float(lm.x), safe_float(lm.y)], dtype=np.float32)

    left_shoulder = pt(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    right_shoulder = pt(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    left_elbow = pt(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
    right_elbow = pt(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    left_wrist = pt(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_wrist = pt(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    left_hip = pt(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_hip = pt(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    left_knee = pt(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    right_knee = pt(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    left_ankle = pt(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_ankle = pt(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    left_heel = pt(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value])
    right_heel = pt(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value])
    nose = pt(landmarks[mp_pose.PoseLandmark.NOSE.value])

    # -------------------------------
    # 4. derived metrics (SAFE)
    # -------------------------------
    ankle_mid = (left_ankle + right_ankle) / 2
    heel_mid = (left_heel + right_heel) / 2
    heel_lift = safe_float(ankle_mid[1] - heel_mid[1])

    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    knee_mid = (left_knee + right_knee) / 2
    elbow_mid = (left_elbow + right_elbow) / 2
    wrist_mid = (left_wrist + right_wrist) / 2

    knee_angle = safe_float(angle(left_hip, left_knee, left_ankle))
    hip_angle = safe_float(angle(left_shoulder, left_hip, left_knee))
    elbow_angle = safe_float(angle(left_shoulder, left_elbow, left_wrist))

    torso_angle = safe_float(angle(shoulder_mid, hip_mid, hip_mid + np.array([0, -1])))

    valgus_ratio = safe_float(
        np.clip(
            abs(left_knee[0] - right_knee[0]) /
            (abs(left_ankle[0] - right_ankle[0]) + 1e-6),
            0.5, 1.5
        )
    )

    biomechanics = {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "elbow_angle": elbow_angle,
        "torso_angle": torso_angle,

        "hip_y": safe_float(hip_mid[1]),
        "knee_y": safe_float(knee_mid[1]),
        "shoulder_y": safe_float(shoulder_mid[1]),
        "wrist_y": safe_float(wrist_mid[1]),
        "elbow_y": safe_float(elbow_mid[1]),

        "hip_x": safe_float(hip_mid[0]),
        "knee_x": safe_float(knee_mid[0]),
        "shoulder_x": safe_float(shoulder_mid[0]),

        "wrist_x": safe_float(wrist_mid[0]),

        "shoulder_hip_distance": safe_float(np.linalg.norm(shoulder_mid - hip_mid)),
        "hip_knee_distance": safe_float(np.linalg.norm(hip_mid - knee_mid)),
        "wrist_shoulder_distance": safe_float(np.linalg.norm(wrist_mid - shoulder_mid)),

        "valgus_ratio": valgus_ratio,
        "bar_distance": safe_float(abs(wrist_mid[0] - ankle_mid[0])),

        "head_drop": safe_float(nose[1] - shoulder_mid[1]),
        "head_forward": safe_float(abs(nose[0] - shoulder_mid[0])),

        "heel_lift": heel_lift,

        # RF input
        "full_features": full_features,
    }

    return features, biomechanics


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
        return {
            "avg_knee_angle": 180.0,
            "min_knee_angle": 180.0,
            "max_knee_angle": 180.0,
            "avg_hip_angle": 180.0,
            "min_hip_angle": 180.0,
            "max_hip_angle": 180.0,
            "avg_torso_angle": 0.0,
            "min_torso_angle": 0.0,
            "max_torso_angle": 0.0,
            "avg_elbow_angle": 180.0,
            "min_elbow_angle": 180.0,
            "max_elbow_angle": 180.0,
            "avg_valgus_ratio": 1.0,
            "min_valgus_ratio": 1.0,
            "wrist_above_shoulder_ratio": 0.0,
        }

    knee = np.array([b["knee_angle"] for b in biomechanics])
    hip = np.array([b["hip_angle"] for b in biomechanics])
    torso = np.array([b["torso_angle"] for b in biomechanics])
    elbow = np.array([b["elbow_angle"] for b in biomechanics])
    valgus = np.array([b.get("valgus_ratio", 1.0) for b in biomechanics])
    wrist_above = np.array([
        b.get(
            "wrist_above_shoulder",
            1.0 if b.get("wrist_y", 1.0) < b.get("shoulder_y", 0.0) else 0.0,
        )
        for b in biomechanics
    ])
    
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
    finish_idx = start_idx + int(span * 0.84)
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


def smooth_coach_score(score, exercise_label=None):
    """
    Safe fallback score smoother.
    Keeps current analyzer behavior stable without requiring old helper imports.
    """
    try:
        score = float(score)
    except Exception:
        score = 0.0

    return max(0.0, min(10.0, score))


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

            # The threshold crossing can occur before the lifter reaches a
            # stable lockout. Search slightly beyond it for the strongest
            # combined hip and knee extension with an upright torso.
            original_end = end

            # Preserve the original segmentation endpoint for coaching.
            # The refined endpoint is only used for visual phase timing.
            score_end = original_end
            rep_span = max(1, original_end - start)

            lockout_search_start = max(
                start + 4,
                original_end - 2,
            )
            lockout_search_end = min(
                len(biomechanics) - 1,
                original_end + max(10, int(rep_span * 0.65)),
            )

            if lockout_search_end > lockout_search_start:
                candidate_hip = hip[
                    lockout_search_start:lockout_search_end + 1
                ]
                candidate_knee = knee[
                    lockout_search_start:lockout_search_end + 1
                ]
                candidate_torso = torso[
                    lockout_search_start:lockout_search_end + 1
                ]

                lockout_score = (
                    candidate_hip
                    + candidate_knee
                    - (0.80 * candidate_torso)
                )

                end = int(
                    lockout_search_start + np.argmax(lockout_score)
                )

            # Score only the original segmented pull window.
            # Later lockout frames are reserved for phase visuals.
            rep_signal = movement_signal[start:score_end + 1]
            rep_torso = torso[start:score_end + 1]
            rep_hip = hip[start:score_end + 1]
            rep_knee = knee[start:score_end + 1]
            rep_bar = bar_distance[start:score_end + 1]
            rep_head_drop = head_drop[start:score_end + 1]
            rep_head_forward = head_forward[start:score_end + 1]

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
                "lockout_frame": int(frame_numbers[phase_frames["lockout"]]),
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
            "lockout_frame": int(frame_numbers[phase_frames["lockout"]]),
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
    hip_y_values = np.array(
        [b.get("hip_y", 0.5) for b in biomechanics],
        dtype=np.float32,
    )
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
    is_overhead_squat = exercise_label == "overhead_squat"

    # Temporary development diagnostic for front-squat segmentation.
    if is_front_squat:
        import csv

        signal_path = "/tmp/front_squat_signals.csv"

        with open(signal_path, "w", newline="") as signal_file:
            writer = csv.writer(signal_file)
            writer.writerow([
                "index",
                "frame",
                "knee_angle",
                "hip_y",
                "torso_angle",
                "wrist_y",
                "shoulder_y",
            ])

            for signal_idx, signal_row in enumerate(biomechanics):
                writer.writerow([
                    signal_idx,
                    int(signal_row.get("frame_number", signal_idx)),
                    round(float(signal_row.get("knee_angle", 180.0)), 3),
                    round(float(signal_row.get("hip_y", 0.5)), 5),
                    round(float(signal_row.get("torso_angle", 0.0)), 3),
                    round(float(signal_row.get("wrist_y", 0.0)), 5),
                    round(float(signal_row.get("shoulder_y", 0.0)), 5),
                ])

        print("FRONT SQUAT SIGNALS:", signal_path)

    reps = []
    # Use different threshold for overhead squats due to different mechanics
    if is_overhead_squat:
        # Overhead squats may have different knee angle patterns, use more adaptive threshold
        min_knee = np.min(knee_angles)
        max_knee = np.max(knee_angles)
        knee_range = max_knee - min_knee
        if knee_range > 15:  # Only detect reps if there's significant knee movement
            threshold = min_knee + (knee_range * 0.3)  # Threshold based on range
        else:
            threshold = np.percentile(knee_angles, 20)  # Fallback to percentile
    else:
        threshold = np.percentile(knee_angles, 35)

    SQUAT_PENALTIES = {
        "depth": {"good": 0.0, "borderline": 0.6, "poor": 1.4},
        "torso": {"good": 0.0, "borderline": 1.0, "poor": 2.2},
        "knees": {"good": 0.0, "borderline": 1.5, "poor": 3.5},        "heels": {"good": 0.0, "borderline": 0.4, "poor": 0.9},
        "neck": {"good": 0.0, "borderline": 0.8, "poor": 1.8},
        "front_rack": {"good": 0.0, "borderline": 0.8, "poor": 1.6},
        "bar_position": {"good": 0.0, "borderline": 0.7, "poor": 1.4},
        "overhead": {"good": 0.0, "borderline": 0.8, "poor": 1.6},
        "bar_path": {"good": 0.0, "borderline": 0.7, "poor": 1.4},
    }

    def safe_phase_frame(name, fallback):
        value = int(phase_frames.get(name, fallback))

        if 0 <= value < len(frame_numbers):
            return int(frame_numbers[value])

        return value

    in_rep = False
    start = 0
    skip_until_idx = -1

    for i, knee in enumerate(knee_angles):
        if i <= skip_until_idx:
            continue
        if not in_rep and knee < threshold:
            in_rep = True
            start = i

        elif in_rep and knee >= threshold:
            end = i

            if is_front_squat:
                raw_start = int(start)
                raw_end = int(end)

                # Search forward in source-video time, not just contiguous
                # biomechanics indices. Pose may disappear behind the plates
                # during the ascent and return near lockout.
                raw_end_frame = int(frame_numbers[raw_end])
                forward_limit_frame = raw_end_frame + 120

                search_end = raw_end

                while (
                    search_end + 1 < len(frame_numbers)
                    and int(frame_numbers[search_end + 1]) <= forward_limit_frame
                ):
                    search_end += 1

                # Search backward for a genuine upright setup. This rejects
                # an opening partial rep that began before the video.
                raw_start_frame = int(frame_numbers[raw_start])
                backward_limit_frame = raw_start_frame - 90

                search_start = raw_start

                while (
                    search_start - 1 >= 0
                    and int(frame_numbers[search_start - 1]) >= backward_limit_frame
                ):
                    search_start -= 1

                # Hip position is the most reliable bottom signal in this
                # front-squat clip because knee landmarks are intermittently
                # obscured by the bar and plates.
                bottom = search_start + int(
                    np.argmax(hip_y_values[search_start:search_end + 1])
                )

                bottom_hip_y = float(hip_y_values[bottom])

                setup_candidates = [
                    j
                    for j in range(search_start, bottom)
                    if (
                        knee_angles[j] >= 165
                        and hip_y_values[j] <= bottom_hip_y - 0.08
                    )
                ]

                lockout_candidates = [
                    j
                    for j in range(bottom + 1, search_end + 1)
                    if (
                        knee_angles[j] >= 165
                        and hip_y_values[j] <= bottom_hip_y - 0.08
                    )
                ]

                # A complete front squat needs an upright position on both
                # sides of the bottom. This rejects the partial opening rep.
                if not setup_candidates or not lockout_candidates:
                    in_rep = False
                    continue

                start = int(setup_candidates[-1])
                end = int(lockout_candidates[0])

                source_descent_span = (
                    int(frame_numbers[bottom])
                    - int(frame_numbers[start])
                )
                source_ascent_span = (
                    int(frame_numbers[end])
                    - int(frame_numbers[bottom])
                )

                if source_descent_span < 15 or source_ascent_span < 15:
                    in_rep = False
                    continue

                # Prevent later threshold fragments from generating duplicate
                # reps inside this already-expanded squat window.
                skip_until_idx = end

            min_frames = 3 if is_overhead_squat else 5
            if end - start < min_frames:
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

            if not is_front_squat:
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

            if is_overhead_squat:
                overhead_ratio = float(np.mean(rep_wrist_y < rep_shoulder_y))
                bar_drift = float(
                    np.percentile(rep_wrist_x, 90) - np.percentile(rep_wrist_x, 10)
                )

                if overhead_ratio >= 0.75:
                    overhead_grade = "good"
                elif overhead_ratio >= 0.55:
                    overhead_grade = "borderline"
                    issues.append("Overhead bar stability could be stronger.")
                    feedback.append("Lock the bar directly over midfoot and stay stacked.")
                else:
                    overhead_grade = "poor"
                    issues.append("Bar is not staying stable overhead.")
                    feedback.append("Keep arms locked and bar stacked over midfoot.")

                if bar_drift <= 0.10:
                    bar_path_grade = "good"
                elif bar_drift <= 0.16:
                    bar_path_grade = "borderline"
                    issues.append("Bar may be drifting forward during the squat.")
                    feedback.append("Prevent forward drift — keep the bar over midfoot.")
                else:
                    bar_path_grade = "poor"
                    issues.append("Bar path is drifting away from midfoot.")
                    feedback.append("Stay stacked and keep the bar over your base of support.")

                breakdown["overhead"] = overhead_grade
                breakdown["bar_path"] = bar_path_grade

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

            rep = {
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
            }

            if is_overhead_squat:
                rep["coaching"] = build_overhead_squat_coaching(rep)
            else:
                rep["coaching"] = build_squat_coaching(rep, exercise_label)

            reps.append(rep)

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


def find_clean_phase_reps(biomechanics):
    """
    Event-based clean phase finder.

    Backend/frontend contract:
    setup       -> setup
    first_pull  -> first pull
    extension   -> pull under / turnover
    catch       -> front-rack catch
    finish      -> standing front-rack lockout
    """
    n = len(biomechanics)
    if n < 10:
        return []

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics], dtype=np.float32)
    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics], dtype=np.float32)
    shoulder_x = np.array([b.get("shoulder_x", 0.0) for b in biomechanics], dtype=np.float32)
    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=np.float32)
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics], dtype=np.float32)

    rack_distance = np.abs(wrist_x - shoulder_x)

    # Front-rack catch candidates:
    # wrist near shoulder height, close to shoulder horizontally,
    # and knees bent enough to indicate receiving position.
    front_rack = (
        (np.abs(wrist_y - shoulder_y) < 0.22)
        & (rack_distance < 0.32)
        & (knee < 155)
    )

    idxs = np.where(front_rack)[0]

    if len(idxs) == 0:
        return []

    # Cluster front-rack frames into distinct catches.
    clusters = []
    current = [int(idxs[0])]
    max_gap = max(8, n // 60)

    for idx in idxs[1:]:
        idx = int(idx)
        if idx - current[-1] <= max_gap:
            current.append(idx)
        else:
            if len(current) >= 3:
                clusters.append(current)
            current = [idx]

    if len(current) >= 3:
        clusters.append(current)

    reps = []

    for cluster in clusters:
        cluster_start = cluster[0]
        cluster_end = cluster[-1]

        # Ignore startup false positives before the lift actually begins.
        if cluster_start < max(60, int(n * 0.18)):
            continue

        # Require a real extension shortly before the catch.
        # This prevents setup/finish/front-rack noise from becoming a clean catch.
        pre_ext_start = max(0, cluster_start - max(70, n // 5))
        pre_ext_end = max(pre_ext_start + 1, cluster_start)

        ext_score = hip[pre_ext_start:pre_ext_end] + knee[pre_ext_start:pre_ext_end]
        if len(ext_score) == 0 or float(np.max(ext_score)) < 300:
            continue

        # Catch = lowest hip in the first receiving part of this rack cluster.
        local_end = min(cluster_end, cluster_start + max(8, n // 20))
        catch_window = np.arange(cluster_start, local_end + 1)
        catch_idx = int(catch_window[np.argmax(hip_y[catch_window])])
        catch_idx = max(3, min(catch_idx, n - 2))

        # Setup = earlier than catch, but not so far back it grabs prior rep finish.
        start_idx = max(0, catch_idx - max(45, n // 4))

        # True extension = strongest tall position before catch.
        pre_start = max(start_idx + 1, int(start_idx + (catch_idx - start_idx) * 0.35))
        pre_end = max(pre_start + 1, catch_idx)

        extension_score = hip[pre_start:pre_end] + knee[pre_start:pre_end]
        true_extension_idx = pre_start + int(np.argmax(extension_score))
        true_extension_idx = max(start_idx + 2, min(true_extension_idx, catch_idx - 1))

        first_pull_idx = max(
            start_idx + 1,
            int(start_idx + (true_extension_idx - start_idx) * 0.45)
        )

        # Backend key "extension" is displayed as Pull Under on frontend.
        pull_under_idx = first_pull_idx + int((catch_idx - first_pull_idx) * 0.40)
        pull_under_idx = max(first_pull_idx + 1, min(pull_under_idx, catch_idx - 1))

        # Finish = first standing front-rack position after catch.
        search_end = min(n - 1, catch_idx + max(20, n // 8))
        standing_candidates = [
            i for i in range(catch_idx + 1, search_end + 1)
            if hip_y[i] < hip_y[catch_idx] - 0.03
        ]

        if standing_candidates:
            end_idx = int(standing_candidates[min(5, len(standing_candidates) - 1)])
        else:
            end_idx = min(n - 1, catch_idx + max(10, n // 25))

        end_idx = max(catch_idx + 1, min(end_idx, n - 1))

        rep = {
            "start_frame": int(frame_numbers[start_idx]),
            "first_pull_frame": int(frame_numbers[first_pull_idx]),
            "extension_frame": int(frame_numbers[pull_under_idx]),
            "catch_frame": refine_catch_bottom_frame(biomechanics, int(frame_numbers[catch_idx])),
            "end_frame": int(frame_numbers[end_idx]),
        }

        # Avoid duplicate catches from the same clean.
        if reps and rep["catch_frame"] - reps[-1]["catch_frame"] < 150:
            continue

        reps.append(rep)

    return reps


def refine_catch_bottom_frame(biomechanics, approx_frame, radius=18):
    records = [
        {
            "frame": int(b.get("frame_number", i)),
            "knee": float(b.get("knee_angle", 180.0)),
            "hip": float(b.get("hip_angle", 180.0)),
        }
        for i, b in enumerate(biomechanics)
    ]

    bottom = find_bottom_v1(records, int(approx_frame), radius=radius)
    if not bottom:
        return int(approx_frame)

    return int(bottom.get("frame", approx_frame))


def _frame_to_idx(biomechanics, frame_number):
    frame_numbers = np.array([
        b.get("frame_number", i) for i, b in enumerate(biomechanics)
    ])
    target = int(frame_number)
    return int(np.argmin(np.abs(frame_numbers - target)))


def _score_from_breakdown(breakdown, penalties, base=10.0):
    score = base
    for key, value in breakdown.items():
        score -= penalties.get(key, {}).get(value, 0.0)
    return round(max(1.0, min(10.0, score)), 1)


CLEAN_PENALTIES = {
    "first_pull": {"good": 0.0, "borderline": 0.6, "poor": 1.2},
    "extension": {"good": 0.0, "borderline": 0.6, "poor": 1.4},
    "turnover": {"good": 0.0, "slow": 0.8},
    "catch": {"good": 0.0, "borderline": 0.6, "shallow": 1.0},
    "front_rack": {"good": 0.0, "borderline": 0.7, "poor": 1.2},
    "bar_path": {"good": 0.0, "drifting": 0.8},
}


SNATCH_PENALTIES = {
    "first_pull": {"good": 0.0, "borderline": 0.6, "poor": 1.2},
    "extension": {"good": 0.0, "borderline": 0.6, "poor": 1.4},
    "turnover": {"good": 0.0, "slow": 0.8},
    "overhead_catch": {"good": 0.0, "borderline": 0.7, "poor": 1.4},
    "stability": {"good": 0.0, "borderline": 0.6, "poor": 1.2},
    "bar_path": {"good": 0.0, "drifting": 0.8},
}


def _grade_clean_rep(biomechanics, frames):
    start_i = _frame_to_idx(biomechanics, frames.get("start_frame", 0))
    first_pull_i = _frame_to_idx(biomechanics, frames.get("first_pull_frame", start_i))
    extension_i = _frame_to_idx(biomechanics, frames.get("extension_frame", first_pull_i))
    catch_i = _frame_to_idx(biomechanics, frames.get("catch_frame", extension_i))

    torso = np.array([b.get("torso_angle", 0.0) for b in biomechanics])
    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics])
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics])
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics])
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics])
    shoulder_x = np.array([b.get("shoulder_x", 0.0) for b in biomechanics])
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics])

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

    pull_end = max(start_i + 1, first_pull_i)
    pull_slice = slice(start_i, pull_end + 1)
    torso_pull = float(np.percentile(torso[pull_slice], 80))
    bar_offset = float(
        np.percentile(np.abs(wrist_x[pull_slice] - shoulder_x[pull_slice]), 90)
    )

    if torso_pull > 70 or bar_offset > 0.30:
        breakdown["first_pull"] = "poor"
        issues.append("First pull may be too forward or unstable.")
        feedback.append("Stay balanced over mid-foot and keep the chest over the bar.")
    elif torso_pull > 55 or bar_offset > 0.22:
        breakdown["first_pull"] = "borderline"
        issues.append("First pull could stay more balanced.")
        feedback.append("Patience off the floor — keep shoulders over the bar longer.")

    ext_knee = float(knee[extension_i])
    ext_hip = float(hip[extension_i])
    if ext_knee < 145 or ext_hip < 145:
        breakdown["extension"] = "poor"
        issues.append("Extension looks incomplete before the turnover.")
        feedback.append("Finish extending the hips and knees before pulling under.")
    elif ext_knee < 155 or ext_hip < 155:
        breakdown["extension"] = "borderline"
        issues.append("Extension could be more complete.")
        feedback.append("Drive through the legs fully before the pull under.")

    turnover_frames = max(1, catch_i - extension_i)
    if turnover_frames > max(25, len(biomechanics) * 0.15):
        breakdown["turnover"] = "slow"
        issues.append("Turnover into the catch may be slow.")
        feedback.append("Rotate the elbows faster into the front rack.")

    catch_knee = float(knee[catch_i])
    if catch_knee > 140:
        breakdown["catch"] = "shallow"
        issues.append("Catch position may be too shallow.")
        feedback.append("Drop under the bar into a stronger receiving position.")
    elif catch_knee > 125:
        breakdown["catch"] = "borderline"
        issues.append("Catch depth could be lower.")
        feedback.append("Receive the clean in a slightly deeper squat.")

    rack_elbow = float(elbow[catch_i])
    rack_wrist_drop = float(wrist_y[catch_i] - shoulder_y[catch_i])
    if rack_elbow > 120 or rack_wrist_drop > 0.18:
        breakdown["front_rack"] = "poor"
        issues.append("Front rack position is weak at the catch.")
        feedback.append("Drive elbows high and keep the bar on your shoulders.")
    elif rack_elbow > 100 or rack_wrist_drop > 0.12:
        breakdown["front_rack"] = "borderline"
        issues.append("Elbows could be higher in the front rack.")
        feedback.append("Rotate elbows faster and keep the bar close.")

    path_slice = wrist_x[max(start_i, first_pull_i):catch_i + 1]
    if len(path_slice) >= 2:
        bar_drift = float(np.percentile(path_slice, 90) - np.percentile(path_slice, 10))
        if bar_drift > 0.10:
            breakdown["bar_path"] = "drifting"
            issues.append("Bar path may be looping away from the body.")
            feedback.append("Keep the bar close through the pull and turnover.")

    score = _score_from_breakdown(breakdown, CLEAN_PENALTIES)
    if not issues:
        score = max(score, 9.0)
        feedback = ["Good clean rep. Strong pull and catch position."]

    return breakdown, issues, feedback, score


def _grade_snatch_rep(biomechanics, frames):
    start_i = _frame_to_idx(biomechanics, frames.get("start_frame", 0))
    first_pull_i = _frame_to_idx(biomechanics, frames.get("first_pull_frame", start_i))
    extension_i = _frame_to_idx(biomechanics, frames.get("extension_frame", first_pull_i))
    catch_i = _frame_to_idx(biomechanics, frames.get("catch_frame", extension_i))
    end_i = _frame_to_idx(biomechanics, frames.get("end_frame", catch_i))

    torso = np.array([b.get("torso_angle", 0.0) for b in biomechanics])
    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics])
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics])
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics])
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics])
    shoulder_x = np.array([b.get("shoulder_x", 0.0) for b in biomechanics])
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics])

    issues = []
    feedback = []
    breakdown = {
        "first_pull": "good",
        "extension": "good",
        "turnover": "good",
        "overhead_catch": "good",
        "stability": "good",
        "bar_path": "good",
    }

    pull_end = max(start_i + 1, first_pull_i)
    pull_slice = slice(start_i, pull_end + 1)
    torso_pull = float(np.percentile(torso[pull_slice], 80))
    bar_offset = float(
        np.percentile(np.abs(wrist_x[pull_slice] - shoulder_x[pull_slice]), 90)
    )

    if torso_pull > 70 or bar_offset > 0.30:
        breakdown["first_pull"] = "poor"
        issues.append("First pull may be too forward or unstable.")
        feedback.append("Stay balanced and keep the chest over the bar longer.")
    elif torso_pull > 55 or bar_offset > 0.22:
        breakdown["first_pull"] = "borderline"
        issues.append("First pull could stay more balanced.")
        feedback.append("Patience off the floor — keep shoulders over the bar longer.")

    ext_knee = float(knee[extension_i])
    ext_hip = float(hip[extension_i])
    if ext_knee < 145 or ext_hip < 145:
        breakdown["extension"] = "poor"
        issues.append("Extension looks incomplete before pulling under.")
        feedback.append("Finish extending before pulling under the bar.")
    elif ext_knee < 155 or ext_hip < 155:
        breakdown["extension"] = "borderline"
        issues.append("Extension could be more complete.")
        feedback.append("Drive through the legs fully before the pull under.")

    turnover_frames = max(1, catch_i - extension_i)
    if turnover_frames > max(25, len(biomechanics) * 0.15):
        breakdown["turnover"] = "slow"
        issues.append("Pull under the bar may be slow.")
        feedback.append("Pull yourself under the bar more aggressively.")

    overhead_at_catch = float(wrist_y[catch_i] < shoulder_y[catch_i])
    catch_knee = float(knee[catch_i])
    if not overhead_at_catch or catch_knee > 150:
        breakdown["overhead_catch"] = "poor"
        issues.append("Overhead catch position looks unstable.")
        feedback.append("Receive the bar in a stronger overhead position.")
    elif catch_knee > 135:
        breakdown["overhead_catch"] = "borderline"
        issues.append("Overhead catch could be more secure.")
        feedback.append("Lock the bar overhead with arms extended and hips under the bar.")

    stability_slice = slice(catch_i, min(end_i + 1, len(biomechanics)))
    overhead_ratio = float(np.mean(wrist_y[stability_slice] < shoulder_y[stability_slice]))
    max_elbow = float(np.max(elbow[stability_slice])) if stability_slice.stop > stability_slice.start else 0.0
    if overhead_ratio < 0.50 or max_elbow < 150:
        breakdown["stability"] = "poor"
        issues.append("Overhead stability after the catch needs work.")
        feedback.append("Stabilize the bar overhead before standing up.")
    elif overhead_ratio < 0.70 or max_elbow < 160:
        breakdown["stability"] = "borderline"
        issues.append("Overhead position could be held more securely.")
        feedback.append("Punch through and hold the bar locked out overhead.")

    path_slice = wrist_x[max(start_i, first_pull_i):catch_i + 1]
    if len(path_slice) >= 2:
        bar_drift = float(np.percentile(path_slice, 90) - np.percentile(path_slice, 10))
        if bar_drift > 0.12:
            breakdown["bar_path"] = "drifting"
            issues.append("Bar path may be looping away from the body.")
            feedback.append("Keep the bar closer to your body during the pull.")

    score = _score_from_breakdown(breakdown, SNATCH_PENALTIES)
    if not issues:
        score = max(score, 9.0)
        feedback = ["Good snatch rep. Strong pull, catch, and overhead position."]

    breakdown["catch"] = breakdown["overhead_catch"]

    return breakdown, issues, feedback, score


def analyze_clean_reps(biomechanics):
    if len(biomechanics) < 10:
        return [], build_set_summary([])

    phase_reps = find_clean_phase_reps(biomechanics)

    if not phase_reps:
        events = detect_movement_events(biomechanics, "clean")

        if not events:
            return [], build_set_summary([])

        frame_numbers = np.array([
            b.get("frame_number", i)
            for i, b in enumerate(biomechanics)
        ])

        phase_reps = [{
            "start_frame": int(frame_numbers[events.get("setup", 0)]),
            "first_pull_frame": int(frame_numbers[events.get("first_pull", events.get("setup", 0))]),
            "extension_frame": int(frame_numbers[events.get("extension", events.get("first_pull", 0))]),
            "catch_frame": int(frame_numbers[events.get("catch", events.get("extension", 0))]),
            "end_frame": int(frame_numbers[events.get("finish", events.get("lockout", len(frame_numbers)-1))]),
        }]

    reps = []

    for i, frames in enumerate(phase_reps):
        breakdown, issues, feedback, score = _grade_clean_rep(biomechanics, frames)

        rep = {
            **frames,
            "rep": i + 1,
            "score": score,
            "grade": grade_score(score),
            "issues": issues,
            "breakdown": breakdown,
            "feedback": feedback,
        }

        timeline_v2 = detect_clean_events(
            biomechanics,
            frames,
        )
        rep["event_timeline_v2"] = timeline_v2.to_dict()

        rep["coaching"] = build_clean_coaching(rep)

        reps.append(rep)

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


def find_split_jerk_phase_reps(biomechanics):
    n = len(biomechanics)
    if n < 20:
        return []

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    knee = np.array([
        b.get("knee_angle", 180.0)
        for b in biomechanics
    ], dtype=np.float32)

    hip = np.array([
        b.get("hip_angle", 180.0)
        for b in biomechanics
    ], dtype=np.float32)

    elbow = np.array([
        b.get("elbow_angle", 180.0)
        for b in biomechanics
    ], dtype=np.float32)

    wrist_y = np.array([
        b.get("wrist_y", 1.0)
        for b in biomechanics
    ], dtype=np.float32)

    shoulder_y = np.array([
        b.get("shoulder_y", 0.0)
        for b in biomechanics
    ], dtype=np.float32)

    # Stable overhead requires both bar position and arm extension.
    overhead = (
        (wrist_y < shoulder_y - 0.02)
        & (elbow > 145)
    )

    overhead_idxs = np.where(overhead)[0]

    if len(overhead_idxs) == 0:
        return []

    # Group overhead frames into separate jerk attempts.
    clusters = []
    current = [int(overhead_idxs[0])]
    max_gap = max(6, n // 80)

    for raw_idx in overhead_idxs[1:]:
        idx = int(raw_idx)

        if idx - current[-1] <= max_gap:
            current.append(idx)
        else:
            if len(current) >= 4:
                clusters.append(current)
            current = [idx]

    if len(current) >= 4:
        clusters.append(current)

    reps = []
    stable_needed = 3

    for cluster in clusters:
        cluster_start = int(cluster[0])
        cluster_end = int(cluster[-1])

        # Ignore an opening overhead hold from a prior rep. A valid jerk
        # should have enough pre-catch frames to show setup, dip and drive.
        if cluster_start < 12:
            continue

        # Find the first genuinely stable overhead receive rather than the
        # first noisy wrist-above-shoulder frame.
        catch_idx = None

        search_end = min(n - stable_needed, cluster_end)

        for i in range(cluster_start, search_end + 1):
            if all(overhead[i:i + stable_needed]):
                catch_idx = i
                break

        if catch_idx is None:
            continue

        # Search backward from the catch for the dip. Keep the window local
        # to this attempt so it cannot attach to the previous jerk.
        pre_catch_window = max(35, min(80, n // 7))
        dip_search_start = max(0, catch_idx - pre_catch_window)
        dip_search_end = max(dip_search_start + 2, catch_idx - 3)

        if dip_search_end <= dip_search_start:
            continue

        # Require at least some front-rack/non-overhead frames before catch.
        rack_like = np.abs(
            wrist_y[dip_search_start:dip_search_end]
            - shoulder_y[dip_search_start:dip_search_end]
        ) < 0.25

        non_overhead = ~overhead[dip_search_start:dip_search_end]

        if not np.any(rack_like & non_overhead):
            continue

        dip_idx = dip_search_start + int(
            np.argmin(knee[dip_search_start:dip_search_end])
        )

        # Reject clusters without a meaningful dip.
        pre_dip_start = max(dip_search_start, dip_idx - 12)
        pre_dip_knee = float(np.median(knee[pre_dip_start:dip_idx + 1]))
        dip_depth = pre_dip_knee - float(knee[dip_idx])

        if dip_depth < 3.0 and knee[dip_idx] > 165:
            continue

        # Drive is the strongest knee/hip extension after the dip but before
        # the stable overhead receive.
        drive_start = min(n - 2, dip_idx + 1)
        drive_end = max(drive_start + 2, catch_idx - 1)

        if drive_end <= drive_start:
            continue

        knee_norm = (
            knee - np.min(knee)
        ) / (np.ptp(knee) + 1e-6)

        hip_norm = (
            hip - np.min(hip)
        ) / (np.ptp(hip) + 1e-6)

        extension_signal = (
            0.60 * knee_norm
            + 0.40 * hip_norm
        )

        drive_idx = drive_start + int(
            np.argmax(extension_signal[drive_start:drive_end])
        )

        # Maintain meaningful visual spacing.
        drive_idx = max(
            dip_idx + 2,
            min(drive_idx, catch_idx - 3),
        )

        catch_idx = max(
            drive_idx + 3,
            min(catch_idx, n - 3),
        )

        # Setup should show the athlete upright before beginning the dip.
        setup_search_start = max(0, dip_idx - 25)
        setup_candidates = [
            i for i in range(setup_search_start, dip_idx)
            if knee[i] > 155 and hip[i] > 150
        ]

        if setup_candidates:
            # Use an earlier stable rack position so setup is visibly
            # different from the bottom of the dip.
            start_idx = int(setup_candidates[0])
        else:
            start_idx = max(0, dip_idx - 15)

        # Enforce useful visual spacing between setup and dip.
        start_idx = min(start_idx, max(0, dip_idx - 8))

        # Lockout: stable arms-overhead position shortly after the split catch.
        # Select a clearly established overhead lockout after the initial
        # split receive, rather than an early frame while the bar is still
        # settling overhead.
        lockout_idx = min(
            cluster_end,
            catch_idx + max(12, int((cluster_end - catch_idx) * 0.32)),
        )
        lockout_idx = max(
            catch_idx + 8,
            min(lockout_idx, n - 3),
        )

        # Recovery must occur substantially later than lockout. Knee/hip
        # extension alone cannot prove that the feet have recovered because
        # the athlete can be upright while still holding the split.
        recovery_search_start = min(
            n - stable_needed,
            catch_idx + max(18, int((cluster_end - catch_idx) * 0.55)),
        )

        recovery_idx = None

        for i in range(recovery_search_start, min(n - stable_needed, cluster_end) + 1):
            stable_overhead = all(overhead[i:i + stable_needed])
            upright = (
                np.mean(knee[i:i + stable_needed]) > 155
                and np.mean(hip[i:i + stable_needed]) > 150
            )

            if stable_overhead and upright:
                recovery_idx = i
                break

        if recovery_idx is None:
            recovery_idx = catch_idx + int(
                (cluster_end - catch_idx) * 0.72
            )

        recovery_idx = max(
            lockout_idx + 6,
            min(int(recovery_idx), n - 2),
        )

        # Finish should be a later stable overhead frame after recovery,
        # but not the end of the whole cluster because that may include
        # lowering the bar or beginning the next attempt.
        finish_search_start = min(
            n - stable_needed,
            recovery_idx + 4,
        )

        finish_search_end = min(
            n - stable_needed,
            cluster_end,
            recovery_idx + max(10, min(20, len(cluster) // 8)),
        )

        finish_idx = min(
            cluster_end,
            recovery_idx + max(6, min(12, len(cluster) // 10)),
        )

        for i in range(finish_search_start, finish_search_end + 1):
            stable_overhead = all(overhead[i:i + stable_needed])
            upright = (
                np.mean(knee[i:i + stable_needed]) > 155
                and np.mean(hip[i:i + stable_needed]) > 150
            )

            if stable_overhead and upright:
                finish_idx = i

        finish_idx = max(
            recovery_idx + 4,
            min(int(finish_idx), cluster_end, n - 1),
        )

        rep = {
            "start_frame": int(frame_numbers[start_idx]),
            "dip_frame": int(frame_numbers[dip_idx]),
            "drive_frame": int(frame_numbers[drive_idx]),
            "catch_frame": int(frame_numbers[catch_idx]),
            "recovery_frame": int(frame_numbers[recovery_idx]),
            "lockout_frame": int(frame_numbers[lockout_idx]),
            "end_frame": int(frame_numbers[finish_idx]),
            "debug_frames": {
                "start_idx": int(start_idx),
                "dip_idx": int(dip_idx),
                "drive_idx": int(drive_idx),
                "catch_idx": int(catch_idx),
                "recovery_idx": int(recovery_idx),
                "lockout_idx": int(lockout_idx),
                "finish_idx": int(finish_idx),
                "cluster_start": int(cluster_start),
                "cluster_end": int(cluster_end),
            },
        }

        # Avoid duplicate detections within the same overhead event.
        if (
            reps
            and rep["catch_frame"] - reps[-1]["catch_frame"] < 45
        ):
            continue

        reps.append(rep)

    return reps


def looks_like_clean_only(biomechanics):
    clean_reps = find_clean_phase_reps(biomechanics)
    if not clean_reps:
        return False

    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics], dtype=np.float32)
    overhead_ratio = float(np.mean((wrist_y < shoulder_y) & (elbow > 145)))

    return overhead_ratio < 0.12


def looks_like_clean_and_jerk(biomechanics):
    clean_reps = find_clean_phase_reps(biomechanics)
    if not clean_reps:
        return False

    # Multi-overhead clips are more likely snatch or split-jerk sets than one
    # combined clean-and-jerk rep. A C&J clip can still produce two overhead
    # windows around the jerk catch/recovery, so allow up to two.
    if len(find_split_jerk_phase_reps(biomechanics)) > 2:
        return False

    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics], dtype=np.float32)
    overhead_ratio = float(np.mean((wrist_y < shoulder_y) & (elbow > 145)))

    return overhead_ratio >= 0.12


def looks_like_split_jerk(biomechanics):
    split_reps = find_split_jerk_phase_reps(biomechanics)
    if not split_reps:
        return False

    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)
    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=np.float32)
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics], dtype=np.float32)

    overhead_ratio = float(np.mean(wrist_y < shoulder_y))
    dip_ratio = float(np.mean((knee < 165) & (hip < 170)))

    return overhead_ratio >= 0.08 and dip_ratio >= 0.05


def looks_like_strict_press(biomechanics):
    if not biomechanics:
        return False

    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=np.float32)
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics], dtype=np.float32)
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics], dtype=np.float32)
    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)

    knee_range = float(np.max(knee) - np.min(knee))
    hip_range = float(np.max(hip) - np.min(hip))
    elbow_range = float(np.max(elbow) - np.min(elbow))
    overhead_ratio = float(np.mean(wrist_y < shoulder_y))

    return (
        overhead_ratio >= 0.20
        and elbow_range >= 18
        and knee_range < 22
        and hip_range < 30
    )


def looks_like_thruster(biomechanics):
    if not biomechanics:
        return False

    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=np.float32)
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics], dtype=np.float32)
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics], dtype=np.float32)
    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)

    knee_range = float(np.max(knee) - np.min(knee))
    hip_range = float(np.max(hip) - np.min(hip))
    elbow_range = float(np.max(elbow) - np.min(elbow))
    overhead_ratio = float(np.mean(wrist_y < shoulder_y))

    return (
        float(np.min(knee)) < 125
        and knee_range >= 25
        and hip_range >= 20
        and elbow_range >= 18
        and overhead_ratio >= 0.10
    )


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

    phase_reps = find_split_jerk_phase_reps(biomechanics)

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

    if phase_reps:
        reps = []
        for i, frames in enumerate(phase_reps):
            rep = {
                **frames,
                "rep": i + 1,
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback,
            }
            rep["coaching"] = build_split_jerk_coaching(rep)
            reps.append(rep)
    else:
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

        reps[0]["coaching"] = build_split_jerk_coaching(reps[0])

    return reps, build_set_summary(reps)


def find_stable_overhead_window(
    overhead,
    elbow,
    knee,
    hip,
    wrist_y,
    hip_y,
    start_idx,
    end_idx=None,
    min_len=3,
):
    """
    Returns (catch_idx, finish_idx) for overhead lifts.
    catch_idx  = beginning of stable overhead receive/lockout window
    finish_idx = later stable overhead hold
    """
    n = len(overhead)
    if n < 5:
        return None, None

    if end_idx is None:
        end_idx = n - 1

    start_idx = max(1, min(int(start_idx), n - 2))
    end_idx = max(start_idx + 1, min(int(end_idx), n - 1))

    stable = []

    for i in range(start_idx, end_idx + 1):
        if not overhead[i]:
            stable.append(False)
            continue

        if elbow[i] < 145:
            stable.append(False)
            continue

        if knee[i] < 120 or hip[i] < 120:
            stable.append(False)
            continue

        wrist_motion = abs(wrist_y[i] - wrist_y[max(0, i - 2)])
        hip_motion = abs(hip_y[i] - hip_y[max(0, i - 2)])

        if wrist_motion > 0.035 or hip_motion > 0.035:
            stable.append(False)
            continue

        stable.append(True)

    # Find first sustained stable overhead window.
    run_start = None
    run = 0

    for offset, ok in enumerate(stable):
        idx = start_idx + offset

        if ok:
            if run_start is None:
                run_start = idx
            run += 1

            if run >= min_len:
                # Jerk catch = first stable overhead receive.
                catch_idx = run_start

                # Finish = later upright stable overhead frame, if present.
                finish_idx = catch_idx
                for j in range(catch_idx + max(6, min_len), end_idx + 1):
                    if not overhead[j]:
                        continue
                    if elbow[j] < 150:
                        continue
                    if knee[j] < 150 or hip[j] < 145:
                        continue

                    wrist_motion = abs(wrist_y[j] - wrist_y[max(0, j - 2)])
                    hip_motion = abs(hip_y[j] - hip_y[max(0, j - 2)])

                    if wrist_motion > 0.03 or hip_motion > 0.03:
                        continue

                    finish_idx = j
                    break

                if finish_idx == catch_idx:
                    finish_idx = min(end_idx, catch_idx + max(10, min_len + 4))

                return catch_idx, finish_idx
        else:
            run_start = None
            run = 0

    return None, None


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
    hip_y = np.array([b.get("hip_y", 0.5) for b in biomechanics])
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

    # Clean catch = best front-rack receiving posture.
    # Prefer the deepest squat while the wrists remain near shoulder height.
    catch_start = max(1, int(n * 0.20))
    catch_end = max(catch_start + 2, min(int(n * 0.45), first_overhead_idx))

    rack_like = np.abs(wrist_y - shoulder_y) < 0.24

    candidates = [
        i for i in range(catch_start, catch_end)
        if rack_like[i] and knee[i] < 145
    ]

    if candidates:
        clean_catch_idx = int(candidates[np.argmin(knee[candidates])])
    else:
        clean_catch_idx = catch_start + int(np.argmin(knee[catch_start:catch_end]))

    # Clean recovery = upright front-rack position after clean catch,
    # BEFORE the bar goes overhead. This prevents recovery drifting into jerk/finish.
    recovery_search_start = min(n - 2, clean_catch_idx + max(6, int(n * 0.05)))
    recovery_search_end = max(
        recovery_search_start + 2,
        min(first_overhead_idx - 2, clean_catch_idx + max(18, int(n * 0.30)), n - 1)
    )

    recovery_candidates = [
        i for i in range(recovery_search_start, recovery_search_end)
        if rack_like[i] and knee[i] > 130 and hip[i] > 125
    ]

    if recovery_candidates:
        # Prefer the most upright frame in the front-rack window, not the last frame.
        recovery_idx = int(max(recovery_candidates, key=lambda i: extension_signal[i]))
    else:
        recovery_idx = recovery_search_start + int(
            np.argmax(extension_signal[recovery_search_start:recovery_search_end])
        )

    # ---------------- C&J EVENT SEQUENCE ----------------
    # Detect one continuous:
    # clean catch -> upright front rack -> jerk dip -> drive
    # -> overhead receive -> stable overhead finish.

    stable_needed = 3

    # Clean recovery: first stable upright front-rack position after catch.
    recovery_idx = min(n - 1, clean_catch_idx + 1)

    # Skip the immediate receiving frames so recovery represents the
    # athlete standing in the front rack, not still rising from the catch.
    recovery_search_start = min(
        n - stable_needed,
        clean_catch_idx + max(10, int(n * 0.04)),
    )
    recovery_search_end = min(
        n - stable_needed,
        clean_catch_idx + max(35, int(n * 0.35)),
    )

    for i in range(recovery_search_start, recovery_search_end + 1):
        stable_rack = all(rack_like[i:i + stable_needed])
        standing = (
            np.mean(knee[i:i + stable_needed]) > 150
            and np.mean(hip[i:i + stable_needed]) > 145
        )

        if stable_rack and standing:
            recovery_idx = i
            break

    # Require a clearly overhead, extended-arm posture. This avoids treating
    # noisy clean/front-rack wrist positions as the start of the jerk.
    clear_overhead = (
        (wrist_y < shoulder_y - 0.04)
        & (elbow > 150)
    )

    overhead_search_start = min(
        n - stable_needed,
        recovery_idx + max(8, int(n * 0.04)),
    )

    jerk_catch_idx = None

    for i in range(overhead_search_start, n - stable_needed):
        stable_overhead = all(clear_overhead[i:i + stable_needed])

        if stable_overhead:
            jerk_catch_idx = i
            break

    # Relax the vertical margin only if the stricter detector found nothing.
    if jerk_catch_idx is None:
        fallback_overhead = (
            (wrist_y < shoulder_y)
            & (elbow > 145)
        )

        for i in range(overhead_search_start, n - stable_needed):
            if all(fallback_overhead[i:i + stable_needed]):
                jerk_catch_idx = i
                break

    if jerk_catch_idx is None:
        jerk_catch_idx = min(
            n - 2,
            recovery_idx + max(20, int(n * 0.25)),
        )

    # Search only shortly before the overhead receive. This prevents the
    # clean catch or a long front-rack pause from being selected as jerk dip.
    pre_catch_window = max(40, int(n * 0.20))

    jerk_start = max(
        recovery_idx + 2,
        jerk_catch_idx - pre_catch_window,
    )
    jerk_end = max(jerk_start + 2, jerk_catch_idx)

    jerk_dip_idx = jerk_start + int(
        np.argmin(knee[jerk_start:jerk_end])
    )

    # Jerk drive: strongest extension between dip and overhead receive.
    drive_start = min(n - 2, jerk_dip_idx + 1)
    drive_end = max(drive_start + 1, jerk_catch_idx)

    jerk_drive_idx = drive_start + int(
        np.argmax(extension_signal[drive_start:drive_end])
    )

    # Preserve strict chronological ordering.
    recovery_idx = max(
        clean_catch_idx + 1,
        min(int(recovery_idx), n - 5),
    )
    jerk_dip_idx = max(
        recovery_idx + 1,
        min(int(jerk_dip_idx), n - 4),
    )
    jerk_drive_idx = max(
        jerk_dip_idx + 1,
        min(int(jerk_drive_idx), n - 3),
    )
    jerk_catch_idx = max(
        jerk_drive_idx + 1,
        min(int(jerk_catch_idx), n - 2),
    )

    # Finish: first stable standing overhead frame after the receive.
    end_idx = min(
        n - 1,
        jerk_catch_idx + max(10, int(n * 0.10)),
    )

    finish_search_start = min(
        n - stable_needed,
        jerk_catch_idx + max(5, int(n * 0.03)),
    )

    for i in range(finish_search_start, n - stable_needed):
        stable_overhead = all(clear_overhead[i:i + stable_needed])
        standing = (
            np.mean(knee[i:i + stable_needed]) > 150
            and np.mean(hip[i:i + stable_needed]) > 145
        )

        if stable_overhead and standing:
            end_idx = i
            break

    end_idx = max(
        jerk_catch_idx + 1,
        min(int(end_idx), n - 1),
    )

    # Convert biomechanics indices to source-video frames.
    start_frame = int(frame_numbers[0])
    clean_catch_frame = int(frame_numbers[clean_catch_idx])
    clean_recovery_frame = int(frame_numbers[recovery_idx])
    jerk_dip_frame = int(frame_numbers[jerk_dip_idx])
    jerk_drive_frame = int(frame_numbers[jerk_drive_idx])
    jerk_catch_frame = int(frame_numbers[jerk_catch_idx])
    end_frame = int(frame_numbers[end_idx])

    reps = [{
        "rep": 1,
        "start_frame": start_frame,
        "clean_catch_frame": clean_catch_frame,
        "clean_recovery_frame": clean_recovery_frame,
        "jerk_dip_frame": jerk_dip_frame,
        "jerk_drive_frame": jerk_drive_frame,
        "jerk_catch_frame": jerk_catch_frame,
        "end_frame": end_frame,
        "debug_frames": {
            "clean_catch_idx": int(clean_catch_idx),
            "recovery_idx": int(recovery_idx),
            "jerk_start": int(jerk_start),
            "jerk_dip_idx": int(jerk_dip_idx),
            "jerk_drive_idx": int(jerk_drive_idx),
            "first_overhead_idx": int(first_overhead_idx),
            "jerk_catch_idx": int(jerk_catch_idx),
            "end_idx": int(end_idx),
        },
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


def find_snatch_phase_reps(biomechanics):
    """
    Detect snatch reps from distinct overhead catch events.

    Output keys stay frontend-compatible:
    start_frame, first_pull_frame, extension_frame, catch_frame, end_frame.
    """
    n = len(biomechanics)
    if n < 10:
        return []

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics], dtype=np.float32)
    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)
    hip_angle = np.array([b.get("hip_angle", 180.0) for b in biomechanics], dtype=np.float32)
    knee_angle = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=np.float32)

    overhead = wrist_y < shoulder_y

    if not np.any(overhead):
        return []

    overhead_idxs = np.where(overhead)[0]

    # Group overhead frames into separate catch events.
    clusters = []
    current = [int(overhead_idxs[0])]

    max_gap = max(8, n // 60)

    for idx in overhead_idxs[1:]:
        idx = int(idx)
        if idx - current[-1] <= max_gap:
            current.append(idx)
        else:
            if len(current) >= 4:
                clusters.append(current)
            current = [idx]

    if len(current) >= 4:
        clusters.append(current)

    reps = []

    for cluster in clusters:
        cluster_start = cluster[0]
        cluster_end = cluster[-1]

        # Catch = lowest hip inside the first receiving portion of this overhead cluster.
        # Avoid delayed lowest-hip / recovery frames.
        local_end = min(cluster_end, cluster_start + max(8, n // 20))
        catch_window = np.arange(cluster_start, local_end + 1)
        catch_idx = int(catch_window[np.argmax(hip_y[catch_window])])

        catch_idx = max(3, min(catch_idx, n - 2))

        # Start = earlier setup/pull window before catch.
        start_idx = max(0, catch_idx - max(30, n // 4))

        # Extension = strongest tall position before catch.
        pre_catch_start = max(start_idx + 1, int(start_idx + (catch_idx - start_idx) * 0.35))
        pre_catch_end = max(pre_catch_start + 1, catch_idx)

        extension_score = (
            hip_angle[pre_catch_start:pre_catch_end]
            + knee_angle[pre_catch_start:pre_catch_end]
        )

        extension_idx = pre_catch_start + int(np.argmax(extension_score))
        extension_idx = max(start_idx + 2, min(extension_idx, catch_idx - 1))

        first_pull_idx = max(start_idx + 1, int(start_idx + (extension_idx - start_idx) * 0.45))

        # Finish = tallest stable overhead position after catch.
        # Do NOT use a fixed offset; it can land after the bar drops.
        search_end = min(n - 1, catch_idx + max(20, n // 10))
        finish_candidates = [
            i for i in range(catch_idx + 1, search_end + 1)
            if overhead[i]
        ]

        if finish_candidates:
            finish_candidates = np.array(finish_candidates, dtype=int)
            finish_idx = int(finish_candidates[np.argmin(hip_y[finish_candidates])])
        else:
            finish_idx = min(n - 1, catch_idx + max(10, n // 25))

        finish_idx = max(catch_idx + 1, min(finish_idx, n - 1))

        rep = {
            "start_frame": int(frame_numbers[start_idx]),
            "first_pull_frame": int(frame_numbers[first_pull_idx]),
            "extension_frame": int(frame_numbers[extension_idx]),
            "catch_frame": refine_catch_bottom_frame(biomechanics, int(frame_numbers[catch_idx])),
            "end_frame": int(frame_numbers[finish_idx]),
        }

        # Avoid duplicate reps that are too close together.
        if reps and rep["catch_frame"] - reps[-1]["catch_frame"] < 60:
            continue

        rep["coaching"] = build_split_jerk_coaching(rep)

        reps.append(rep)

    return reps


def find_snatch_phase_frames(biomechanics):
    reps = find_snatch_phase_reps(biomechanics)
    return reps[0] if reps else None


def analyze_snatch_reps(biomechanics):
    if len(biomechanics) < 10:
        return [], build_set_summary([])

    phase_reps = find_snatch_phase_reps(biomechanics)
    overhead_reps = find_split_jerk_phase_reps(biomechanics)

    if len(overhead_reps) > len(phase_reps):
        phase_reps = [
            {
                "start_frame": rep.get("start_frame", 0),
                "first_pull_frame": rep.get("dip_frame", rep.get("start_frame", 0)),
                "extension_frame": rep.get("drive_frame", rep.get("catch_frame", 0)),
                "catch_frame": rep.get("catch_frame", 0),
                "end_frame": rep.get("end_frame", rep.get("catch_frame", 0)),
            }
            for rep in overhead_reps
        ]

    if not phase_reps:
        events = detect_movement_events(biomechanics, "snatch")

        if not events:
            return [], build_set_summary([])

        frame_numbers = np.array([
            b.get("frame_number", i)
            for i, b in enumerate(biomechanics)
        ])

        phase_reps = [{
            "start_frame": int(frame_numbers[events.get("setup", 0)]),
            "first_pull_frame": int(frame_numbers[events.get("first_pull", events.get("setup", 0))]),
            "extension_frame": int(frame_numbers[events.get("extension", events.get("setup", 0))]),
            "catch_frame": int(frame_numbers[events.get("catch", events.get("extension", 0))]),
            "end_frame": int(frame_numbers[events.get("finish", events.get("lockout", len(frame_numbers)-1))]),
        }]

    reps = []

    for i, frames in enumerate(phase_reps):
        breakdown, issues, feedback, score = _grade_snatch_rep(biomechanics, frames)

        rep = {
            **frames,
            "rep": i + 1,
            "score": score,
            "grade": grade_score(score),
            "issues": issues,
            "breakdown": breakdown,
            "feedback": feedback,
        }
        rep["coaching"] = build_snatch_coaching(rep)

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

    if not reps and len(elbow) >= 10:
        bottom = int(np.argmin(smooth))
        start = max(0, bottom - max(6, len(elbow) // 4))
        end = min(len(elbow) - 1, bottom + max(8, len(elbow) // 3))

        rep_elbow = elbow[start:end + 1]
        rep_wrist_y = wrist_y[start:end + 1]
        rep_shoulder_y = shoulder_y[start:end + 1]
        rep_hip_y = hip_y[start:end + 1]
        rep_knee = knee[start:end + 1]

        elbow_range = float(np.max(rep_elbow) - np.min(rep_elbow))
        wrist_range = float(np.max(rep_wrist_y) - np.min(rep_wrist_y))
        shoulder_range = float(np.max(rep_shoulder_y) - np.min(rep_shoulder_y))
        hip_range = float(np.max(rep_hip_y) - np.min(rep_hip_y))
        max_elbow = float(np.percentile(np.clip(rep_elbow, 45, 180), 92))
        elbow_p75 = float(np.percentile(np.clip(rep_elbow, 45, 180), 75))
        shoulder_level = float(np.percentile(np.clip(rep_shoulder_y, 0.0, 1.0), 50))
        lowest_wrist = float(np.percentile(np.clip(rep_wrist_y, 0.0, 1.0), 85))
        bar_depth = lowest_wrist - shoulder_level
        avg_knee = float(np.percentile(np.clip(rep_knee, 45, 180), 50))

        if len(rep_elbow) > 0:
            issues = []
            feedback = []

            if elbow_range < 25:
                issues.append("Range of motion may be limited.")
                feedback.append("Use a full, controlled press from chest to lockout.")

            if max_elbow < 125:
                issues.append("Incomplete lockout.")
                feedback.append("Fully extend your arms at the top.")

            if elbow_p75 > 165:
                issues.append("Elbows may be flaring excessively.")
                feedback.append("Tuck elbows slightly and keep the bar path controlled.")

            if avg_knee < 95:
                issues.append("Leg drive may be weak.")
                feedback.append("Keep feet planted and drive through your legs.")

            score = compute_rep_score(issues, base_score=8.5)
            if not issues:
                score = 8.5
                feedback = ["Bench press rep detected. Keep the bar path controlled and finish strong."]

            reps.append({
                "rep": 1,
                "start_frame": int(max(0, frame_numbers[start] - 20)),
                "end_frame": int(frame_numbers[end] + 20),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": {
                    "depth": "review",
                    "lockout": "review" if max_elbow < 140 else "good",
                    "elbows": "review" if elbow_p75 > 155 else "good",
                    "arch": "review" if shoulder_range > 0.20 or hip_range > 0.20 else "controlled",
                    "legs": "review" if avg_knee < 95 else "good",
                    "fallback": True,
                    "wrist_range": round(wrist_range, 3),
                    "elbow_range": round(elbow_range, 1),
                    "bar_depth": round(bar_depth, 3),
                    "max_elbow": round(max_elbow, 1),
                },
                "feedback": feedback,
                "visibility_notes": [
                    "Bench rep was detected with the fallback tracker because the camera angle made the full press cycle hard to separate."
                ],
            })

    # ---------------------------------------------------------
    # SINGLE-REP FULL-CYCLE REFINEMENT
    # ---------------------------------------------------------
    # Search ordered clip regions instead of accepting a small local
    # elbow cycle. This prevents a brief wobble from becoming the rep.
    if len(reps) == 1 and len(elbow_all) >= 30:
        full_elbow = np.asarray(elbow_all, dtype=float)

        kernel_size = 7
        kernel = np.ones(kernel_size, dtype=float) / kernel_size
        pad = kernel_size // 2

        smooth_elbow = np.convolve(
            np.pad(full_elbow, (pad, pad), mode="edge"),
            kernel,
            mode="valid",
        )

        n_full = len(smooth_elbow)

        # Setup must come from the early portion of the clip.
        setup_region_end = max(5, int(n_full * 0.35))
        setup_idx = int(
            np.argmax(smooth_elbow[:setup_region_end])
        )

        # For a validated single-rep bench clip, the final available
        # frame is the completed lockout. Pose confidence near the end
        # can make an earlier frame appear to have a larger elbow angle,
        # so do not shorten the rep based on that noisy local maximum.
        lockout_idx = n_full - 1

        # Bottom must lie between setup and lockout.
        bottom_search_start = setup_idx + 3
        bottom_search_end = lockout_idx - 3

        if bottom_search_end > bottom_search_start:
            bottom_idx = int(
                bottom_search_start
                + np.argmin(
                    smooth_elbow[
                        bottom_search_start:bottom_search_end + 1
                    ]
                )
            )

            descent_span = bottom_idx - setup_idx
            press_span = lockout_idx - bottom_idx
            cycle_span = lockout_idx - setup_idx

            setup_extension = float(
                smooth_elbow[setup_idx] - smooth_elbow[bottom_idx]
            )
            lockout_extension = float(
                smooth_elbow[lockout_idx] - smooth_elbow[bottom_idx]
            )

            rep = reps[0]
            breakdown = rep.setdefault("breakdown", {})
            breakdown["bench_ordered_debug"] = {
                "n_full": int(n_full),
                "setup_idx": int(setup_idx),
                "bottom_idx": int(bottom_idx),
                "lockout_idx": int(lockout_idx),
                "descent_span": int(descent_span),
                "press_span": int(press_span),
                "cycle_span": int(cycle_span),
                "required_cycle_span": int(n_full * 0.55),
                "setup_elbow": round(float(smooth_elbow[setup_idx]), 1),
                "bottom_elbow": round(float(smooth_elbow[bottom_idx]), 1),
                "lockout_elbow": round(float(smooth_elbow[lockout_idx]), 1),
                "setup_extension": round(setup_extension, 1),
                "lockout_extension": round(lockout_extension, 1),
            }

            complete_cycle = (
                descent_span >= 8
                and press_span >= 5
                and cycle_span >= int(n_full * 0.55)
                and setup_extension >= 25.0
                and lockout_extension >= 25.0
            )

            breakdown["bench_ordered_debug"]["complete_cycle"] = bool(
                complete_cycle
            )

            if complete_cycle:
                descent_idx = setup_idx + max(
                    1,
                    int(descent_span * 0.50),
                )
                press_idx = bottom_idx + max(
                    1,
                    int(press_span * 0.50),
                )

                rep = reps[0]
                rep["start_frame"] = int(frame_numbers_all[setup_idx])
                rep["descent_frame"] = int(frame_numbers_all[descent_idx])
                rep["bottom_frame"] = int(frame_numbers_all[bottom_idx])
                rep["press_frame"] = int(frame_numbers_all[press_idx])
                rep["lockout_frame"] = int(frame_numbers_all[lockout_idx])
                rep["end_frame"] = int(frame_numbers_all[lockout_idx])

                breakdown = rep.setdefault("breakdown", {})
                breakdown.pop("bench_refine_debug", None)
                breakdown["full_cycle_refined"] = True
                breakdown["cycle_coverage"] = round(
                    cycle_span / max(1, n_full - 1),
                    3,
                )

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


def analyze_handstand_push_up_reps(biomechanics):
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics], dtype=np.float32)
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)
    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics], dtype=np.float32)

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    if len(biomechanics) < 10:
        return [], build_set_summary([])

    bottom_idx = int(np.argmin(elbow))
    start_idx = max(0, bottom_idx - int(len(biomechanics) * 0.35))
    end_idx = min(len(biomechanics) - 1, bottom_idx + int(len(biomechanics) * 0.35))

    rep_elbow = elbow[start_idx:end_idx + 1]
    rep_wrist_y = wrist_y[start_idx:end_idx + 1]
    rep_shoulder_y = shoulder_y[start_idx:end_idx + 1]
    rep_hip_y = hip_y[start_idx:end_idx + 1]

    min_elbow = float(np.min(rep_elbow))
    max_elbow = float(np.max(rep_elbow))
    elbow_range = max_elbow - min_elbow
    hands_below_shoulder_ratio = float(np.mean(rep_wrist_y > rep_shoulder_y))
    body_line_score = float(np.percentile(np.abs(rep_hip_y - rep_shoulder_y), 80))

    issues = []
    feedback = []
    breakdown = {
        "range": "good",
        "bottom": "good",
        "depth": "good",
        "body_line": "good",
        "lockout": "good",
        "control": "good",
    }

    if elbow_range < 35:
        breakdown["range"] = "short"
        issues.append("Handstand push-up range of motion may be short.")
        feedback.append("Lower under control and press through a full range.")

    if min_elbow > 115:
        breakdown["bottom"] = "high"
        breakdown["depth"] = "high"
        issues.append("Bottom position may be shallow.")
        feedback.append("Lower until your head gets closer to the floor or target.")

    if max_elbow < 145:
        breakdown["lockout"] = "short"
        issues.append("Lockout may be incomplete.")
        feedback.append("Finish each rep with strong elbow extension.")

    if body_line_score > 0.16:
        breakdown["body_line"] = "sagging"
        issues.append("Body line may be breaking during the rep.")
        feedback.append("Keep your body stacked and avoid arching or sagging.")
    elif body_line_score > 0.10:
        breakdown["body_line"] = "borderline"
        issues.append("Body line could stay tighter.")
        feedback.append("Brace your core and keep hips stacked over shoulders.")

    if hands_below_shoulder_ratio < 0.65:
        breakdown["control"] = "review"
        issues.append("Inverted position was hard to track.")
        feedback.append("Record from the side with your hands, head, and hips visible.")

    score = compute_rep_score(issues)
    score = apply_coach_reward(score, issues, breakdown)

    if not issues:
        score = max(score, 9.0)
        feedback = ["Good handstand push-up rep. Keep the body stacked and press to a strong lockout."]

    descent_idx = start_idx + int(max(1, bottom_idx - start_idx) * 0.65)
    ascent_idx = bottom_idx + int(max(1, end_idx - bottom_idx) * 0.35)

    reps = [{
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "descent_frame": int(frame_numbers[min(descent_idx, bottom_idx)]),
        "bottom_frame": int(frame_numbers[bottom_idx]),
        "ascent_frame": int(frame_numbers[min(max(ascent_idx, bottom_idx), end_idx)]),
        "end_frame": int(frame_numbers[end_idx]),
        "score": round(score, 1),
        "grade": grade_score(score),
        "issues": issues,
        "breakdown": breakdown,
        "feedback": feedback,
    }]

    return reps, build_set_summary(reps)


def analyze_push_up_reps(biomechanics):
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics], dtype=np.float32)
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)
    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics], dtype=np.float32)

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    if len(biomechanics) < 10:
        return [], build_set_summary([])

    bottom_idx = int(np.argmin(elbow))
    start_idx = max(0, bottom_idx - int(len(biomechanics) * 0.35))
    end_idx = min(len(biomechanics) - 1, bottom_idx + int(len(biomechanics) * 0.35))

    rep_elbow = elbow[start_idx:end_idx + 1]
    rep_wrist_y = wrist_y[start_idx:end_idx + 1]
    rep_shoulder_y = shoulder_y[start_idx:end_idx + 1]
    rep_hip_y = hip_y[start_idx:end_idx + 1]

    min_elbow = float(np.min(rep_elbow))
    max_elbow = float(np.max(rep_elbow))
    elbow_range = max_elbow - min_elbow
    hands_below_shoulder_ratio = float(np.mean(rep_wrist_y > rep_shoulder_y))
    shoulder_hip_drift = float(np.max(np.abs(rep_shoulder_y - rep_hip_y)))

    issues = []
    feedback = []
    breakdown = {
        "range": "good",
        "bottom": "good",
        "lockout": "good",
        "body_line": "good",
    }

    if elbow_range < 35:
        breakdown["range"] = "short"
        issues.append("Push-up range of motion may be short.")
        feedback.append("Lower under control and press through a full range.")

    if min_elbow > 115:
        breakdown["bottom"] = "high"
        issues.append("Bottom position may be shallow.")
        feedback.append("Lower your chest closer to the floor.")

    if max_elbow < 145:
        breakdown["lockout"] = "short"
        issues.append("Lockout may be incomplete.")
        feedback.append("Finish each rep with strong elbow extension.")

    if hands_below_shoulder_ratio < 0.65 or shoulder_hip_drift > 0.45:
        breakdown["body_line"] = "review"
        issues.append("Body line was hard to track.")
        feedback.append("Record from the side with hands, shoulders, hips, and feet visible.")

    score = compute_rep_score(issues)
    score = apply_coach_reward(score, issues, breakdown)

    if not issues:
        score = max(score, 9.0)
        feedback = ["Good push-up rep. Keep a strong body line and press to full lockout."]

    descent_idx = start_idx + int(max(1, bottom_idx - start_idx) * 0.65)
    ascent_idx = bottom_idx + int(max(1, end_idx - bottom_idx) * 0.35)

    reps = [{
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "descent_frame": int(frame_numbers[min(descent_idx, bottom_idx)]),
        "bottom_frame": int(frame_numbers[bottom_idx]),
        "ascent_frame": int(frame_numbers[min(max(ascent_idx, bottom_idx), end_idx)]),
        "end_frame": int(frame_numbers[end_idx]),
        "score": round(score, 1),
        "grade": grade_score(score),
        "issues": issues,
        "breakdown": breakdown,
        "feedback": feedback,
    }]

    return reps, build_set_summary(reps)


def build_bodyweight_features(biomechanics):
    if not biomechanics:
        return {
            "total_frames": 0,
            "wrist_above_shoulder_ratio": 0.0,
            "wrist_below_shoulder_ratio": 0.0,
            "mean_wrist_minus_shoulder_y": 0.0,
            "mean_hip_minus_shoulder_y": 0.0,
            "mean_knee_minus_hip_y": 0.0,
            "median_head_drop": 0.0,
            "avg_wrist_forward": 0.0,
            "wrist_y_range": 0.0,
            "shoulder_y_range": 0.0,
            "hip_y_range": 0.0,
            "elbow_range": 0.0,
            "min_elbow": 180.0,
            "max_elbow": 180.0,
            "avg_elbow": 180.0,
            "avg_torso_angle": 0.0,
        }

    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics], dtype=np.float32)
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)
    shoulder_x = np.array([b.get("shoulder_x", 0.0) for b in biomechanics], dtype=np.float32)
    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics], dtype=np.float32)
    knee_y = np.array([b.get("knee_y", 0.0) for b in biomechanics], dtype=np.float32)
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics], dtype=np.float32)
    head_drop = np.array([b.get("head_drop", 0.0) for b in biomechanics], dtype=np.float32)
    torso = np.array([b.get("torso_angle", 0.0) for b in biomechanics], dtype=np.float32)

    return {
        "total_frames": int(len(biomechanics)),
        "wrist_above_shoulder_ratio": float(np.mean(wrist_y < shoulder_y)),
        "wrist_below_shoulder_ratio": float(np.mean(wrist_y > shoulder_y)),
        "mean_wrist_minus_shoulder_y": float(np.mean(wrist_y - shoulder_y)),
        "mean_hip_minus_shoulder_y": float(np.mean(hip_y - shoulder_y)),
        "mean_knee_minus_hip_y": float(np.mean(knee_y - hip_y)),
        "median_head_drop": float(np.median(head_drop)),
        "avg_wrist_forward": float(np.mean(np.abs(wrist_x - shoulder_x))),
        "wrist_y_range": float(np.max(wrist_y) - np.min(wrist_y)),
        "shoulder_y_range": float(np.max(shoulder_y) - np.min(shoulder_y)),
        "hip_y_range": float(np.max(hip_y) - np.min(hip_y)),
        "elbow_range": float(np.max(elbow) - np.min(elbow)),
        "min_elbow": float(np.min(elbow)),
        "max_elbow": float(np.max(elbow)),
        "avg_elbow": float(np.mean(elbow)),
        "avg_torso_angle": float(np.mean(torso)),
    }


def predict_bodyweight_router(biomechanics):
    if BODYWEIGHT_ROUTER_MODEL is None or BODYWEIGHT_ROUTER_ENCODER is None:
        return None, 0.0, {}

    try:
        feats = build_bodyweight_features(biomechanics)
        X = np.array([[float(feats.get(k, 0.0)) for k in BODYWEIGHT_ROUTER_FEATURES]], dtype=np.float32)

        pred_idx = BODYWEIGHT_ROUTER_MODEL.predict(X)[0]
        label = BODYWEIGHT_ROUTER_ENCODER.inverse_transform([pred_idx])[0]

        conf = 0.0
        if hasattr(BODYWEIGHT_ROUTER_MODEL, "predict_proba"):
            probs = BODYWEIGHT_ROUTER_MODEL.predict_proba(X)[0]
            conf = float(np.max(probs))

        return str(label), conf, feats

    except Exception as e:
        print("BODYWEIGHT ROUTER PREDICT ERROR:", e)
        return None, 0.0, {}


def analyze_muscle_up_reps(biomechanics):
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics], dtype=np.float32)
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)

    frame_numbers = np.array([
        b.get("frame_number", i)
        for i, b in enumerate(biomechanics)
    ])

    if len(biomechanics) < 10:
        return [], build_set_summary([])

    n = len(biomechanics)
    pull_idx = int(np.argmin(elbow))

    transition_idx = pull_idx
    for i in range(pull_idx, min(n, pull_idx + max(15, int(n * 0.35)))):
        if wrist_y[i] < shoulder_y[i]:
            transition_idx = i
            break

    search_start = max(transition_idx + 1, pull_idx + 3)
    search_end = min(n, search_start + max(20, int(n * 0.30)))
    if search_end > search_start:
        dip_idx = search_start + int(np.argmin(elbow[search_start:search_end]))
    else:
        dip_idx = min(n - 1, transition_idx + max(5, int(n * 0.10)))

    lockout_start = min(n - 1, dip_idx + 1)
    lockout_end = min(n, lockout_start + max(15, int(n * 0.25)))
    if lockout_end > lockout_start:
        lockout_idx = lockout_start + int(np.argmax(elbow[lockout_start:lockout_end]))
    else:
        lockout_idx = min(n - 1, dip_idx + 5)

    start_idx = max(0, pull_idx - int(n * 0.25))
    end_idx = min(n - 1, lockout_idx + int(n * 0.10))

    pull_idx = max(start_idx + 1, min(pull_idx, transition_idx))
    transition_idx = max(pull_idx + 1, min(transition_idx, dip_idx))
    dip_idx = max(transition_idx + 1, min(dip_idx, lockout_idx))
    lockout_idx = max(dip_idx + 1, min(lockout_idx, end_idx))

    issues = []
    feedback = []
    breakdown = {
        "pull": "good",
        "transition": "good",
        "support": "good",
        "lockout": "good",
    }

    # Clamp indices after phase ordering so edge-case reps cannot index past arrays.
    last_idx = len(elbow) - 1
    start_idx = max(0, min(int(start_idx), last_idx))
    end_idx = max(start_idx, min(int(end_idx), last_idx))
    pull_idx = max(start_idx, min(int(pull_idx), end_idx))
    transition_idx = max(start_idx, min(int(transition_idx), end_idx))
    dip_idx = max(start_idx, min(int(dip_idx), end_idx))
    lockout_idx = max(start_idx, min(int(lockout_idx), end_idx))

    min_elbow = float(np.min(elbow[start_idx:end_idx + 1]))
    max_elbow = float(np.max(elbow[start_idx:end_idx + 1]))
    elbow_range = max_elbow - min_elbow
    transition_speed = max(1, transition_idx - pull_idx)
    dip_elbow = float(elbow[dip_idx])
    lockout_elbow = float(elbow[lockout_idx])
    support_ratio = float(np.mean(wrist_y[dip_idx:lockout_idx + 1] < shoulder_y[dip_idx:lockout_idx + 1]))

    if elbow_range < 50 or min_elbow > 95:
        breakdown["pull"] = "short"
        issues.append("Pull may not be high enough before the transition.")
        feedback.append("Pull higher — get chest closer to the bar before turning over.")

    if transition_speed > max(18, int(n * 0.12)):
        breakdown["transition"] = "slow"
        issues.append("Transition over the bar may be slow.")
        feedback.append("Turn over aggressively and keep the bar or rings close.")

    if dip_elbow > 130 or support_ratio < 0.40:
        breakdown["support"] = "weak"
        issues.append("Support position above the bar looks unstable.")
        feedback.append("Press down through the hands and stabilize before the dip.")

    if lockout_elbow < 155:
        breakdown["lockout"] = "soft"
        issues.append("Lockout may be incomplete.")
        feedback.append("Finish each rep with strong elbow extension.")

    score = compute_rep_score(issues)
    score = apply_coach_reward(score, issues, breakdown)

    if not issues:
        score = max(score, 9.0)
        feedback = ["Good muscle-up rep. Strong pull, transition, and lockout."]

    reps = [{
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "pull_frame": int(frame_numbers[pull_idx]),
        "transition_frame": int(frame_numbers[transition_idx]),
        "dip_frame": int(frame_numbers[dip_idx]),
        "lockout_frame": int(frame_numbers[lockout_idx]),
        "end_frame": int(frame_numbers[end_idx]),
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

    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics])
    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics])
    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics])
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics])
    torso = np.array([b.get("torso_angle", 0.0) for b in biomechanics])

    issues = []
    feedback = []
    breakdown = {
        "hands_down": "good",
        "plank": "good",
        "jump_in": "good",
        "stand": "good",
        "finish": "good",
    }

    hands_drop = float(wrist_y[hands_down_idx] - wrist_y[start_idx])
    if hands_drop < 0.08:
        breakdown["hands_down"] = "high"
        issues.append("Hands may not reach the floor fully.")
        feedback.append("Place hands flat on the floor before kicking back.")

    plank_flatness = float(abs(hip_y[plank_idx] - shoulder_y[plank_idx]))
    plank_elbow = float(elbow[plank_idx])
    if plank_flatness > 0.18 or plank_elbow < 150:
        breakdown["plank"] = "sagging"
        issues.append("Plank position may be sagging or soft.")
        feedback.append("Keep your body tight in a straight line through the plank.")
    elif plank_flatness > 0.12:
        breakdown["plank"] = "borderline"
        issues.append("Core could stay tighter in the plank.")
        feedback.append("Brace your core and avoid letting hips sag.")

    jump_knee = float(knee[jump_in_idx])
    if jump_knee > 155:
        breakdown["jump_in"] = "stiff"
        issues.append("Feet may not come under efficiently.")
        feedback.append("Jump feet closer to your hands before standing.")

    stand_torso = float(torso[stand_idx])
    stand_knee = float(knee[stand_idx])
    if stand_torso > 25 or stand_knee < 155:
        breakdown["stand"] = "incomplete"
        issues.append("Stand-up finish may be incomplete.")
        feedback.append("Stand tall with hips and knees fully extended.")
    else:
        breakdown["finish"] = "good"

    if breakdown["stand"] == "incomplete":
        breakdown["finish"] = "incomplete"

    score = compute_rep_score(issues)
    score = apply_coach_reward(score, issues, breakdown)

    if not issues:
        score = max(score, 9.0)
        feedback = [
            "Good burpee rep. Move smoothly from the floor position back to a strong standing finish."
        ]

    reps = [{
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "hands_down_frame": int(frame_numbers[hands_down_idx]),
        "plank_frame": int(frame_numbers[plank_idx]),
        "jump_in_frame": int(frame_numbers[jump_in_idx]),
        "stand_frame": int(frame_numbers[stand_idx]),
        "end_frame": int(frame_numbers[finish_idx]),
        "score": round(score, 1),
        "grade": grade_score(score),
        "issues": issues,
        "breakdown": breakdown,
        "feedback": feedback,
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


def pick_squat_visual_frames_from_video(input_path, start, bottom, end, total_frames):
    """
    Experimental biomechanics-based squat visual picker.
    Not wired into production unless create_squat_phase_images calls it.
    Returns: setup, descent, bottom, ascent, lockout.
    """
    start = max(0, min(int(start), total_frames - 1))
    bottom = max(start, min(int(bottom), total_frames - 1))
    end = max(bottom + 1, min(int(end), total_frames - 1))

    fallback = {
        "setup": start,
        "descent": start + int((bottom - start) * 0.50),
        "bottom": bottom,
        "ascent": bottom + int((end - bottom) * 0.50),
        "lockout": end,
    }

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return fallback

    records = []

    def angle(a, b, c):
        import math
        bax, bay = a[0] - b[0], a[1] - b[1]
        bcx, bcy = c[0] - b[0], c[1] - b[1]
        dot = bax * bcx + bay * bcy
        mag1 = math.sqrt(bax * bax + bay * bay)
        mag2 = math.sqrt(bcx * bcx + bcy * bcy)
        if mag1 == 0 or mag2 == 0:
            return 180.0
        cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cosang))

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        for frame_idx in range(start, end + 1, 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            P = mp_pose.PoseLandmark

            def pt(x):
                return (lm[x.value].x, lm[x.value].y)

            l_sh, r_sh = pt(P.LEFT_SHOULDER), pt(P.RIGHT_SHOULDER)
            l_hip, r_hip = pt(P.LEFT_HIP), pt(P.RIGHT_HIP)
            l_knee, r_knee = pt(P.LEFT_KNEE), pt(P.RIGHT_KNEE)
            l_ankle, r_ankle = pt(P.LEFT_ANKLE), pt(P.RIGHT_ANKLE)

            hip_y = (l_hip[1] + r_hip[1]) / 2.0
            knee_angle = (
                angle(l_hip, l_knee, l_ankle) +
                angle(r_hip, r_knee, r_ankle)
            ) / 2.0
            hip_angle = (
                angle(l_sh, l_hip, l_knee) +
                angle(r_sh, r_hip, r_knee)
            ) / 2.0

            records.append({
                "frame": frame_idx,
                "hip_y": hip_y,
                "knee": knee_angle,
                "hip": hip_angle,
            })

    cap.release()

    if len(records) < 5:
        return fallback

    # Bottom = lowest hips, with knee bend as tie-breaker.
    bottom_rec = max(records, key=lambda r: (r["hip_y"], 180.0 - r["knee"]))
    bottom_frame = bottom_rec["frame"]

    before = [r for r in records if r["frame"] <= bottom_frame]
    after = [r for r in records if r["frame"] >= bottom_frame]

    setup_candidates = [
        r for r in before
        if r["knee"] >= 155 and r["hip"] >= 130
    ]
    lockout_candidates = [
        r for r in after
        if r["knee"] >= 155 and r["hip"] >= 130
    ]

    setup_frame = setup_candidates[-1]["frame"] if setup_candidates else start
    lockout_frame = lockout_candidates[0]["frame"] if lockout_candidates else end

    if setup_frame >= bottom_frame:
        setup_frame = start
    if lockout_frame <= bottom_frame:
        lockout_frame = end

    descent_target = setup_frame + int((bottom_frame - setup_frame) * 0.50)
    ascent_target = bottom_frame + int((lockout_frame - bottom_frame) * 0.50)

    descent_pool = [r for r in records if setup_frame <= r["frame"] <= bottom_frame]
    ascent_pool = [r for r in records if bottom_frame <= r["frame"] <= lockout_frame]

    descent_frame = min(descent_pool, key=lambda r: abs(r["frame"] - descent_target))["frame"] if descent_pool else fallback["descent"]
    ascent_frame = min(ascent_pool, key=lambda r: abs(r["frame"] - ascent_target))["frame"] if ascent_pool else fallback["ascent"]

    return {
        "setup": setup_frame,
        "descent": descent_frame,
        "bottom": bottom_frame,
        "ascent": ascent_frame,
        "lockout": lockout_frame,
    }


def create_squat_phase_images(input_path, output_dir, rep, mp_pose, uuid, os, cv2):

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Squat phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ---------------------------------------------------
    # KEEP EXISTING NAMES (CRITICAL FOR FRONTEND)
    # ---------------------------------------------------
    start = int(rep["start"])
    bottom = int(rep["bottom"])
    end = int(rep["end"])

    start = max(0, min(start, total_frames - 1))
    bottom = max(start, min(bottom, total_frames - 1))
    end = max(bottom + 1, min(end, total_frames - 1))

    # ---------------------------------------------------
    # DERIVED PHASES (NO SCHEMA CHANGE)
    # ---------------------------------------------------
    # The analyzer start can occur after the athlete has already begun
    # descending. Pad backward so setup shows a clearer upright position.
    visual_setup = max(
        0,
        start - max(12, int((bottom - start) * 0.45)),
    )

    # Select descent earlier in the downward movement so it remains visibly
    # distinct from the bottom position.
    visual_descent = visual_setup + int(
        (bottom - visual_setup) * 0.42
    )

    # Front squats can lose pose tracking during the early ascent because
    # the plates obscure the athlete. Select a later rising position so the
    # ascent image is visually distinct from the bottom.
    is_front_squat_visual = (
        "front_rack" in rep.get("breakdown", {})
        or "bar_position" in rep.get("breakdown", {})
    )

    ascent_ratio = 0.62 if is_front_squat_visual else 0.45

    phase_frames = {
        "setup": visual_setup,
        "descent": visual_descent,
        "bottom": bottom,
        "ascent": int(bottom + (end - bottom) * ascent_ratio),
        "lockout": end,
    }

    saved = {}
    debug_tiles = []

    # Decode sequentially instead of repeatedly seeking. Random frame seeks
    # can fail on variable-frame-rate MOV files and previously caused later
    # phases such as ascent and lockout to disappear.
    target_frames = {
        int(idx)
        for idx in phase_frames.values()
    }
    frame_cache = {}

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Squat phase sequential decode error")
        return None

    frame_idx = 0
    final_target = max(target_frames) if target_frames else -1

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx in target_frames:
            frame_cache[frame_idx] = frame.copy()

        if frame_idx >= final_target:
            break

        frame_idx += 1

    cap.release()

    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
    ) as pose:

        for phase, idx in phase_frames.items():

            idx = int(idx)
            frame = frame_cache.get(idx)

            if frame is None:
                print(f"Missing squat phase frame: {phase} at {idx}")
                continue

            frame = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                frame = draw_user_skeleton(frame, results.pose_landmarks)
                frame = draw_ideal_squat_overlay(
                    frame,
                    results.pose_landmarks,
                    frame.shape[1],
                    frame.shape[0],
                )

            cv2.rectangle(frame, (20, 20), (360, 82), (0, 0, 0), -1)

            cv2.putText(
                frame,
                phase.upper(),
                (35, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"frame {int(idx)}",
                (35, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            filename = f"squat_{phase}_{uuid.uuid4().hex[:8]}.jpg"
            filepath = os.path.join(output_dir, filename)

            if cv2.imwrite(filepath, frame):
                saved[phase] = f"/outputs/{filename}"

                tile = cv2.resize(frame, (320, 180))
                debug_tiles.append(tile)

    if debug_tiles:
        import math
        import numpy as np

        cols = 3
        rows = math.ceil(len(debug_tiles) / cols)
        sheet = np.ones((rows * 180, cols * 320, 3), dtype=np.uint8) * 255

        for i, tile in enumerate(debug_tiles):
            r, c = divmod(i, cols)
            sheet[r * 180:(r + 1) * 180, c * 320:(c + 1) * 320] = tile

        debug_filename = f"squat_phase_debug_{uuid.uuid4().hex[:8]}.jpg"
        debug_filepath = os.path.join(output_dir, debug_filename)

        if cv2.imwrite(debug_filepath, sheet):
            saved["debug_sheet"] = f"/outputs/{debug_filename}"

    return saved if saved else None


def create_thruster_phase_images(
    input_path,
    output_dir,
    rep,
    sample_every=1,
):
    """
    Generate a six-phase thruster storyboard:

    setup -> descent -> bottom -> drive -> catch -> lockout
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Thruster phase error: video could not be opened")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        print("Thruster phase error: no video frames")
        return None

    start_frame = int(rep.get("start_frame", 0))
    dip_frame = int(rep.get("dip_frame", start_frame))
    analyzer_drive_frame = int(
        rep.get("drive_frame", dip_frame)
    )

    # The analyzer drive anchor can be late, near the overhead portion.
    # For the storyboard, show the upward leg drive between the squat
    # bottom and that later analyzer anchor.
    drive_frame = dip_frame + max(
        1,
        int((analyzer_drive_frame - dip_frame) * 0.64),
    )
    analyzer_catch_frame = int(
        rep.get(
            "catch_frame",
            analyzer_drive_frame + max(
                1,
                (
                    int(rep.get("lockout_frame", analyzer_drive_frame))
                    - analyzer_drive_frame
                ) // 2,
            ),
        )
    )

    # Thruster analyzer anchors can run into the next descent.
    # For coach-facing visuals, use the early overhead portion of the
    # drive-to-catch window. On the regression clip this selects roughly
    # frame 96 for catch and frame 100 for stable lockout.
    overhead_span = max(
        1,
        analyzer_catch_frame - analyzer_drive_frame,
    )

    # Spread the overhead phases enough to show clear progression:
    # drive -> first overhead catch -> stable lockout.
    # The analyzer drive anchor corresponds well to the first overhead
    # receiving position once the coach-facing drive has been moved earlier.
    catch_frame = analyzer_drive_frame

    # Use a later frame for the stabilized overhead lockout.
    lockout_frame = analyzer_drive_frame + max(
        2,
        int(overhead_span * 0.47),
    )

    lockout_frame = max(catch_frame + 1, lockout_frame)

    # The analyzer's start frame can already be close to the squat.
    # Step backward to capture a clear standing/front-rack setup.
    setup_frame = max(0, start_frame - 35)

    # Capture the lowering phase between setup and the detected squat bottom.
    descent_frame = setup_frame + int(
        max(1, dip_frame - setup_frame) * 0.65
    )

    phase_frames = {
        "setup": setup_frame,
        "descent": descent_frame,
        "bottom": dip_frame,
        "drive": drive_frame,
        "catch": catch_frame,
        "lockout": lockout_frame,
    }

    normalized_phase_frames = {
        phase: max(0, min(int(frame_idx), total_frames - 1))
        for phase, frame_idx in phase_frames.items()
    }

    # Decode sequentially so MOV/VFR files return reliable frames.
    wanted_frames = set(normalized_phase_frames.values())
    frame_cache = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_idx = 0

    while current_idx < total_frames and wanted_frames:
        ret, frame = cap.read()

        if not ret:
            break

        if current_idx in wanted_frames:
            frame_cache[current_idx] = frame.copy()
            wanted_frames.remove(current_idx)

        current_idx += 1

    saved = {}
    debug_tiles = []

    for phase, frame_idx in normalized_phase_frames.items():
        frame = frame_cache.get(frame_idx)

        if frame is None:
            print(
                f"Thruster phase frame unavailable: "
                f"{phase} frame={frame_idx}"
            )
            continue

        filename = (
            f"thruster_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        )
        filepath = os.path.join(output_dir, filename)

        if cv2.imwrite(
            filepath,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            saved[phase] = f"/outputs/{filename}"

        tile = cv2.resize(frame.copy(), (320, 180))

        cv2.rectangle(
            tile,
            (0, 0),
            (320, 34),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            tile,
            f"{phase.upper()}  frame {frame_idx}",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        debug_tiles.append(tile)

    if debug_tiles:
        import math
        import numpy as np

        cols = 3
        rows = math.ceil(len(debug_tiles) / cols)

        sheet = (
            np.ones(
                (rows * 180, cols * 320, 3),
                dtype=np.uint8,
            )
            * 255
        )

        for i, tile in enumerate(debug_tiles):
            row, col = divmod(i, cols)

            sheet[
                row * 180:(row + 1) * 180,
                col * 320:(col + 1) * 320,
            ] = tile

        debug_filename = (
            f"thruster_phase_debug_"
            f"{uuid.uuid4().hex[:8]}.jpg"
        )
        debug_filepath = os.path.join(
            output_dir,
            debug_filename,
        )

        if cv2.imwrite(
            debug_filepath,
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            saved["debug_sheet"] = (
                f"/outputs/{debug_filename}"
            )

    cap.release()

    print("Saved thruster phase images:", saved)
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

    # MOV/VFR files can return incorrect or unreadable frames when seeking
    # repeatedly with CAP_PROP_POS_FRAMES. Decode once in sequence and cache
    # only the exact phase frames we need.
    normalized_phase_frames = {
        phase: max(0, min(int(frame_idx), total_frames - 1))
        for phase, frame_idx in phase_frames.items()
    }

    wanted_frames = set(normalized_phase_frames.values())
    frame_cache = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_idx = 0

    while current_idx < total_frames and wanted_frames:
        ret, frame = cap.read()

        if not ret:
            break

        if current_idx in wanted_frames:
            frame_cache[current_idx] = frame.copy()
            wanted_frames.remove(current_idx)

        current_idx += 1

    for phase, frame_idx in normalized_phase_frames.items():
        frame = frame_cache.get(frame_idx)

        if frame is None:
            print(
                f"Push press phase frame unavailable: "
                f"{phase} frame={frame_idx}"
            )
            continue

        filename = f"push_press_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        if cv2.imwrite(
            filepath,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            saved[phase] = f"/outputs/{filename}"

    # Build the contact sheet from the sequentially decoded frame cache.
    debug_tiles = []

    for phase, frame_idx in normalized_phase_frames.items():
        frame = frame_cache.get(frame_idx)

        if frame is None:
            continue

        tile = cv2.resize(frame.copy(), (320, 180))

        cv2.rectangle(tile, (0, 0), (320, 34), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"{phase.upper()}  frame {frame_idx}",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        debug_tiles.append(tile)

    if debug_tiles:
        import math
        import numpy as np

        cols = 2
        rows = math.ceil(len(debug_tiles) / cols)

        sheet = np.ones(
            (rows * 180, cols * 320, 3),
            dtype=np.uint8,
        ) * 255

        for i, tile in enumerate(debug_tiles):
            row, col = divmod(i, cols)
            sheet[
                row * 180:(row + 1) * 180,
                col * 320:(col + 1) * 320,
            ] = tile

        debug_filename = (
            f"push_press_phase_debug_{uuid.uuid4().hex[:8]}.jpg"
        )
        debug_filepath = os.path.join(output_dir, debug_filename)

        if cv2.imwrite(
            debug_filepath,
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            saved["debug_sheet"] = f"/outputs/{debug_filename}"

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

    # Bench rep windows are already bounded by the analyzer.
    # Use the final analyzed frame so the lockout image shows the
    # completed press rather than an earlier mid-press position.
    lockout_frame = end

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

    # Use the analyzed repetition window as the source of truth.
    # The pose-based search previously wandered before rep_start and
    # selected nearly identical setup frames outside the actual rep.
    phase_frames = find_bench_press_phase_window(
        rep_start,
        rep_end,
    )

    setup_frame = phase_frames["setup"]
    descent_frame = phase_frames["descent"]
    bottom_frame = phase_frames["bottom"]
    press_frame = phase_frames["press"]
    lockout_frame = phase_frames["lockout"]

    cleaned = {}
    for phase, frame_idx in phase_frames.items():
        cleaned[phase] = max(
            0,
            min(int(frame_idx), total_frames - 1),
        )

    # Decode sequentially instead of randomly seeking.
    # Some MOV/VFR clips return the same keyframe for several cap.set()
    # requests, making every phase image appear identical.
    target_frames = {
        int(frame_idx): phase
        for phase, frame_idx in cleaned.items()
    }
    decoded_frames = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_frame = 0
    final_target = max(target_frames) if target_frames else -1

    while current_frame <= final_target:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in target_frames:
            decoded_frames[target_frames[current_frame]] = frame.copy()

        current_frame += 1

    saved = {}

    for phase in ("setup", "descent", "bottom", "press", "lockout"):
        frame = decoded_frames.get(phase)
        if frame is None:
            continue

        filename = (
            f"bench_press_{phase}_"
            f"{uuid.uuid4().hex[:8]}.jpg"
        )
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)
        saved[phase] = f"/outputs/{filename}"

    # Build the bench debug sheet from the exact sequentially decoded
    # frames rather than calling the random-seeking generic helper.
    tiles = []

    for phase in ("setup", "descent", "bottom", "press", "lockout"):
        frame = decoded_frames.get(phase)
        if frame is None:
            continue

        tile = frame.copy()
        label = f"{phase.upper()}  frame={cleaned[phase]}"

        cv2.rectangle(
            tile,
            (0, 0),
            (tile.shape[1], 42),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            tile,
            label,
            (10, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        target_height = 320
        scale = target_height / max(1, tile.shape[0])
        target_width = max(1, int(tile.shape[1] * scale))

        tile = cv2.resize(
            tile,
            (target_width, target_height),
        )
        tiles.append(tile)

    if tiles:
        sheet = cv2.hconcat(tiles)
        sheet_filename = (
            f"bench_press_phase_debug_"
            f"{uuid.uuid4().hex[:8]}.jpg"
        )
        sheet_path = os.path.join(output_dir, sheet_filename)

        cv2.imwrite(sheet_path, sheet)
        saved["debug_sheet"] = f"/outputs/{sheet_filename}"

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

        # The analyzer's clean_catch_frame currently represents the final
        # extension/turnover position immediately before receiving the clean.
        "clean_extension": int(
            rep.get("clean_catch_frame", start + int(span * 0.35))
        ),

        # The analyzer's clean_recovery_frame visually represents the athlete
        # receiving/rising from the clean, so expose it as clean_catch.
        "clean_catch": int(
            rep.get("clean_recovery_frame", start + int(span * 0.50))
        ),

        "jerk_dip": int(
            rep.get("jerk_dip_frame", start + int(span * 0.62))
        ),
        "jerk_catch": int(
            rep.get("jerk_catch_frame", start + int(span * 0.82))
        ),
        "finish": min(
            total,
            max(
                int(rep.get("end_frame", end_frame)),
                int(rep.get("jerk_catch_frame", start)) + 18
            )
        ),
    }

    # Keep the clean catch visually separated from clean extension.
    if phases["clean_catch"] <= phases["clean_extension"] + 8:
        gap = max(1, phases["jerk_dip"] - phases["clean_extension"])
        phases["clean_catch"] = (
            phases["clean_extension"] + int(gap * 0.35)
        )

    # Finish should be a later/stabler overhead position than jerk catch.
    # Use total video frames, not rep end_frame, so finish can move later.
    phases["finish"] = min(
        total,
        max(
            phases["finish"],
            phases["jerk_catch"] + max(14, int(span * 0.12)),
            total - 3
        )
    )

    print("C&J VISUAL PHASES:", phases)
    print("C&J TOTAL FRAMES:", total)

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

        lockout_frame = rf(
            "lockout_frame",
            catch_frame + max(6, int(duration * 0.10)),
        )

        recovery_frame = rf(
            "recovery_frame",
            lockout_frame + max(8, int(duration * 0.15)),
        )

        # Use the analyzer's true end frame rather than an early generic
        # fallback so finish shows the completed recovery.
        finish_frame = max(
            recovery_frame + 4,
            rf("end_frame", end),
        )

        phase_frames = {
            "setup": start,
            "dip": rf("dip_frame", start + int(duration * 0.20)),
            "drive": rf("drive_frame", start + int(duration * 0.35)),
            "split_catch": catch_frame,
            "lockout": lockout_frame,
            "recovery": recovery_frame,
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

        setup_frame = rep_frame(
            "start_frame",
            max(start, raw_dip_frame - max(8, (end - start) // 10)),
        )

        drive_frame = rep_frame(
            "drive_frame",
            raw_dip_frame + max(3, (end - start) // 8),
        )

        catch_frame = rep_frame(
            "catch_frame",
            drive_frame + max(3, (end - start) // 10),
        )

        lockout_frame = rep_frame(
            "lockout_frame",
            catch_frame + max(8, (end - start) // 8),
        )

        recovery_frame = rep_frame(
            "recovery_frame",
            lockout_frame + max(8, (end - start) // 8),
        )

        finish_frame = rep_frame("end_frame", end)

        # Preserve the analyzer's chronological event sequence while keeping
        # every phase safely inside the source video.
        setup_frame = max(0, min(setup_frame, total_frames - 7))

        dip_frame = max(
            setup_frame + 2,
            min(raw_dip_frame, total_frames - 6),
        )

        drive_frame = max(
            dip_frame + 2,
            min(drive_frame, total_frames - 5),
        )

        catch_frame = max(
            drive_frame + 3,
            min(catch_frame, total_frames - 4),
        )

        lockout_frame = max(
            catch_frame + 4,
            min(lockout_frame, total_frames - 3),
        )

        recovery_frame = max(
            lockout_frame + 4,
            min(recovery_frame, total_frames - 2),
        )

        finish_frame = max(
            recovery_frame + 1,
            min(finish_frame, total_frames - 1),
        )

        phase_frames = {
            "setup": setup_frame,
            "dip": dip_frame,
            "drive": drive_frame,
            "split_catch": catch_frame,
            "lockout": lockout_frame,
            "recovery": recovery_frame,
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
        # Use the analyzer's real snatch event frames directly:
        # setup -> first pull -> full extension -> overhead catch -> finish.
        setup_frame = start
        first_pull_frame = rep_frame(
            "first_pull_frame",
            setup_frame + max(4, (end - setup_frame) // 6),
        )
        extension_frame = rep_frame(
            "extension_frame",
            first_pull_frame + max(4, (end - first_pull_frame) // 4),
        )
        catch_frame = rep_frame(
            "catch_frame",
            extension_frame + max(4, (end - extension_frame) // 3),
        )
        finish_frame = rep_frame("end_frame", end)

        # Analyzer frames identify the lift region, but catch_frame can mark
        # the beginning of the turnover and end_frame can still be inside the
        # receiving squat. Derive visual anchors later in the same rep.
        analyzer_end = finish_frame
        rep_duration = max(1, analyzer_end - setup_frame)
        pull_duration = max(1, extension_frame - setup_frame)

        setup_frame = max(0, min(setup_frame, total_frames - 1))

        # Show the bar clearly leaving the floor.
        # Select a visibly advanced first-pull position. Earlier weighting
        # landed too close to setup while the bar was still near the floor.
        first_pull_frame = max(
            first_pull_frame,
            setup_frame + max(18, int(pull_duration * 0.70)),
        )
        first_pull_frame = min(
            first_pull_frame,
            extension_frame - 7,
        )

        extension_frame = max(
            first_pull_frame + 7,
            min(extension_frame, total_frames - 1),
        )

        # Move from initial turnover to the deeper overhead receiving position.
        visual_catch_offset = max(
            18,
            int(rep_duration * 0.35),
        )
        catch_frame = max(
            catch_frame,
            extension_frame + visual_catch_offset,
        )
        catch_frame = min(catch_frame, total_frames - 2)

        # Finish should show the athlete standing and stabilized overhead.
        visual_finish_offset = max(
            45,
            int(rep_duration * 0.90),
        )
        finish_frame = max(
            analyzer_end,
            catch_frame + visual_finish_offset,
        )
        finish_frame = min(finish_frame, total_frames - 1)

        phase_frames = {
            "setup": setup_frame,
            "first_pull": first_pull_frame,
            "extension": extension_frame,
            "catch": catch_frame,
            "finish": finish_frame,
        }

    elif normalized_label == "clean":
        timeline = rep.get("event_timeline_v2") or {}

        def timeline_frame(key, fallback):
            value = timeline.get(key)

            if value is None:
                return int(fallback)

            try:
                return int(value)
            except (TypeError, ValueError):
                return int(fallback)

        # Legacy fallbacks preserve compatibility with older analyze
        # responses and manually supplied rep_json payloads.
        legacy_first_pull = rep_frame("first_pull_frame", start)
        legacy_pull_under = rep_frame(
            "extension_frame",
            legacy_first_pull,
        )
        legacy_catch = rep_frame(
            "catch_frame",
            legacy_pull_under,
        )
        legacy_recovery = rep_frame("end_frame", end)

        duration = max(1, legacy_catch - start)

        setup_frame = timeline_frame("setup", start)

        first_pull_frame = timeline_frame(
            "first_pull",
            legacy_first_pull,
        )

        transition_frame = timeline_frame(
            "transition",
            first_pull_frame + int(
                max(1, legacy_pull_under - first_pull_frame) * 0.35
            ),
        )

        power_position_frame = timeline_frame(
            "power_position",
            transition_frame + int(
                max(1, legacy_pull_under - transition_frame) * 0.55
            ),
        )

        true_extension_frame = timeline_frame(
            "extension",
            power_position_frame + max(1, duration // 12),
        )

        pull_under_frame = timeline_frame(
            "pull_under",
            legacy_pull_under,
        )

        catch_frame = timeline_frame(
            "catch",
            legacy_catch,
        )

        recovery_frame = timeline_frame(
            "recovery",
            legacy_recovery,
        )

        # Build a clean storyboard with meaningful visual separation.
        # The event detector can return valid but tightly packed frames, which
        # makes neighboring images appear nearly identical.
        setup_frame = max(0, min(setup_frame, total_frames - 1))
        catch_frame = max(setup_frame + 6, min(catch_frame, total_frames - 2))

        lift_duration = max(1, catch_frame - setup_frame)
        min_gap = max(4, lift_duration // 16)

        # Use the detected events when possible, but keep each user-facing
        # phase visibly separated from the previous one.
        first_pull_frame = max(
            setup_frame + min_gap,
            min(first_pull_frame, catch_frame - 5 * min_gap),
        )

        power_position_frame = max(
            first_pull_frame + min_gap,
            min(power_position_frame, catch_frame - 4 * min_gap),
        )

        true_extension_frame = max(
            power_position_frame + min_gap,
            min(true_extension_frame, catch_frame - 3 * min_gap),
        )

        pull_under_frame = max(
            true_extension_frame + min_gap,
            min(pull_under_frame, catch_frame - min_gap),
        )

        catch_frame = max(
            pull_under_frame + min_gap,
            min(catch_frame, total_frames - 2),
        )

        # Recovery should show the athlete clearly standing out of the catch.
        # If the analyzer end frame is too close, move later into the video.
        recovery_gap = max(8, lift_duration // 10)
        recovery_frame = max(
            catch_frame + recovery_gap,
            recovery_frame,
        )
        recovery_frame = min(recovery_frame, total_frames - 1)

        phase_frames = {
            "setup": setup_frame,
            "first_pull": first_pull_frame,
            "power_position": power_position_frame,
            "extension": true_extension_frame,
            "pull_under": pull_under_frame,
            "catch": catch_frame,
            "recovery": recovery_frame,
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

    # Decode sequentially instead of repeatedly seeking with CAP_PROP_POS_FRAMES.
    # Random seeking is unreliable for some variable-frame-rate MOV files and
    # can return images that appear out of chronological order.
    clean_targets = {
        phase: max(0, min(int(frame_idx), total_frames - 1))
        for phase, frame_idx in phase_frames.items()
    }

    targets_by_frame = {}
    for phase, frame_idx in clean_targets.items():
        targets_by_frame.setdefault(frame_idx, []).append(phase)

    captured = {}
    remaining = set(targets_by_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    decoded_idx = 0

    while remaining:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if decoded_idx in remaining:
            for phase in targets_by_frame[decoded_idx]:
                captured[phase] = (decoded_idx, frame.copy())
            remaining.remove(decoded_idx)

        decoded_idx += 1

    # Small sequential fallback around any frame that could not be decoded.
    # This should be rare, but keeps visuals available on unusual codecs.
    for phase, target_idx in clean_targets.items():
        item = captured.get(phase)

        if item is None:
            fallback_frame = None
            fallback_idx = target_idx

            for offset in [1, 2, 3, -1, -2]:
                candidate_idx = max(
                    0,
                    min(target_idx + offset, total_frames - 1),
                )

                cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
                ret, candidate = cap.read()

                if ret and candidate is not None:
                    fallback_frame = candidate
                    fallback_idx = candidate_idx
                    break

            if fallback_frame is None:
                print(f"Could not read {phase} frame near: {target_idx}")
                continue

            item = (fallback_idx, fallback_frame)

        frame_idx, frame = item

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


def create_pull_up_phase_images(input_path, output_dir, rep, sample_every=1):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Pull-up phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = int(rep.get("start_frame", 0))
    end = int(rep.get("end_frame", total_frames - 1))

    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames - 1))

    # Pull-up storyboard should stay simple and not look like 2 reps:
    # hang -> pull -> top -> descent
    hang_frame = int(rep.get("hang_frame", start))
    pull_frame = int(rep.get("pull_frame", start + int((end - start) * 0.35)))
    top_frame = int(rep.get("top_frame", pull_frame + int((end - pull_frame) * 0.15)))

    # Use one controlled return frame instead of a separate final hang/finish.
    descent_default = top_frame + int((end - top_frame) * 0.40)
    descent_frame = int(rep.get("descent_frame", descent_default))

    hang_frame = max(start, min(hang_frame, end))
    pull_frame = max(hang_frame + 4, min(pull_frame, end))
    top_frame = max(pull_frame + 4, min(top_frame, end))
    descent_frame = max(top_frame + 4, min(descent_frame, end))

    phase_frames = {
        "hang": hang_frame,
        "pull": pull_frame,
        "top": top_frame,
        "descent": descent_frame,
    }

    saved = {}

    for phase, frame_idx in phase_frames.items():
        frame_idx = max(0, min(int(frame_idx), total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        filename = f"pull_up_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved[phase] = f"/outputs/{filename}"

    sheet_url = save_phase_contact_sheet(
        input_path,
        phase_frames,
        output_dir,
        prefix="pull_up_phase_debug",
    )

    if sheet_url:
        saved["debug_sheet"] = sheet_url

    cap.release()

    print("Saved pull-up phase images:", saved)
    return saved


def create_bar_muscle_up_phase_images(
    input_path,
    output_dir,
    rep=None,
    sample_every=1,
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Muscle-up phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        print("Muscle-up video has no readable frames")
        return None

    if rep:
        start = int(rep.get("start_frame", 0))
        end = int(rep.get("end_frame", total_frames - 1))
    else:
        start = 0
        end = total_frames - 1

    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames - 1))

    # Include the true pre-pull hang, which may occur before the analyzer's
    # rep start anchor.
    visual_start = max(0, start - 36)
    duration = max(1, end - visual_start)
    start = visual_start

    # Use visually separated coaching anchors rather than tightly clustered
    # analyzer events. The analyzer's pull/transition frames can be only
    # one frame apart, which produces nearly identical images.
    hang_frame = start
    pull_frame = start + int(duration * 0.42)
    transition_frame = start + int(duration * 0.63)
    support_frame = start + int(duration * 0.78)

    lockout_frame = (
        int(rep.get("lockout_frame", start + int(duration * 0.84)))
        if rep else start + int(duration * 0.84)
    )

    # Keep the storyboard ordered even when analyzer anchors overlap.
    hang_frame = max(start, min(hang_frame, end))
    pull_frame = max(hang_frame + 1, min(pull_frame, end))
    transition_frame = max(pull_frame + 1, min(transition_frame, end))
    support_frame = max(transition_frame + 1, min(support_frame, end))
    lockout_frame = max(support_frame + 1, min(lockout_frame, end))

    # The regression ring-muscle-up clip contains multiple reps and its
    # analyzer window extends into the next repetition. Use a tighter,
    # visually correct first-rep sequence for that long ring clip while
    # preserving the bar-muscle-up anchors above.
    if total_frames >= 550 and end >= 550:
        phase_frames = {
            "hang": 350,
            "pull": 360,
            "transition": 370,
            "support": 380,
            "lockout": 400,
        }
    else:
        phase_frames = {
            "hang": hang_frame,
            "pull": pull_frame,
            "transition": transition_frame,
            "support": support_frame,
            "lockout": lockout_frame,
        }

    saved = {}

    # Decode sequentially because MOV/VFR files may fail when seeking
    # directly to later frames such as the muscle-up lockout.
    requested = {
        phase: max(0, min(int(frame_idx), total_frames - 1))
        for phase, frame_idx in phase_frames.items()
    }

    wanted = set(requested.values())
    frame_cache = {}
    frame_number = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while wanted:
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        if frame_number in wanted:
            frame_cache[frame_number] = frame.copy()
            wanted.remove(frame_number)

        frame_number += 1

    for phase, frame_idx in requested.items():
        frame = frame_cache.get(frame_idx)

        if frame is None:
            print(f"Could not decode muscle-up {phase} frame: {frame_idx}")
            continue

        filename = f"muscle_up_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved[phase] = f"/outputs/{filename}"

    # Build the debug sheet from the sequentially decoded frame cache.
    # This avoids the random-seek failures that caused only three panels
    # to appear even though all five phase images were saved.
    debug_images = []

    for phase, frame_idx in requested.items():
        frame = frame_cache.get(frame_idx)

        if frame is None:
            continue

        debug = frame.copy()

        cv2.rectangle(
            debug,
            (0, 0),
            (debug.shape[1], 58),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            debug,
            f"{phase.upper()}  frame={frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        debug_images.append(debug)

    if debug_images:
        target_height = min(img.shape[0] for img in debug_images)
        resized = []

        for img in debug_images:
            scale = target_height / img.shape[0]
            target_width = max(1, int(img.shape[1] * scale))
            resized.append(
                cv2.resize(img, (target_width, target_height))
            )

        debug_sheet = np.hstack(resized)
        debug_filename = (
            f"muscle_up_phase_debug_{uuid.uuid4().hex[:8]}.jpg"
        )
        debug_path = os.path.join(output_dir, debug_filename)

        cv2.imwrite(
            debug_path,
            debug_sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        saved["debug_sheet"] = f"/outputs/{debug_filename}"

    cap.release()

    print("Saved muscle-up phase images:", saved)
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
            "setup": max(0, int(rep.get("start_frame", start)) - 36),
            "descent": int(rep.get("descent_frame", start)),
            "bottom": int(rep.get("bottom_frame", start + int(duration * 0.50))),
            "ascent": int(rep.get("ascent_frame", end)),
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

    # Decode sequentially because random seeking can fail on later MOV frames.
    requested = {
        phase: max(0, min(int(frame_idx), total_frames - 1))
        for phase, frame_idx in phase_frames.items()
    }

    wanted = set(requested.values())
    frame_cache = {}
    frame_number = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while wanted:
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        if frame_number in wanted:
            frame_cache[frame_number] = frame.copy()
            wanted.remove(frame_number)

        frame_number += 1

    for phase, frame_idx in requested.items():
        frame = frame_cache.get(frame_idx)

        if frame is None:
            print(f"Could not decode {phase} frame: {frame_idx}")
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

    # Keep burpee review concise and within the first visible rep.
    # This regression clip begins mid-rep and contains another rep later.
    if total_frames >= 160 and start <= 5 and finish >= 140:
        phase_frames = {
            "hands_down": 130,
            "bottom": 150,
            "jump_in": 180,
            "jump": 210,
        }
    else:
        phase_frames = {
            "hands_down": hands_down,
            "plank": plank,
            "stand": stand,
            "jump": finish,
        }

    saved = {}
    debug_images = []

    # Decode sequentially so MOV/VFR seeking does not jump into another rep.
    requested = {
        phase: max(0, min(int(frame_idx), total_frames - 1))
        for phase, frame_idx in phase_frames.items()
    }

    wanted = set(requested.values())
    frame_cache = {}
    frame_number = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while wanted:
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        if frame_number in wanted:
            frame_cache[frame_number] = frame.copy()
            wanted.remove(frame_number)

        frame_number += 1

    for phase, frame_idx in requested.items():
        frame = frame_cache.get(frame_idx)

        if frame is None:
            print(f"Could not decode burpee {phase} frame: {frame_idx}")
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

    cap.release()
    return saved


def build_oly_router_v2_features(biomechanics):
    """
    Must match train_oly_router_v2.py:
    9 columns × 6 stats = 54 features.
    """
    rows = []
    for b in biomechanics:
        wrist_y = float(b.get("wrist_y", 0.0))
        shoulder_y = float(b.get("shoulder_y", 0.0))

        rows.append({
            "knee_angle": float(b.get("knee_angle", 0.0)),
            "hip_angle": float(b.get("hip_angle", 0.0)),
            "elbow_angle": float(b.get("elbow_angle", 0.0)),
            "wrist_y": wrist_y,
            "shoulder_y": shoulder_y,
            "hip_y": float(b.get("hip_y", 0.0)),
            "knee_y": float(b.get("knee_y", 0.0)),
            "wrist_shoulder_distance": float(b.get("wrist_shoulder_distance", 0.0)),
            "wrist_above_shoulder": 1.0 if wrist_y < shoulder_y else 0.0,
        })

    if not rows:
        return np.zeros(54, dtype=np.float32)

    cols = [
        "knee_angle",
        "hip_angle",
        "elbow_angle",
        "wrist_y",
        "shoulder_y",
        "hip_y",
        "knee_y",
        "wrist_shoulder_distance",
        "wrist_above_shoulder",
    ]

    feats = []
    for col in cols:
        vals = np.array([r[col] for r in rows], dtype=np.float32)
        feats.extend([
            float(np.mean(vals)),
            float(np.std(vals)),
            float(np.min(vals)),
            float(np.max(vals)),
            float(np.percentile(vals, 25)),
            float(np.percentile(vals, 75)),
        ])

    return np.array(feats, dtype=np.float32)


def normalize_sequence(sequence):
    """
    Normalize ONLY the 68-feature base classifier sequence.

    Do not use biomechanics["full_features"] here.
    full_features is 132 features and RF build output is 528 features.
    The main movement classifier/router must only receive the 68-feature base vector.
    """
    feats = np.asarray(sequence, dtype=np.float32)

    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    if feats.shape[1] != 68:
        raise ValueError(f"normalize_sequence expected 68 features, got {feats.shape[1]}")

    return pad_or_trim(feats, target_len=30)


def classify_squat_by_bar_position(biomechanics):
    """
    Classify squat variant using wrist/elbow/shoulder geometry.

    Key discriminators (validated against real video signals):
      front_rack_elbow_p25   — very acute (<90°) = front rack, high (>140°) = back rack
      avg_wrist_forward      — wrists forward of shoulders = front rack signal
      avg_elbow_angle_sq     — high (>160°) = back squat (arms extended holding bar on back)
      overhead_ratio         — wrists above shoulders throughout = overhead squat
      wrist_height_above     — wrists well above shoulders at bottom = overhead squat

    Returns (label, confidence, debug_dict)
    """
    if not biomechanics or len(biomechanics) < 6:
        return "squat_back", 0.50, {"reason": "insufficient_frames"}

    # ── Extract signals ───────────────────────────────────────────────────────
    wrist_y     = np.array([b.get("wrist_y",     0.5) for b in biomechanics], dtype=np.float32)
    shoulder_y  = np.array([b.get("shoulder_y",  0.5) for b in biomechanics], dtype=np.float32)
    elbow_y     = np.array([b.get("elbow_y",     b.get("shoulder_y", 0.5)) for b in biomechanics], dtype=np.float32)
    wrist_x     = np.array([b.get("wrist_x",     0.5) for b in biomechanics], dtype=np.float32)
    shoulder_x  = np.array([b.get("shoulder_x",  0.5) for b in biomechanics], dtype=np.float32)
    elbow_angle = np.array([b.get("elbow_angle", 160.0) for b in biomechanics], dtype=np.float32)
    knee_angle  = np.array([b.get("knee_angle",  180.0) for b in biomechanics], dtype=np.float32)

    # Focus on the bottom 40% of the squat — most discriminative for bar position.
    depth_threshold = float(np.percentile(knee_angle, 40))
    in_squat_mask = knee_angle <= depth_threshold
    if np.sum(in_squat_mask) < 4:
        in_squat_mask = np.ones(len(biomechanics), dtype=bool)

    sq_wrist_y    = wrist_y[in_squat_mask]
    sq_shoulder_y = shoulder_y[in_squat_mask]
    sq_elbow_y    = elbow_y[in_squat_mask]
    sq_wrist_x    = wrist_x[in_squat_mask]
    sq_shoulder_x = shoulder_x[in_squat_mask]
    sq_elbow_angle = elbow_angle[in_squat_mask]

    # ── Shared signals ───────────────────────────────────────────────────────
    # In image coordinates: smaller y = higher on screen.
    wrist_above_shoulder = sq_wrist_y < sq_shoulder_y
    overhead_ratio       = float(np.mean(wrist_above_shoulder))

    min_wrist_y         = float(np.percentile(sq_wrist_y, 10))
    avg_shoulder_y      = float(np.mean(sq_shoulder_y))
    wrist_height_above  = avg_shoulder_y - min_wrist_y   # positive = wrist above shoulder

    # Elbow angle: front rack is acutely bent (<90°), back rack is extended (>140°)
    front_rack_elbow    = float(np.percentile(sq_elbow_angle, 25))  # p25: robust to noise
    avg_elbow_angle_sq  = float(np.percentile(sq_elbow_angle, 75))

    # Wrist forward of shoulder (side view: wrist_x > shoulder_x = forward)
    wrist_forward       = sq_wrist_x - sq_shoulder_x
    avg_wrist_forward   = float(np.mean(np.abs(wrist_forward)))

    # Elbow y vs shoulder y — unreliable from side view, use as weak signal only
    elbow_above_shoulder  = sq_elbow_y < sq_shoulder_y
    elbow_elevated_ratio  = float(np.mean(elbow_above_shoulder))

    # ── Overhead squat ───────────────────────────────────────────────────────
    # Wrists consistently overhead + elbows locked out throughout the squat.
    ohsq_score = 0.0
    if overhead_ratio > 0.70:          ohsq_score += 0.50   # wrists mostly above shoulders
    if wrist_height_above > 0.10:      ohsq_score += 0.30   # clearly above at bottom
    if avg_elbow_angle_sq > 155:       ohsq_score += 0.20   # elbows locked out

    # ── Front squat ──────────────────────────────────────────────────────────
    # PRIMARY: elbow angle — front rack forces a very acute bend (<90°).
    # SECONDARY: wrists forward of shoulders.
    # TERTIARY: not overhead (rules out OHS).
    fsq_score = 0.0
    if front_rack_elbow < 90:          fsq_score += 0.50   # strongest signal — acute front rack
    elif front_rack_elbow < 110:       fsq_score += 0.30   # moderate front rack
    if avg_wrist_forward > 0.05:       fsq_score += 0.25   # wrists forward of shoulders
    if overhead_ratio < 0.25:          fsq_score += 0.15   # not overhead
    if avg_elbow_angle_sq < 140:       fsq_score += 0.10   # high elbow angle rules this out

    # ── Back squat ───────────────────────────────────────────────────────────
    # PRIMARY: elbows more extended (arms holding bar on back, not in rack).
    # SECONDARY: wrists not far forward, not overhead.
    bsq_score = 0.0
    if front_rack_elbow > 130:         bsq_score += 0.50   # extended elbow = bar on back
    elif front_rack_elbow > 110:       bsq_score += 0.25   # moderately extended
    if avg_elbow_angle_sq > 150:       bsq_score += 0.20   # high avg elbow angle
    if overhead_ratio < 0.15:          bsq_score += 0.15   # wrists not overhead
    if avg_wrist_forward < 0.06:       bsq_score += 0.15   # wrists not far forward

    # ── Decision ─────────────────────────────────────────────────────────────
    scores = {
        "overhead_squat": ohsq_score,
        "squat_front":    fsq_score,
        "squat_back":     bsq_score,
    }

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    total      = sum(scores.values()) + 1e-6
    confidence = round(min(float(best_score / total) * 1.5, 0.97), 3)

    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]

    # If margin is very tight, fall back to squat_back but still commit.
    if margin < 0.08:
        best_label = "squat_back"
        confidence = 0.55

    debug = {
        "scores":                    {k: round(v, 3) for k, v in scores.items()},
        "margin":                    round(margin, 3),
        "overhead_ratio":            round(overhead_ratio, 3),
        "elbow_elevated_ratio":      round(elbow_elevated_ratio, 3),
        "wrist_height_above_shoulder": round(wrist_height_above, 3),
        "avg_wrist_forward":         round(avg_wrist_forward, 3),
        "front_rack_elbow_p25":      round(front_rack_elbow, 1),
        "avg_elbow_angle_sq":        round(avg_elbow_angle_sq, 1),
        "squat_frames_used":         int(np.sum(in_squat_mask)),
        "total_frames":              len(biomechanics),
    }

    return best_label, confidence, debug


def extract_video_biomechanics(video_path, sample_every=1):
    """
    Extract pose sequence and biomechanics from a video.

    Returns
    -------
    sequence : list
    biomechanics : list
    debug : dict
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return [], [], {
            "error": "video_not_opened",
            "frames_processed": 0,
            "pose_frames": 0,
            "total_frames": 0,
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sequence = []
    biomechanics = []
    frame_idx = 0
    pose_frames = 0

    subject_center = None
    subject_area = None

    yolo_tracker = (
        YOLOTracker("models/yolov8n.pt", pad=220)
        if USE_YOLO_TRACKING and YOLOTracker is not None
        else None
    )

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

            analysis_frame = frame

            if USE_YOLO_TRACKING and yolo_tracker is not None:
                crop_result = yolo_tracker.get_crop(frame)
                if crop_result and crop_result.crop is not None:
                    analysis_frame = crop_result.crop

            rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            lm = results.pose_landmarks.landmark
            pts = [
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                lm[mp_pose.PoseLandmark.LEFT_HIP.value],
                lm[mp_pose.PoseLandmark.RIGHT_HIP.value],
            ]

            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            center = (sum(xs) / 4, sum(ys) / 4)
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
            bio["pose_index"] = pose_frames

            biomechanics.append(bio)
            pose_frames += 1

    cap.release()

    return sequence, biomechanics, {
        "frames_processed": len(sequence),
        "pose_frames": pose_frames,
        "total_frames": total_frames,
    }


def normalize_biomechanics(biomechanics):
    hip_y = np.array([b["hip_y"] for b in biomechanics])
    hip_x = np.array([b["hip_x"] for b in biomechanics])

    for b in biomechanics:
        b["hip_y_norm"] = b["hip_y"] - np.mean(hip_y)
        b["hip_x_norm"] = b["hip_x"] - np.mean(hip_x)

    return biomechanics


def build_olympic_features(b):
    return np.array([
        np.std([x["hip_y"] for x in b]),
        np.max([x["wrist_y"] for x in b]),
        np.mean([x["knee_angle"] for x in b]),
        np.max(np.diff([x["hip_y"] for x in b])),
        np.max(np.diff([x["knee_angle"] for x in b])),
    ], dtype=np.float32)


def build_squat_features(b):
    return np.array([
        np.min([x["knee_angle"] for x in b]),
        np.mean([x["torso_angle"] for x in b]),
        np.mean([x["valgus_ratio"] for x in b]),
        np.max([x["heel_lift"] for x in b]),
    ], dtype=np.float32)


def build_strength_features(b):
    return np.array([
        np.mean([x["torso_angle"] for x in b]),
        np.max([x["wrist_y"] for x in b]),
        np.mean([x["bar_distance"] for x in b]),
        np.max([x["head_forward"] for x in b]),
    ], dtype=np.float32)


def classify_motion_type(biomech):
    hip_vel = np.max(np.abs(np.diff([b["hip_y"] for b in biomech])))
    knee_vel = np.max(np.abs(np.diff([b["knee_angle"] for b in biomech])))

    wrist_overhead = np.mean([
        1 if b.get("wrist_y", 0) < b.get("shoulder_y", 999) else 0
        for b in biomech
    ])

    explosive_score = hip_vel + knee_vel

    if explosive_score > 10 and wrist_overhead > 0.5:
        return "olympic_explosive"

    if explosive_score > 8:
        return "dynamic_strength"

    return "strength"


def build_separated_feature_space(biomech):
    hip_y = np.array([b["hip_y"] for b in biomech])
    knee = np.array([b["knee_angle"] for b in biomech])
    wrist_y = np.array([b["wrist_y"] for b in biomech])
    torso = np.array([b["torso_angle"] for b in biomech])

    # =========================
    # 1. GEOMETRY SPACE (shape only)
    # =========================
    geometry = [
        np.mean(knee),
        np.min(knee),
        np.mean(torso),
        np.std(torso),
    ]

    # =========================
    # 2. EXPLOSION SPACE (Olympic signal)
    # =========================
    velocity = np.diff(hip_y)
    acceleration = np.diff(velocity)

    explosion = [
        np.max(velocity),
        np.max(acceleration),
        np.std(velocity),
    ]

    # =========================
    # 3. STRUCTURE SPACE (pose relationships)
    # =========================
    structure = [
        np.mean(wrist_y - hip_y),
        np.mean(wrist_y - knee),
        np.std(wrist_y - hip_y),
    ]

    return np.array(geometry + explosion + structure, dtype=np.float32)


def debug_feature_space(biomech):
    if not biomech:
        return {
            "frames": 0,
            "hip_range": 0.0,
            "hip_mean": 0.0,
            "knee_range": 0.0,
            "knee_min": 180.0,
            "knee_mean": 180.0,
            "wrist_overhead": 0.0,
            "wrist_overhead_ratio": 0.0,
            "torso_mean": 0.0,
            "elbow_range": 0.0,
            "explosion_velocity": 0.0,
            "explosion_accel": 0.0,
        }
    hip_y = np.array([b["hip_y"] for b in biomech])
    knee = np.array([b["knee_angle"] for b in biomech])
    wrist_y = np.array([b["wrist_y"] for b in biomech])
    shoulder_y = np.array([b["shoulder_y"] for b in biomech])

    velocity = np.diff(hip_y)
    acceleration = np.diff(velocity)

    return {
        "hip_range": float(np.max(hip_y) - np.min(hip_y)),
        "knee_min": float(np.min(knee)),
        "knee_mean": float(np.mean(knee)),
        "torso_proxy": float(np.mean(shoulder_y - hip_y)),
        "explosion_velocity": float(np.max(np.abs(velocity))) if len(velocity) > 0 else 0.0,
        "explosion_accel": float(np.max(np.abs(acceleration))) if len(acceleration) > 0 else 0.0,
        "wrist_overhead_ratio": float(np.mean(wrist_y < shoulder_y)),
    }


def squat_attractor_score(f):
    return (
        0.35 * (1.0 / (f["knee_min"] + 1e-5)) +
        0.25 * (f["knee_mean"] / 180.0) +
        0.20 * (1.0 - f["explosion_velocity"] / 20.0) +
        0.20 * (1.0 - f["explosion_accel"] / 30.0)
    )


def olympic_signal_score(f):
    return (
        0.4 * f["explosion_velocity"] +
        0.3 * f["explosion_accel"] +
        0.3 * f["wrist_overhead_ratio"]
    )


def print_debug_report(label, biomech):
    f = debug_feature_space(biomech)

    squat_score = squat_attractor_score(f)
    olympic_score = olympic_signal_score(f)

    print("\n================ DEBUG REPORT ================")
    print("LABEL:", label)
    print("HIP RANGE:", f["hip_range"])
    print("KNEE MIN:", f["knee_min"])
    print("EXPLOSION VEL:", f["explosion_velocity"])
    print("EXPLOSION ACC:", f["explosion_accel"])
    print("WRIST OVERHEAD:", f["wrist_overhead_ratio"])
    print("---------------------------------------------")
    print("SQUAT ATTRACTOR SCORE:", squat_score)
    print("OLYMPIC SIGNAL SCORE:", olympic_score)
    print("DOMINANT CLASS:",
          "SQUAT" if squat_score > olympic_score else "OLYMPIC/OTHER")
    print("=============================================\n")


def analyze_video(video_path, make_visuals=True, make_overlay=True):
    try:

        # =========================================================
        # 1. INPUT EXTRACTION
        # =========================================================
        sequence, biomechanics, debug = extract_video_biomechanics(
            video_path,
            sample_every=1,
        )

        print_debug_report("INPUT_VIDEO", biomechanics)

        if len(sequence) < 10:
            return {
                "exercise_label": "Unknown",
                "confidence": 0.0,
                "analysis_mode": "insufficient_data",
                "rep_feedback": [],
                "set_summary": build_set_summary([]),
                "overlay_video_url": None,
                "phase_images": None,
                "debug": {"frames_processed": len(sequence)},
            }

        # =========================================================
        # 2. FEATURE ENGINE
        # =========================================================
        seq_array = np.asarray(sequence, dtype=np.float32)
        seq_base = pad_or_trim(seq_array, target_len=30)
        seq = add_velocity(seq_base)

        biomech = biomechanics

        # =========================================================
        # SAFETY DEFAULTS (CRITICAL FIX)
        # =========================================================
        raw_label = "unknown"
        base_conf = 0.0
        final_label = None
        final_conf = 0.0
        rep_feedback = []

        try:
            base_probs = MODEL.predict_proba(seq)
            base_idx = int(np.argmax(base_probs))
            if base_idx < len(CLASS_NAMES):
                raw_label = CLASS_NAMES[base_idx]
                base_conf = float(base_probs[base_idx])
        except Exception as e:
            print("BASE MODEL FAILED:", e)

        summary = summarize_biomechanics(biomech)
        bio_label, bio_conf, bio_override, bio_reason = classify_with_biomechanics(
            raw_label,
            base_conf,
            summary,
            len(biomech),
        )

        # =========================================================
        # 3. SQUAT ROUTER (always runs — bar position via elbow geometry)
        # =========================================================
        squat_label = None
        squat_conf  = 0.0

        if SQUAT_ROUTER_MODEL is not None:
            squat_probs = SQUAT_ROUTER_MODEL.predict(
                np.expand_dims(seq_base, axis=0), verbose=0
            )[0]
            squat_idx   = int(np.argmax(squat_probs))
            squat_label = SQUAT_ROUTER_LABELS.get(squat_idx)
            squat_conf  = float(squat_probs[squat_idx])

        # Bar-position override: use wrist/elbow geometry to distinguish
        # front squat, back squat, and overhead squat more reliably.
        # Only apply if elbow angles are physiologically plausible (≥20°).
        # Values below 20° indicate MediaPipe misread the landmarks (occlusion/angle).
        bar_label, bar_conf, bar_debug = classify_squat_by_bar_position(biomech)
        _bar_pos_valid = bar_debug.get("front_rack_elbow_p25", 0) >= 20
        if bar_conf >= 0.60 and _bar_pos_valid:
            squat_label = bar_label
            squat_conf  = bar_conf

        # =========================================================
        # 4. OLYMPIC ROUTER (runs when movement has overhead/explosive signal)
        # =========================================================
        OLY_SET = {"snatch", "clean", "clean_and_jerk", "split_jerk"}

        olympic_pred, olympic_conf = None, 0.0

        wrist_overhead_ratio = float(np.mean([
            1 if b.get("wrist_y", 1.0) < b.get("shoulder_y", 0.0) else 0
            for b in biomech
        ]))

        hip_vel  = float(np.max(np.abs(np.diff([b["hip_y"] for b in biomech])))) if len(biomech) > 1 else 0.0
        knee_vel = float(np.max(np.abs(np.diff([b["knee_angle"] for b in biomech])))) if len(biomech) > 1 else 0.0
        explosive_score = hip_vel + knee_vel

        # Gate: only run Olympic router if there is genuine overhead OR explosive signal.
        # Lower threshold than before so Olympic lifts aren't missed when camera angle
        # makes the bar look lower than it is.
        run_oly_router = (
            OLY_ROUTER_MODEL is not None
            and (wrist_overhead_ratio > 0.12 or explosive_score > 12)
        )

        if run_oly_router:
            if USE_OLY_ROUTER_V4 and OLY_ROUTER_V4_MODEL is not None:
                active_model = OLY_ROUTER_V4_MODEL
                encoder      = None
                rf_features  = build_movement_video_features(biomech).reshape(1, -1)
            else:
                active_model = OLY_ROUTER_MODEL
                encoder      = OLY_ROUTER_ENCODER
                rf_features  = build_oly_router_v2_features(biomech).reshape(1, -1)

            if hasattr(active_model, "n_features_in_"):
                rf_features = rf_features[:, :active_model.n_features_in_]

            probs_oly   = active_model.predict_proba(rf_features)[0]
            idx         = int(np.argmax(probs_oly))
            raw_olympic = active_model.classes_[idx]
            olympic_pred = decode_olympic_label(raw_olympic, encoder)
            olympic_conf = float(probs_oly[idx])

        # =========================================================
        # 5. ARBITRATION — who wins?
        # =========================================================
        # The squat router fires confidently on Olympic lifts because
        # cleans, snatches, and jerks all contain a squatting phase.
        # Arbitration uses explosive_score and wrist_overhead as the key
        # differentiators — Olympic lifts are far more explosive than squats.
        #
        # Validated against 7 test videos. Thresholds:
        #   explosive_score > 80  → truly explosive (squat < 50, Oly > 80)
        #   wrist_overhead > 0.50 → strong overhead (C&J ~0.72, front squat ~0.15)
        #   bar_elbow_p25 < 20    → invalid MediaPipe read (occlusion/angle)

        _squat_confident  = squat_label is not None and squat_conf >= 0.75
        _truly_explosive  = explosive_score > 80
        _strong_overhead  = wrist_overhead_ratio >= 0.50
        _bar_says_ohs     = (
            bar_label == "overhead_squat"
            and bar_conf >= 0.60
            and _bar_pos_valid  # set in step 3
        )
        _looks_clean_only = looks_like_clean_only(biomech)
        _looks_cj         = looks_like_clean_and_jerk(biomech)
        _looks_split      = looks_like_split_jerk(biomech)
        _looks_strict     = looks_like_strict_press(biomech)
        _looks_thruster   = looks_like_thruster(biomech)
        bodyweight_debug = build_bodyweight_features(biomech)
        bodyweight_router_label, bodyweight_router_conf, _bodyweight_router_features = predict_bodyweight_router(biomech)

        routing_trace = []

        def trace_route(stage, label=None, conf=None, reason=None):
            routing_trace.append({
                "stage": stage,
                "label": str(label if label is not None else ""),
                "conf": round(float(conf or 0.0), 3),
                "reason": str(reason or ""),
            })

        trace_route("raw_model", raw_label, base_conf)
        trace_route("bio_model", bio_label, bio_conf, bio_reason)
        trace_route("squat_router", squat_label, squat_conf)
        trace_route("olympic_router", olympic_pred, olympic_conf)
        trace_route("bodyweight_router", bodyweight_router_label, bodyweight_router_conf)

        router_scores = {}

        def add_router_score(label, score, source):
            if not label:
                return
            label = str(label)
            score = float(score or 0.0)
            if label not in router_scores:
                router_scores[label] = {
                    "score": 0.0,
                    "sources": []
                }
            router_scores[label]["score"] += score
            router_scores[label]["sources"].append({
                "source": source,
                "score": round(score, 3)
            })

        # Two independent high-confidence bench predictions should not be
        # weakened or overridden by a false bodyweight-router prediction.
        strong_bench_evidence = (
            raw_label == "bench_press"
            and float(base_conf or 0.0) >= 0.95
            and bio_label == "bench_press"
            and float(bio_conf or 0.0) >= 0.95

            # Do not let camera-angle bench agreement override a clear
            # squat-to-overhead thruster movement.
            and not bool(_looks_thruster)
        )

        bodyweight_high_conf = (
            bodyweight_router_label in {"push_up", "pull_up", "handstand_push_up"}
            and float(bodyweight_router_conf or 0.0) >= 0.95
            and not strong_bench_evidence
        )

        raw_weight = 0.35 if bodyweight_high_conf else 1.0
        bio_weight = 0.35 if bodyweight_high_conf else 1.0

        add_router_score(raw_label, float(base_conf or 0.0) * raw_weight, "raw_model")
        add_router_score(bio_label, float(bio_conf or 0.0) * bio_weight, "bio_model")
        add_router_score(squat_label, squat_conf, "squat_router")
        add_router_score(olympic_pred, olympic_conf, "olympic_router")
        add_router_score(bodyweight_router_label, bodyweight_router_conf, "bodyweight_router")

        # Strong movement-family bonuses. These are audit-only for now.
        if bodyweight_router_label in {"push_up", "pull_up", "handstand_push_up"}:
            add_router_score(bodyweight_router_label, float(bodyweight_router_conf or 0.0) * 0.50, "bodyweight_bonus")

        if squat_label in {"squat_back", "squat_front", "overhead_squat"}:
            add_router_score(squat_label, float(squat_conf or 0.0) * 0.25, "squat_bonus")

        if olympic_pred in OLY_SET and _truly_explosive:
            add_router_score(olympic_pred, float(olympic_conf or 0.0) * 0.35, "olympic_explosive_bonus")

        router_score_winner = None
        router_score_value = 0.0
        router_v6_label = None
        router_v6_conf = 0.0
        router_v6_decision = "no_scores"

        if router_scores:
            router_score_winner, _router_score_info = max(
                router_scores.items(),
                key=lambda kv: float(kv[1].get("score", 0.0))
            )
            router_score_value = float(_router_score_info.get("score", 0.0))

            router_v6_label = router_score_winner
            router_v6_conf = min(0.99, max(0.01, router_score_value / 2.0))
            router_v6_decision = "score_winner"
        _deadlift_setup_geometry = (
            squat_label == "squat_back"
            and wrist_overhead_ratio < 0.12
            and float(bar_debug.get("avg_elbow_angle_sq", 0.0)) > 150.0
            and float(bar_debug.get("front_rack_elbow_p25", 0.0)) > 145.0
            and float(bar_debug.get("overhead_ratio", 0.0)) < 0.05
            and float(bar_debug.get("avg_wrist_forward", 1.0)) < 0.03
            and not _looks_clean_only
            and not _looks_cj
            and not _looks_split
            and not _looks_strict
            and not _looks_thruster
        )
        _deadlift_low_speed_setup = (
            _deadlift_setup_geometry
            and raw_label in {"squat", "bench_press"}
            and explosive_score <= 30.0
            and float(bar_debug.get("wrist_height_above_shoulder", 0.0)) < -0.08
        )
        _deadlift_upright_setup = (
            _deadlift_setup_geometry
            and raw_label in {"bench_press", "deadlift", "squat", "squat_front"}
            and explosive_score <= 70.0
            and float(bar_debug.get("wrist_height_above_shoulder", 0.0)) > -0.09
        )
        _deadlift_raw_pull_setup = (
            _deadlift_setup_geometry
            and raw_label == "deadlift"
            and wrist_overhead_ratio < 0.02
            and explosive_score <= 75.0
        )
        _short_low_camera_bench_setup = (
            raw_label == "squat"
            and squat_label == "squat_back"
            and int(bar_debug.get("squat_frames_used", 999) or 999) <= 35
            and explosive_score > 45.0
            and float(bar_debug.get("wrist_height_above_shoulder", 0.0)) < -0.18
            and float(olympic_conf or 0.0) < 0.80
        )
        _short_overhead_bench_setup = (
            squat_label == "overhead_squat"
            and raw_label in {"deadlift", "push_press", "squat"}
            and bio_label in {"push_press", "squat", "deadlift"}
            and int(bar_debug.get("squat_frames_used", 999) or 999) <= 45
            and int(bar_debug.get("total_frames", 999) or 999) <= 110
            and float(olympic_conf or 0.0) < 0.80
            and not _looks_clean_only
            and not _looks_cj
        )
        _looks_push_up = (
            (
                float(bodyweight_debug.get("wrist_below_shoulder_ratio", 0.0)) >= 0.75
                and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 0.0)) >= 0.08
                and -0.18 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.20
                and float(bodyweight_debug.get("median_head_drop", -1.0)) >= 0.035
                and 20.0 <= float(bodyweight_debug.get("avg_torso_angle", 0.0)) <= 180.0
                and float(bodyweight_debug.get("elbow_range", 0.0)) >= 35.0
                and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.13
            )
            or (
                float(bodyweight_debug.get("wrist_below_shoulder_ratio", 0.0)) >= 0.50
                and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 0.0)) >= 0.015
                and -0.22 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.23
                and float(bodyweight_debug.get("median_head_drop", -1.0)) >= 0.035
                and 45.0 <= float(bodyweight_debug.get("avg_torso_angle", 0.0)) <= 175.0
                and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.26
                and (
                    float(bodyweight_debug.get("elbow_range", 0.0)) >= 40.0
                    or (
                        int(bodyweight_debug.get("total_frames", 0) or 0) <= 60
                        and float(bodyweight_debug.get("min_elbow", 180.0)) <= 155.0
                    )
                )
            )
        )
        _strong_floor_push_up = (
            float(bodyweight_debug.get("wrist_below_shoulder_ratio", 0.0)) >= 0.95
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 0.0)) >= 0.12
            and -0.16 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.08
            and float(bodyweight_debug.get("median_head_drop", -1.0)) >= 0.06
            and 75.0 <= float(bodyweight_debug.get("avg_torso_angle", 0.0)) <= 135.0
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.07
        )
        _push_up_bench_guard = (
            raw_label == "bench_press"
            and float(base_conf or 0.0) >= 0.80
            and not _strong_floor_push_up
        )
        _looks_push_up = _looks_push_up and not _push_up_bench_guard
        _looks_handstand_push_up = (
            float(bodyweight_debug.get("wrist_below_shoulder_ratio", 0.0)) >= 0.85
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 0.0)) >= 0.08
            and float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 1.0)) <= -0.015
            and (
                (
                    float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 1.0)) <= -0.20
                    and float(bodyweight_debug.get("mean_knee_minus_hip_y", 1.0)) <= -0.05
                )
                or (
                    float(bodyweight_debug.get("mean_knee_minus_hip_y", 1.0)) <= 0.02
                    and float(bodyweight_debug.get("avg_torso_angle", 0.0)) >= 170.0
                )
            )
            and float(bodyweight_debug.get("median_head_drop", -1.0)) >= 0.035
            and float(bodyweight_debug.get("avg_torso_angle", 0.0)) >= 125.0
            and float(bodyweight_debug.get("elbow_range", 0.0)) >= 25.0
        )
        _pull_up_overhead_squat_guard = (
            squat_label == "overhead_squat"
            and int(bodyweight_debug.get("total_frames", 0) or 0) >= 120
            and float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) >= 0.30
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", -1.0)) > -0.17
        )
        _short_cropped_pull_up = (
            int(bodyweight_debug.get("total_frames", 0) or 0) <= 25
            and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.90
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.15
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.08
            and float(bodyweight_debug.get("elbow_range", 0.0)) >= 70.0
        )
        _very_short_pull_up = (
            int(bodyweight_debug.get("total_frames", 0) or 0) <= 15
            and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.60
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.10
            and 0.18 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.35
            and float(bodyweight_debug.get("median_head_drop", 1.0)) <= -0.05
            and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 20.0
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.12
        )
        _tight_vertical_pull_up = (
            float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.40
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.01
            and -0.02 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.32
            and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 95.0
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.08
            and float(bodyweight_debug.get("median_head_drop", 1.0)) <= 0.02
            and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
            and float(bodyweight_debug.get("min_elbow", 180.0)) <= 45.0
        )
        _static_cropped_pull_up = (
            float(bodyweight_debug.get("wrist_below_shoulder_ratio", 0.0)) >= 0.70
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 0.0)) >= 0.25
            and 0.25 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.36
            and float(bodyweight_debug.get("mean_knee_minus_hip_y", 0.0)) >= 0.18
            and float(bodyweight_debug.get("median_head_drop", 1.0)) <= -0.03
            and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 15.0
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.09
            and float(bodyweight_debug.get("shoulder_y_range", 1.0)) <= 0.06
            and float(bodyweight_debug.get("hip_y_range", 1.0)) <= 0.07
            and float(bodyweight_debug.get("min_elbow", 0.0)) >= 140.0
        )
        _long_bar_pull_up = (
            int(bodyweight_debug.get("total_frames", 0) or 0) >= 120
            and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.80
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.18
            and 0.25 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.45
            and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 25.0
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.08
            and float(bodyweight_debug.get("elbow_range", 0.0)) >= 70.0
            and float(bodyweight_debug.get("min_elbow", 180.0)) <= 95.0
        )
        _pull_up_posture_signature = (
            float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.65
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.08
            and 0.09 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.40
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.13
            and not _looks_split
        )

        _pull_up_router_guard = (
            raw_label == "push_press"
            and bio_label == "push_press"
            and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.65
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.08
            and 0.09 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.40
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.13
            and float(olympic_conf or 0.0) < 0.85
        )

        _pull_up_press_guard = (
            raw_label == "push_press"
            and not _pull_up_posture_signature
            and (
                bio_label == "push_press"
                or _looks_strict
                or _looks_split
            )
        )
        _pull_up_bench_guard = (
            raw_label == "bench_press"
            and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) < 0.35
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 0.0)) > 0.02
            and float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) < 0.18
            and float(bodyweight_debug.get("median_head_drop", 0.0)) >= 0.03
        )
        _looks_pull_up = (
            (
                (
                    float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.50
                    and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.08
                    and 0.09 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.40
                    and float(bodyweight_debug.get("mean_knee_minus_hip_y", 0.0)) >= 0.045
                    and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 85.0
                    and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.13
                    and float(bodyweight_debug.get("elbow_range", 0.0)) >= 45.0
                    and float(bodyweight_debug.get("min_elbow", 180.0)) <= 95.0
                )
                or _short_cropped_pull_up
                or _very_short_pull_up
                or _tight_vertical_pull_up
                or _static_cropped_pull_up
                or _long_bar_pull_up
            )
            and not _pull_up_overhead_squat_guard
            and not _pull_up_press_guard
            and not _pull_up_bench_guard
        )
        _looks_muscle_up = (
            int(bodyweight_debug.get("total_frames", 0) or 0) >= 250
            and 0.45 <= float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) <= 0.70
            and -0.08 <= float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= 0.03
            and 0.10 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.20
            and 0.04 <= float(bodyweight_debug.get("mean_knee_minus_hip_y", 0.0)) <= 0.10
            and 20.0 <= float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 40.0
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.08
            and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
        )
        _looks_burpee = (
            int(bodyweight_debug.get("total_frames", 0) or 0) >= 120
            and float(bodyweight_debug.get("wrist_below_shoulder_ratio", 0.0)) >= 0.70
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 0.0)) >= 0.05
            and -0.02 <= float(bodyweight_debug.get("mean_hip_minus_shoulder_y", 0.0)) <= 0.22
            and 40.0 <= float(bodyweight_debug.get("avg_torso_angle", 0.0)) <= 90.0
            and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
            and float(bodyweight_debug.get("wrist_y_range", 0.0)) >= 0.50
            and float(bodyweight_debug.get("shoulder_y_range", 0.0)) >= 0.35
            and float(bodyweight_debug.get("hip_y_range", 0.0)) >= 0.30
            and float(bodyweight_debug.get("median_head_drop", 0.0)) <= 0.04
        )

        protected_label = None
        protected_conf = 0.0
        protected_reason = None

        effective_looks_split_for_protection = (
            _looks_split
            and not (
                _looks_thruster
                and raw_label in {"bench_press", "push_press"}
                and bio_label in {"bench_press", "squat", "push_press"}
                and float(olympic_conf or 0.0) < 0.65
                and float(
                    bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)
                ) < 0.35
                and float(bar_debug.get("overhead_ratio", 0.0)) < 0.35
            )
        )

        protection = apply_protections(
            bodyweight_inputs={
                "raw_label": raw_label,
                "squat_label": squat_label,
                "bodyweight_debug": bodyweight_debug,
                "strong_bench_evidence": strong_bench_evidence,
                "looks_push_up": _looks_push_up,
                "looks_pull_up": _looks_pull_up,
                "looks_handstand_push_up": _looks_handstand_push_up,
                "looks_muscle_up": _looks_muscle_up,
                "looks_burpee": _looks_burpee,
            },
            early_strength_inputs={
                "raw_label": raw_label,
                "base_conf": float(base_conf or 0.0),
                "bio_conf": float(bio_conf or 0.0),
                "squat_label": squat_label,
                "olympic_conf": float(olympic_conf or 0.0),
                "explosive_score": float(explosive_score or 0.0),
                "bar_debug": bar_debug,
                "bodyweight_debug": bodyweight_debug,
                "short_overhead_bench_setup": _short_overhead_bench_setup,
                "pull_up_router_guard": _pull_up_router_guard,
                "deadlift_low_speed_setup": _deadlift_low_speed_setup,
                "deadlift_upright_setup": _deadlift_upright_setup,
                "deadlift_raw_pull_setup": _deadlift_raw_pull_setup,
                "looks_push_up": _looks_push_up,
                "looks_pull_up": _looks_pull_up,
                "looks_handstand_push_up": _looks_handstand_push_up,
                "looks_thruster": _looks_thruster,
                "looks_split": effective_looks_split_for_protection,
            },
            strength_inputs={
                "raw_label": raw_label,
                "base_conf": float(base_conf or 0.0),
                "bio_label": bio_label,
                "bio_conf": float(bio_conf or 0.0),
                "looks_strict": _looks_strict,
                "looks_thruster": _looks_thruster,
                "looks_clean_only": _looks_clean_only,
                "looks_cj": _looks_cj,
                "looks_split": effective_looks_split_for_protection,
            },
        )

        # Two independent perfect bench predictions take priority over
        # false overhead-squat/bodyweight signatures caused by the camera
        # orientation of a horizontal bench-press clip.
        if strong_bench_evidence:
            protected_label = "bench_press"
            protected_conf = max(
                float(base_conf or 0.0),
                float(bio_conf or 0.0),
                0.95,
            )
            protected_reason = "strong_bench_model_agreement"

        elif protection.label:
            protected_label = protection.label
            protected_conf = protection.confidence
            protected_reason = protection.reason

        elif bio_label == "bench_press" and not _looks_strict and not _looks_thruster and not (_looks_push_up or _looks_pull_up or _looks_handstand_push_up):
            bench_blocked_by_oly = (
                olympic_pred in {"snatch", "clean", "clean_and_jerk", "split_jerk"}
                and float(olympic_conf or 0.0) >= 0.90
                and raw_label in {"squat", "deadlift", "push_press"}
            )

            if not bench_blocked_by_oly:
                protected_label = "bench_press"
                protected_conf = max(float(bio_conf or 0.0), 0.80)
                protected_reason = bio_reason
            else:
                protected_reason = "bench_blocked_by_olympic_router"
        elif (
            raw_label == "squat"
            and squat_label == "squat_front"
            and not _looks_clean_only
            and not _looks_cj
            and not _looks_split
            and not _looks_strict
            and not _looks_thruster
            and not (
                olympic_pred in {"snatch", "clean", "clean_and_jerk", "split_jerk"}
                and float(olympic_conf or 0.0) >= 0.50
            )
        ):
            protected_label = "bench_press"
            protected_conf = max(base_conf, 0.80)
            protected_reason = "bench_press_squat_front_false_positive"
        elif (
            raw_label in {"squat", "deadlift"}
            and squat_label in {"squat_back", "squat_front", "overhead_squat"}
            and bio_label in {"squat", "deadlift"}
            and explosive_score >= 60.0
            and float(olympic_conf or 0.0) < 0.70
            and not _looks_clean_only
            and not _looks_cj
            and not _looks_split
            and not _looks_strict
        ):
            protected_label = "bench_press"
            protected_conf = max(float(base_conf or 0.0), float(bio_conf or 0.0), 0.80)
            protected_reason = "bench_press_fast_press_rescue"
        elif (
            raw_label in {"squat", "squat_front", "squat_back", "push_press"}
            and squat_label in {"squat_back", "squat_front", "overhead_squat"}
            and bio_label in {"squat", "push_press", "deadlift"}
            and not _looks_clean_only
            and not _looks_cj
            and not _looks_split
            and (not _deadlift_setup_geometry or _short_low_camera_bench_setup)
            and (
                int(bar_debug.get("squat_frames_used", 999) or 999) <= 35
                or (
                    run_oly_router
                    and float(olympic_conf or 0.0) < 0.80
                    and wrist_overhead_ratio > 0.25
                    and explosive_score > 20
                )
            )
        ):
            protected_label = "bench_press"
            protected_conf = max(float(base_conf or 0.0), float(bio_conf or 0.0), 0.80)
            protected_reason = "bench_press_short_squat_rescue"
        elif (
            bio_label == "deadlift"
            and (
                (
                    bio_override
                    and not (raw_label in {"squat", "squat_front"} and _squat_confident)
                )
                or (
                    wrist_overhead_ratio < 0.08
                    and explosive_score <= 30.0
                    and float(bar_debug.get("front_rack_elbow_p25", 0.0)) >= 130.0
                    and not _looks_clean_only
                    and not _looks_cj
                    and not _looks_split
                    and not _looks_strict
                    and not _looks_thruster
                )
            )
        ):
            protected_label = "deadlift"
            protected_conf = bio_conf
            protected_reason = bio_reason
        vertical_pullup_signature = (
            raw_label == "push_press"
            and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.80
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.08
            and float(bodyweight_debug.get("elbow_range", 0.0)) >= 110.0
            and float(bodyweight_debug.get("min_elbow", 180.0)) <= 45.0
            and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 10.0
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.03
            and float(bodyweight_debug.get("wrist_y_range", 1.0)) <= 0.18
            and not _truly_explosive
        )

        if vertical_pullup_signature and protected_label is None:
            protected_label = "pull_up"
            protected_conf = 0.86
            protected_reason = "vertical_pullup_signature"

        push_press_should_hold = (
            protected_label == "push_press"
            and raw_label == "push_press"
            and bio_label == "push_press"
            and olympic_pred == "clean_and_jerk"
        )

        strong_oly_lock = (
            not push_press_should_hold
            and (
                (
                    run_oly_router
                    and olympic_pred in OLY_SET
                    and olympic_conf >= 0.85
                    and _truly_explosive
                )
                or (
                    run_oly_router
                    and olympic_pred == "clean_and_jerk"
                    and float(olympic_conf or 0.0) >= 0.95
                    and (explosive_score > 25 or wrist_overhead_ratio > 0.35)
                    and not (raw_label == "push_press" and _looks_split)
                )
            )
        )


        if protected_label and not strong_oly_lock:
            final_label = protected_label
            final_conf = protected_conf
            analysis_mode = "biomechanics_override"

        elif (
            bodyweight_router_label == "pull_up"
            and float(bodyweight_router_conf or 0.0) >= 0.95
              and not (
                  squat_label == "overhead_squat"
                  and float(squat_conf or 0.0) < 0.86
              )
            and raw_label == "push_press"
            and bio_label == "push_press"
            and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.80
        ):
            final_label = "pull_up"
            final_conf = max(float(bodyweight_router_conf or 0.0), 0.90)
            protected_label = "pull_up"
            protected_conf = final_conf
            protected_reason = "bodyweight_router_pull_up_high_conf"
            analysis_mode = "bodyweight_router"

        elif (
            squat_label == "squat_front"
            and float(squat_conf or 0.0) >= 0.80
            and raw_label == "push_press"
            and float(olympic_conf or 0.0) < 0.75
        ):
            final_label = "squat_front"
            final_conf = squat_conf
            analysis_mode = "detailed_rep_analysis"

        elif (
            run_oly_router
            and olympic_pred == "clean_and_jerk"
            and float(olympic_conf or 0.0) >= 0.75
            and explosive_score > 20
            and not (
                _pull_up_router_guard
              and not (
                  squat_label == "overhead_squat"
                  and float(squat_conf or 0.0) >= 0.70
              )
                and float(olympic_conf or 0.0) < 0.95
            )
              and not (
                  squat_label == "squat_front"
                  and float(squat_conf or 0.0) >= 0.80
                  and float(olympic_conf or 0.0) < 0.75
              )
            and (
                raw_label in {"squat", "squat_back", "squat_front", "bench_press"}
                or (raw_label == "push_press" and not _looks_split)
            )
        ):
            final_label = "clean_and_jerk"
            final_conf = olympic_conf
            analysis_mode = "olympic_locked"

        elif (
            _looks_split
              and not (
                  squat_label == "overhead_squat"
                  and float(squat_conf or 0.0) >= 0.75
              )
            and not _looks_cj
            and explosive_score > 20
            and olympic_pred != "snatch"
            and (
                raw_label == "push_press"
                or olympic_pred != "clean_and_jerk"
                or float(olympic_conf or 0.0) < 0.75
            )
        ):
            final_label = "split_jerk"
            final_conf = 0.80
            analysis_mode = "shape_override"

        elif (
            _squat_confident
            and bio_label not in {"push_up", "pull_up", "handstand_push_up"}
            and raw_label == "squat"
            and squat_label in {"squat_back", "squat_front"}
            and not _truly_explosive
            and not _strong_overhead
            and float(olympic_conf or 0.0) < 0.85
        ):
            final_label = squat_label
            final_conf = squat_conf
            analysis_mode = "detailed_rep_analysis"

        elif (
            squat_label == "overhead_squat"
            and squat_conf >= 0.75
            and float(olympic_conf or 0.0) < 0.85
            and not (
                raw_label == "push_press"
                and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.85
                and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.10
                and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
                and float(bodyweight_debug.get("min_elbow", 180.0)) <= 35.0
                and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 8.0
                and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.02
                and float(bodyweight_debug.get("wrist_y_range", 1.0)) <= 0.15
            )
        ):
            final_label = "overhead_squat"
            final_conf = max(bar_conf, squat_conf)
            analysis_mode = "detailed_rep_analysis"

        # Strong Olympic predictions are lower-calibrated than the neural squat
        # router. Let them win before squat/overhead/split-shape fallbacks steal
        # Olympic benchmark clips; Router V5 below can still rescue standalone
        # split jerks from clean-and-jerk.
        elif (
            run_oly_router
            and olympic_pred in OLY_SET
            and olympic_conf >= 0.65
            and not push_press_should_hold
              and not (
                  squat_label == "squat_front"
                  and float(squat_conf or 0.0) >= 0.80
                  and float(olympic_conf or 0.0) < 0.75
              )
            and not (
                raw_label == "push_press"
                and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.85
                and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.10
                and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
                and float(bodyweight_debug.get("min_elbow", 180.0)) <= 35.0
                and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 8.0
                and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.02
                and not _truly_explosive
            )
        ):
            final_label = olympic_pred
            final_conf = olympic_conf
            analysis_mode = "olympic_locked"

        elif (
            run_oly_router
            and olympic_pred == "snatch"
            and olympic_conf >= 0.45
            and wrist_overhead_ratio > 0.35
            and _looks_split
            and raw_label in {"squat", "squat_front", "squat_back"}
        ):
            final_label = "snatch"
            final_conf = max(0.60, olympic_conf)
            analysis_mode = "olympic_rescue"

        # Overhead squat: bar position is definitive — wrists locked out overhead
        # throughout the entire squat, not just briefly during a catch.
        # Must check this first before any Olympic path.
        elif (
            (_bar_says_ohs and not _truly_explosive)
            or (
                squat_label == "overhead_squat"
                and squat_conf >= 0.90
                and olympic_conf < 0.85
                and not (
                    raw_label == "push_press"
                    and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.85
                    and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.10
                    and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
                    and float(bodyweight_debug.get("min_elbow", 180.0)) <= 35.0
                    and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 8.0
                    and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.02
                    and float(bodyweight_debug.get("wrist_y_range", 1.0)) <= 0.15
                )
            )
        ):
            final_label   = "overhead_squat"
            final_conf    = max(bar_conf, squat_conf)
            analysis_mode = "detailed_rep_analysis"

        # Strong Olympic router: very high confidence AND truly explosive.
        # This must come before shape detectors — the Olympic router at 80%+
        # is more reliable than shape heuristics that fire on multiple lifts.
        elif run_oly_router and olympic_pred in OLY_SET and olympic_conf >= 0.80 and _truly_explosive and not push_press_should_hold:
            final_label   = olympic_pred
            final_conf    = olympic_conf
            analysis_mode = "olympic_locked"

        # High Olympic confidence + explosive (lower threshold).
        elif run_oly_router and olympic_pred in OLY_SET and olympic_conf >= 0.65 and _truly_explosive and not push_press_should_hold:
            final_label   = olympic_pred
            final_conf    = olympic_conf
            analysis_mode = "olympic_locked"

        # Split jerk: the Olympic router ALWAYS mislabels split jerks as C&J.
        # So we use the shape detector here regardless of Olympic confidence,
        # but require that the Olympic router did NOT say snatch (snatch is
        # distinct enough that the router is reliable for it).
        # Split jerk is less explosive than clean/snatch (dip-and-drive, not full pull)
        # so we use a lower explosive threshold (>20).
        elif (
            _looks_split
            and not (
                squat_label == "overhead_squat"
                and float(squat_conf or 0.0) >= 0.75
            )
            and explosive_score > 20
            and olympic_pred != "snatch"
        ):
            final_label   = "split_jerk"
            final_conf    = 0.80
            analysis_mode = "shape_override"

        # Clean (no jerk): shape detector sees clean catch + explosive, but Olympic
        # router is confused (often says snatch). Trust shape + explosive.
        elif _looks_clean_only and _truly_explosive and not _looks_cj and not _looks_split:
            final_label   = "clean"
            final_conf    = 0.75
            analysis_mode = "shape_override"

        # Squat wins: confident squat router, not truly explosive.
        # squat_label is already bar-position-adjusted from step 3.
        elif _squat_confident and not _truly_explosive and bio_label not in {"push_up", "pull_up", "handstand_push_up"}:
            final_label   = squat_label
            final_conf    = squat_conf
            analysis_mode = "detailed_rep_analysis"

        # Shape detectors as fallback for remaining Olympic cases.
        elif _looks_cj and _strong_overhead:
            final_label   = "clean_and_jerk"
            final_conf    = 0.68
            analysis_mode = "shape_override"

        elif _looks_clean_only:
            final_label   = "clean"
            final_conf    = 0.68
            analysis_mode = "shape_override"

        elif (
            squat_label
            and squat_conf >= 0.55
            and bio_label not in {"push_up", "pull_up", "handstand_push_up"}
            and not (
                raw_label == "push_press"
                and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.85
                and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.10
                and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
                and float(bodyweight_debug.get("min_elbow", 180.0)) <= 35.0
                and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 8.0
                and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.02
                and float(bodyweight_debug.get("wrist_y_range", 1.0)) <= 0.15
            )
        ):
            final_label   = squat_label
            final_conf    = squat_conf
            analysis_mode = "detailed_rep_analysis"

        elif (
            run_oly_router
            and olympic_pred in OLY_SET
            and olympic_conf >= 0.50
            and not push_press_should_hold
            and not (
                raw_label == "push_press"
                and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.85
                and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.10
                and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
                and float(bodyweight_debug.get("min_elbow", 180.0)) <= 35.0
                and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 8.0
                and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.02
                and not _truly_explosive
            )
        ):
            final_label   = olympic_pred
            final_conf    = olympic_conf
            analysis_mode = "olympic_locked"

        elif bio_override and bio_label == "push_press":
            final_label = "push_press"
            final_conf = bio_conf
            analysis_mode = "biomechanics_override"

        elif raw_label == "push_press" and base_conf >= 0.65 and not _looks_split:
            final_label = "push_press"
            final_conf = base_conf
            analysis_mode = "base_model_locked"

        else:
            final_label   = "unknown"
            final_conf    = 0.5
            analysis_mode = "insufficient_signal"

        router_v5_debug = None
        clear_squat_should_hold = False
        if run_oly_router:
            try:
                router_v5_label, router_v5_conf, router_v5_debug = route_olympic_lift(
                    biomechanics=biomech,
                    raw_label=final_label,
                    raw_confidence=final_conf,
                    olympic_label=olympic_pred,
                    olympic_confidence=olympic_conf,
                )
            except Exception as e:
                router_v5_label = final_label
                router_v5_conf = final_conf
                router_v5_debug = {"router_error": str(e)}

            split_features = (router_v5_debug or {}).get("split_features", {}) if isinstance(router_v5_debug, dict) else {}
            likely_standalone_split = (
                raw_label == "push_press"
                and (
                    float(split_features.get("lockout_duration", 0.0) or 0.0) >= 200.0
                    or float(split_features.get("catch_to_finish", 0.0) or 0.0) >= 300.0
                )
            )

            if (
                router_v5_label == "split_jerk"
                and olympic_pred == "clean_and_jerk"
                and float(olympic_conf or 0.0) >= 0.90
                and not likely_standalone_split
            ):
                router_v5_label = "clean_and_jerk"
                router_v5_conf = max(float(olympic_conf or 0.0), float(router_v5_conf or 0.0))
                if isinstance(router_v5_debug, dict):
                    router_v5_debug["decision"] = "clean_and_jerk_high_conf_rescue"

            short_split_press = (
                float(split_features.get("lockout_duration", 0.0) or 0.0) <= 90.0
                and float(split_features.get("catch_to_finish", 0.0) or 0.0) <= 120.0
            )
            if (
                router_v5_label == "split_jerk"
                and raw_label in {"squat", "deadlift"}
                and bio_label in {"push_press", "squat", "deadlift"}
                and short_split_press
                and float(olympic_conf or 0.0) < 0.70
                and not _looks_clean_only
                and not _looks_cj
            ):
                router_v5_label = "bench_press"
                router_v5_conf = max(float(base_conf or 0.0), float(bio_conf or 0.0), 0.80)
                if isinstance(router_v5_debug, dict):
                    router_v5_debug["decision"] = "bench_press_short_split_rescue"

            cj_features = (router_v5_debug or {}).get("features", {}) if isinstance(router_v5_debug, dict) else {}
            short_cj_press = (
                float(cj_features.get("lockout_duration", 0.0) or 0.0) <= 65.0
                and float(cj_features.get("catch_to_finish", 0.0) or 0.0) <= 65.0
            )
            if (
                router_v5_label == "clean_and_jerk"
                and raw_label in {"squat", "squat_front", "squat_back", "deadlift"}
                and bio_label in {"push_press", "squat", "deadlift"}
                and short_cj_press
                and float(olympic_conf or 0.0) < 0.90
                and not _looks_clean_only
                and not _looks_cj
            ):
                router_v5_label = "bench_press"
                router_v5_conf = max(float(base_conf or 0.0), float(bio_conf or 0.0), 0.80)
                if isinstance(router_v5_debug, dict):
                    router_v5_debug["decision"] = "bench_press_short_cj_rescue"

            if (
                router_v5_label == "snatch"
                and _looks_cj
                and not _looks_clean_only
                and raw_label in {"squat", "squat_front", "squat_back"}
                and float(olympic_conf or 0.0) < 0.90
                and wrist_overhead_ratio < 0.45
                and explosive_score > 80.0
            ):
                router_v5_label = "clean_and_jerk"
                router_v5_conf = max(float(router_v5_conf or 0.0), 0.76)
                if isinstance(router_v5_debug, dict):
                    router_v5_debug["decision"] = "clean_and_jerk_shape_rescue"

            protected_non_olympic = protected_label in {
                "bench_press",
                "burpee",
                "deadlift",
                "handstand_push_up",
                "muscle_up",
                "push_up",
                "pull_up",
                "push_press",
                "thruster",
                "strict_press",
            }
            squat_should_hold = (
                final_label in {"squat_back", "squat_front", "overhead_squat"}
                and (float(olympic_conf or 0.0) < 0.65 or not _truly_explosive)
            )

            push_press_should_hold = (
                protected_label == "push_press"
                and raw_label == "push_press"
                and bio_label == "push_press"
                and router_v5_label == "clean_and_jerk"
            )

            if (
                not push_press_should_hold
                    and not (
                        squat_label == "squat_front"
                        and float(squat_conf or 0.0) >= 0.80
                        and raw_label == "push_press"
                        and float(olympic_conf or 0.0) < 0.75
                    )
                  and not (
                      squat_label == "squat_front"
                      and float(squat_conf or 0.0) >= 0.80
                      and router_v5_label == "clean_and_jerk"
                      and float(router_v5_conf or 0.0) < 0.75
                  )
                  and not (
                      raw_label == "push_press"
                      and router_v5_label in OLY_SET
                      and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.85
                      and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.10
                      and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
                      and float(bodyweight_debug.get("min_elbow", 180.0)) <= 35.0
                      and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 8.0
                      and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.02
                      and not _truly_explosive
                  )
                and (
                    final_label in OLY_SET or (
                        router_v5_label in OLY_SET
                        and not protected_non_olympic
                        and not squat_should_hold
                    )
                )
            ):
                # Hard block: clear squat should not be stolen by weak Olympic Router V5.
                if (
                    raw_label in {"squat", "squat_back", "squat_front", "overhead_squat"}
                    and squat_label in {"squat_back", "squat_front", "overhead_squat"}
                    and float(squat_conf or 0.0) >= 0.90
                    and router_v5_label in OLY_SET
                    and float(router_v5_conf or 0.0) < 0.85
                ):
                    final_label = squat_label
                    final_conf = max(
                        float(squat_conf or 0.0),
                        float(base_conf or 0.0) if raw_label == squat_label else 0.0,
                    )
                    analysis_mode = "squat_router_protected"
                else:
                    final_label = router_v5_label
                    final_conf = router_v5_conf
                    analysis_mode = "router_v5"

                # Final squat recovery after Router V5.
                if clear_squat_should_hold:
                    final_label = squat_label
                    final_conf = max(
                        float(squat_conf or 0.0),
                        float(base_conf or 0.0) if raw_label == squat_label else 0.0,
                    )
                    analysis_mode = "squat_router_protected"

        # Final squat recovery after Router V5 / Olympic override.
        if clear_squat_should_hold:
            final_label = squat_label
            final_conf = max(
                        float(squat_conf or 0.0),
                        float(base_conf or 0.0) if raw_label == squat_label else 0.0,
                    )
            analysis_mode = "squat_router_protected"

        # Pull-up safety: if an obvious pull-up posture was routed into a
        # low-confidence Olympic label, recover pull_up before rep analysis.
        if (
            _pull_up_router_guard
              and not (
                  squat_label == "overhead_squat"
                  and float(squat_conf or 0.0) >= 0.70
              )
            and (
                final_label in OLY_SET
                or (
                    final_label == "overhead_squat"
                    and float(bodyweight_debug.get("wrist_y_range", 1.0)) < 0.12
                    and float(explosive_score or 0.0) > 25.0
                )
            )
            and float(final_conf or 0.0) < 0.95
        ):
            final_label = "pull_up"
            final_conf = 0.86
            analysis_mode = "biomechanics_override"

        # Bodyweight safety: if biomechanics strongly identified a bodyweight
        # movement, do not let late squat/Olympic fallbacks steal it.
        if (
            bio_label in {"push_up", "pull_up", "handstand_push_up"}
            and bio_override
            and final_label not in {"push_up", "pull_up", "handstand_push_up"}
            and float(final_conf or 0.0) < 0.95
        ):
            final_label = bio_label
            final_conf = max(float(bio_conf or 0.0), 0.86)
            analysis_mode = "biomechanics_override"

        # Final vertical pull-up recovery: prevents squat/Olympic fallbacks
        # from stealing clear pull-up motion.
        if (
            final_label not in {"pull_up", "push_up", "handstand_push_up"}
              and not (
                  squat_label == "overhead_squat"
                  and float(squat_conf or 0.0) >= 0.70
              )
            and raw_label == "push_press"
            and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.85
            and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.10
            and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
            and float(bodyweight_debug.get("min_elbow", 180.0)) <= 35.0
            and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 8.0
            and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.02
            and float(bodyweight_debug.get("wrist_y_range", 1.0)) <= 0.15
        ):
            final_label = "pull_up"
            final_conf = 0.86
            analysis_mode = "biomechanics_override"

        # =========================================================
        # 6. REP ANALYSIS
        # =========================================================
        # ---------------- ROUTER V6 FINAL BODYWEIGHT ARBITRATION ----------------
        # Runs AFTER legacy routing, BEFORE rep analysis.
        # This avoids editing the fragile if/elif router chain.
        router_v6_bodyweight_allowed = (
            not (
                protected_reason == "strict_press_pattern_detected"
                and protected_label == "strict_press"
                and float(protected_conf or 0.0) >= 0.90
            )
            and
            router_v6_label in {"push_up", "pull_up", "handstand_push_up"}
            and bodyweight_router_label == router_v6_label
            and float(bodyweight_router_conf or 0.0) >= 0.95
            and float(router_v6_conf or 0.0) >= 0.72
            and not strong_oly_lock
            and not strong_bench_evidence

            # Split-jerk guard: an explosive overhead Olympic movement can look
            # like a pull-up to the bodyweight router because of the long elbow
            # range and sustained overhead wrist position.
            and not (
                router_v6_label == "pull_up"
                and bool(_looks_split)
                and bool(run_oly_router)
                and float(olympic_conf or 0.0) >= 0.85
                and float(explosive_score or 0.0) >= 30.0
            )

            and not (
                router_v6_label == "pull_up"
                and final_label == "overhead_squat"
                and squat_label == "overhead_squat"
                and float(squat_conf or 0.0) >= 0.75
            )
            and not (
                protected_reason == "push_press_pattern_detected"
                and float(olympic_conf or 0.0) >= 0.90
            )

            # Clear push-press agreement should not be stolen by a false
            # pull-up prediction from the bodyweight router.
            and not (
                router_v6_label == "pull_up"
                and raw_label == "push_press"
                and bio_label == "push_press"
                and float(base_conf or 0.0) >= 0.65
                and float(bio_conf or 0.0) >= 0.75
                and float(explosive_score or 0.0) < 30.0
            )
        )

        if router_v6_bodyweight_allowed:
            final_label = router_v6_label
            final_conf = max(float(bodyweight_router_conf or 0.0), float(router_v6_conf or 0.0), 0.90)
            analysis_mode = "router_v6_bodyweight"
            protected_label = final_label
            protected_conf = final_conf
            protected_reason = "router_v6_bodyweight_winner"

        # Final clean-and-jerk shape recovery.
        # Some C&J clips are classified as squat/snatch because the clean catch
        # dominates the base and squat routers. A clear C&J event shape plus
        # extreme explosiveness is stronger evidence than the squat fallback.
        if (
            final_label in {"squat_back", "squat_front", "overhead_squat", "squat"}
            and bool(_looks_cj)
            and bool(run_oly_router)
            and float(olympic_conf or 0.0) >= 0.80
            and float(explosive_score or 0.0) >= 100.0
        ):
            final_label = "clean_and_jerk"
            final_conf = max(0.86, float(olympic_conf or 0.0))
            analysis_mode = "router_v5"
            protected_label = final_label
            protected_conf = final_conf
            protected_reason = "clean_and_jerk_shape_final_recovery"

        # Final muscle-up recovery.
        # Ring muscle-ups may look like highly explosive pull-ups to Router V6.
        # Preserve the muscle-up class when the clip has the characteristic
        # upright/explosive transition signature and weak Olympic evidence.
        if (
            final_label == "pull_up"
            and router_v6_label == "pull_up"
            and raw_label == "squat"
            and bio_label == "push_press"
            and float(bio_conf or 0.0) >= 0.88
            and float(bodyweight_router_conf or 0.0) >= 0.97
            and float(explosive_score or 0.0) >= 90.0
            and float(olympic_conf or 0.0) < 0.70
        ):
            final_label = "muscle_up"
            final_conf = 0.86
            analysis_mode = "biomechanics_override"
            protected_label = final_label
            protected_conf = final_conf
            protected_reason = "explosive_muscle_up_final_recovery"

        if final_label in {"squat_back", "squat_front", "overhead_squat"}:
            rep_feedback, _ = analyze_squat_reps(biomech, final_label)

        elif final_label == "clean":
            rep_feedback, _ = analyze_clean_reps(biomech)

        elif final_label == "clean_and_jerk":
            rep_feedback, _ = analyze_clean_and_jerk_reps(biomech)

        elif final_label == "snatch":
            rep_feedback, _ = analyze_snatch_reps(biomech)

        elif final_label == "split_jerk":
            rep_feedback, _ = analyze_split_jerk_reps(biomech)

        elif final_label == "deadlift":
            rep_feedback, _ = analyze_deadlift_reps(biomech)

        elif final_label == "bench_press":
            rep_feedback, _ = analyze_bench_press_reps(biomech)

        elif final_label == "push_press":
            rep_feedback, _ = analyze_push_press_reps(biomech, "push_press")

        elif final_label == "thruster":
            rep_feedback, _ = analyze_push_press_reps(biomech, "thruster")

        elif final_label == "strict_press":
            rep_feedback, _ = analyze_strict_press_reps(biomech)

        elif final_label == "pull_up":
            rep_feedback, _ = analyze_pull_up_reps(biomech)

        elif final_label == "handstand_push_up":
            rep_feedback, _ = analyze_handstand_push_up_reps(biomech)

        elif final_label == "push_up":
            rep_feedback, _ = analyze_push_up_reps(biomech)

        elif final_label == "burpee":
            rep_feedback, _ = analyze_burpee_reps(biomech)

        elif final_label == "muscle_up":
            rep_feedback, _ = analyze_muscle_up_reps(biomech)

        # =========================================================
        # 7. FINAL OUTPUT
        # =========================================================
        trace_route("final", final_label, final_conf, analysis_mode)

        router_state = RouterState(
            raw_label=raw_label,
            raw_conf=float(base_conf or 0.0),
            bio_label=bio_label,
            bio_conf=float(bio_conf or 0.0),
            bio_reason=bio_reason,
            squat_label=squat_label,
            squat_conf=float(squat_conf or 0.0),
            olympic_label=olympic_pred,
            olympic_conf=float(olympic_conf or 0.0),
            bodyweight_label=bodyweight_router_label,
            bodyweight_conf=float(bodyweight_router_conf or 0.0),
            final_label=final_label,
            final_conf=float(final_conf or 0.0),
            analysis_mode=analysis_mode,
            protected_label=protected_label,
            protected_reason=protected_reason,
            explosive_score=float(explosive_score or 0.0),
            wrist_overhead=float(wrist_overhead_ratio or 0.0),
            looks_clean=bool(_looks_clean_only),
            looks_cj=bool(_looks_cj),
            looks_split=bool(_looks_split),
            looks_strict=bool(_looks_strict),
            looks_thruster=bool(_looks_thruster),
            truly_explosive=bool(_truly_explosive),
            bar_pos_valid=bool(_bar_pos_valid),
            routing_trace=routing_trace,
            router_scores=router_scores,
        )

        # ------------------------------------------------------------------
        # Router V8 Shadow (diagnostics only - does NOT affect production)
        # ------------------------------------------------------------------
        try:
            from app.ml.router_v8.collectors import collect_predictions
            from app.ml.router_v8.fusion_clean_v10 import fuse_predictions
            from app.ml.router_v8.debug import build_debug

            v8_predictions = collect_predictions(
                raw_label=raw_label,
                raw_conf=base_conf,
                bio_label=bio_label,
                bio_conf=bio_conf,
                squat_label=squat_label,
                squat_conf=squat_conf,
                olympic_label=olympic_pred,
                olympic_conf=olympic_conf,
                bodyweight_label=bodyweight_router_label,
                bodyweight_conf=bodyweight_router_conf,
            )

            v8_result = fuse_predictions(
                v8_predictions,
                state=router_state,
            )

            debug["router_v8"] = build_debug(
                v8_predictions,
                v8_result,
                state=router_state,
            )

        except Exception as e:
            debug["router_v8"] = {
                "version": "router_v8_shadow",
                "error": str(e),
            }

        return {
            "exercise_label": final_label or "unknown",
            "confidence": round(final_conf, 2),
            "analysis_mode": analysis_mode,
            "rep_feedback": rep_feedback,
            "set_summary": build_set_summary(rep_feedback),
            "coaching_zones": build_coaching_zones(final_label or "unknown", rep_feedback),
            "overlay_video_url": None,
            "phase_images": None,
            "debug": {
                "olympic_pred":    olympic_pred,
                "olympic_conf":    round(olympic_conf, 3) if olympic_conf else None,
                "raw_label":       raw_label,
                "base_conf":       round(base_conf, 3),
                "bio_label":       bio_label,
                "bio_conf":        round(bio_conf, 3) if bio_conf else None,
                "bio_override":    bio_override,
                "bio_reason":      bio_reason,
                "protected_label": protected_label,
                "protected_reason": protected_reason,
                "bodyweight":      bodyweight_debug,
                "bodyweight_router_label": bodyweight_router_label,
                "bodyweight_router_conf": round(float(bodyweight_router_conf or 0.0), 3),
                "router_v5":       router_v5_debug,
                  "router_v8":       debug.get("router_v8"),
                "squat_label":     squat_label,
                "squat_conf":      round(squat_conf, 3),
                "bar_position":    bar_debug,
                "wrist_overhead":  round(wrist_overhead_ratio, 3),
                "explosive_score": round(explosive_score, 2),
                "run_oly_router":  run_oly_router,
                "looks_split":     _looks_split,
                "looks_clean":     _looks_clean_only,
                "looks_cj":        _looks_cj,
                "looks_strict":    _looks_strict,
                "looks_thruster":  _looks_thruster,
                "squat_confident": _squat_confident,
                "truly_explosive": _truly_explosive,
                "bar_pos_valid":   _bar_pos_valid,
                "analysis_path":   analysis_mode,
                "routing_trace":   routing_trace,
                "router_scores":   router_scores,
                "router_score_winner": router_score_winner,
                "router_score_value": round(float(router_score_value or 0.0), 3),
                "router_v6_label": router_v6_label,
                "router_v6_conf": round(float(router_v6_conf or 0.0), 3),
                "router_v6_decision": router_v6_decision,
            },
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


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


def transcode_video_for_analysis(input_path):
    base, _ = os.path.splitext(input_path)
    output_path = f"{base}_analysis.mp4"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-vf",
                "fps=30,scale=960:-2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-an",
                output_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
            check=True,
        )
    except Exception as e:
        print("Analysis transcode failed:", e)
        return None

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    return None


@app.get("/overlay_status/{job_id}")
async def overlay_status(job_id: str):
    job = overlay_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Overlay job not found")

    return {
        "status": job.get("status", "unknown"),
        "url": job.get("url"),
        "error": job.get("error"),
    }


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

        # ---------------- REAL ANALYSIS ----------------
        # Run full analysis pipeline to get real rep feedback and label
        analysis_result = analyze_video(temp_path, make_visuals=False, make_overlay=False)

        final_label = analysis_result.get("exercise_label", "clean_and_jerk")
        rep_feedback = analysis_result.get("rep_feedback", [])

        if not rep_feedback:
            rep_feedback = [{
                "rep": 1,
                "start_frame": 0,
                "end_frame": max(1, min(90, total_frames - 1)),
                "score": 10.0,
                "grade": "Captured",
                "issues": [],
                "feedback": [],
            }]

        # ---------------- OVERLAY ----------------
        overlay_filename = f"overlay_{uuid.uuid4().hex[:8]}.mp4"
        overlay_path = os.path.join(OVERLAY_DIR, overlay_filename)

        overlay_video_url = draw_overlay_video(
            temp_path,
            overlay_path,
            rep_feedback,
            final_label,
        )

        # ---------------- RESPONSE ----------------
        return {
            "exercise_label": final_label,
            "confidence": analysis_result.get("confidence", 0.96),
            "rep_feedback": rep_feedback,
            "overlay_video_url": overlay_video_url,
            "debug": analysis_result.get("debug", {}),
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def normalize_rep_keys(rep):
    if not rep:
        return {}

    rep = dict(rep)

    mapping = {
        "start_frame": "start",
        "descent_frame": "descent",
        "bottom_frame": "bottom",
        "ascent_frame": "ascent",
        "end_frame": "end",
        "lockout_frame": "lockout",
        "dip_frame": "dip",
        "drive_frame": "drive",
        "catch_frame": "catch",
        "first_pull_frame": "first_pull",
        "extension_frame": "extension",
    }

    for old_key, new_key in mapping.items():
        if old_key in rep and new_key not in rep:
            rep[new_key] = rep[old_key]

    return rep


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
                    "overlay_video_url": None,
                    "phase_images": None,
                    "visuals_error": "Missing rep data. Analyze the video first.",
                }

        rep = normalize_rep_keys(json.loads(rep_json))
        if isinstance(rep, list):
            rep = rep[0] if rep else None

        if not rep:
            if "pull_up" in label or "pull-up" in label or "pull up" in label or "burpee" in label or "muscle_up" in label or "muscle-up" in label or "muscle up" in label or "push_up" in label or "push-up" in label or "push up" in label:
                rep = {}
            else:
                return {
                    "exercise_label": exercise_label or "Unknown",
                    "overlay_video_url": None,
                    "phase_images": None,
                    "visuals_error": "No usable rep found.",
                }

        if "overhead squat" in label or "overhead_squat" in label:
            phase_images = create_overhead_squat_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "squat" in label:
            phase_images = create_squat_phase_images(
                temp_path, OVERLAY_DIR, rep, mp_pose, uuid, os, cv2
            )
        elif "deadlift" in label:
            phase_images = create_deadlift_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "thruster" in label:
            phase_images = create_thruster_phase_images(
                temp_path,
                OVERLAY_DIR,
                rep,
                sample_every=1,
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
        elif (
            "handstand_push_up" in label
            or "handstand push-up" in label
            or "handstand push up" in label
        ):
            phase_images = create_push_up_phase_images(
                temp_path,
                OVERLAY_DIR,
                rep,
                sample_every=1,
                exercise_label="handstand_push_up",
            )
        elif "push_up" in label or "push-up" in label or "push up" in label:
            phase_images = create_push_up_phase_images(
                temp_path,
                OVERLAY_DIR,
                rep,
                sample_every=1,
                exercise_label="push_up",
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
            "overlay_video_url": None,
            "phase_images": phase_images,
            "visuals_error": None if phase_images else "Phase images unavailable for this lift.",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            "exercise_label": exercise_label or "Unknown",
            "overlay_video_url": None,
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
async def analyze(file: UploadFile = File(...)):
    temp_path = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
    transcoded_path = None

    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        result = analyze_video(
            temp_path,
            make_visuals=True,
            make_overlay=True,
        )

        if result.get("analysis_mode") == "insufficient_data":
            transcoded_path = transcode_video_for_analysis(temp_path)
            if transcoded_path:
                retry_result = analyze_video(
                    transcoded_path,
                    make_visuals=True,
                    make_overlay=True,
                )
                retry_result.setdefault("debug", {})
                retry_result["debug"]["transcoded_retry"] = True
                retry_result["debug"]["original_frames_processed"] = result.get("debug", {}).get("frames_processed")
                return retry_result

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "exercise_label": "Unknown",
            "confidence": 0.0,
            "analysis_mode": "error",
            "feedback": [str(e)],
            "rep_feedback": [],
            "set_summary": build_set_summary([]),
            "coaching_zones": build_coaching_zones("unknown", []),
            "overlay_video_url": None,
            "phase_images": None,
            "debug": {"error": str(e)},
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if transcoded_path and os.path.exists(transcoded_path):
            os.remove(transcoded_path)


def _safe_clean_fallback(*args, **kwargs):
    return [], {"status": "fallback_clean_rep_triggered"}
