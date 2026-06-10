from sys import prefix
import tempfile
from pathlib import Path

import os
import uuid
import shutil

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

import joblib

import threading

overlay_jobs = {}

import boto3

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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

CLASS_NAMES = ["bench_press", "deadlift", "push_press", "squat"]

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
    return {"status": "ok", "model_loaded": True}


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


def classify_with_biomechanics(
    raw_label,
    confidence,
    summary,
    pose_frames,
):
    if pose_frames < 10 or not summary:
        return raw_label, confidence, False, "low_pose_data"

    # Trust strong model predictions first
    if confidence >= 0.45:
        return raw_label, confidence, False, "trusted_model_prediction"

    min_knee = summary["min_knee_angle"]
    max_knee = summary["max_knee_angle"]

    min_hip = summary["min_hip_angle"]
    max_hip = summary["max_hip_angle"]

    min_torso = summary["min_torso_angle"]
    max_torso = summary["max_torso_angle"]

    min_elbow = summary["min_elbow_angle"]
    max_elbow = summary["max_elbow_angle"]

    wrist_ratio = summary["wrist_above_shoulder_ratio"]

    knee_range = max_knee - min_knee
    hip_range = max_hip - min_hip
    torso_range = max_torso - min_torso
    elbow_range = max_elbow - min_elbow

    avg_torso = summary.get("avg_torso_angle", 0)

    # -----------------------------
    # PUSH PRESS
    # -----------------------------
    if (
        wrist_ratio > 0.65
        and elbow_range > 80
        and knee_range > 25
        and avg_torso > 55
    ):
        return "push_press", max(confidence, 0.78), True, "overhead_press_detected"

    # -----------------------------
    # DEADLIFT
    # -----------------------------
    if (
        wrist_ratio < 0.20
        and hip_range >= 50
        and torso_range >= 20
        and min_knee > 85
        and min_hip < 120
    ):
        return "deadlift", max(confidence, 0.80), True, "deadlift_pattern_detected"

    # -----------------------------
    # TRUST RAW SQUAT
    # -----------------------------
    if (
        raw_label == "squat"
        and confidence >= 0.35
        and min_knee < 100
        and knee_range >= 45
    ):
        return "squat", max(confidence, 0.75), False, "trusted_raw_squat_prediction"

    # -----------------------------
    # BENCH PRESS
    # -----------------------------
    if (
        raw_label != "squat"
        and elbow_range >= 45
        and wrist_ratio < 0.55
        and max_elbow >= 140
        and knee_range < 80
    ):
        return "bench_press", max(confidence, 0.80), True, "bench_press_pattern_detected"

    # -----------------------------
    # SQUAT PATTERN
    # -----------------------------
    if (
        knee_range >= 45
        and hip_range >= 25
        and min_knee < 105
        and wrist_ratio < 0.35
    ):
        return "squat", max(confidence, 0.75), True, "squat_pattern_detected"

    return raw_label, confidence, False, "model_prediction"


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

            if max_torso > 75:
                breakdown["back"] = "poor"
                issues.append("Back may be rounding during the pull.")
                feedback.append("Brace your core and keep a neutral spine.")
            elif max_torso > 60:
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
        "knees": {"good": 0.0, "borderline": 0.6, "poor": 1.2},
        "heels": {"good": 0.0, "borderline": 0.4, "poor": 0.9},
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
            clean_valgus = np.clip(rep_valgus, 0.75, 1.5)
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

            if valgus_score < 0.80:
                knees_grade = "poor"
                issues.append("Knees cave inward noticeably.")
                feedback.append("Drive knees out over your toes.")
            elif valgus_score < 0.92:
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

    return {
        "setup": start_idx + int(span * 0.10),
        "dip": start_idx + int(span * 0.25),
        "drive": start_idx + int(span * 0.45),
        "catch": start_idx + int(span * 0.62),
        "lockout": start_idx + int(span * 0.78),
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

            if end - start < 3:
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

            dip_idx = int(np.argmin(rep_knee))
            first_overhead = np.where(rep_wrist_y < rep_shoulder_y)[0]

            if len(first_overhead) > 0:
                overhead_idx = int(first_overhead[0])
                drive_timing = overhead_idx - dip_idx
            else:
                drive_timing = 999

            if wrist_drift > 0.05:
                drift_severity = "severe"
            elif wrist_drift > 0.03:
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

                if drive_timing < -3:
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

                if torso_score > 65:
                    issues.append("Torso is leaning too far forward during the thruster.")
                    feedback.append("Stay tall through the squat and drive straight overhead.")

                if elbow_lockout < 135:
                    issues.append("Finish stronger overhead.")
                    feedback.append("Fully lock out the bar overhead at the top.")

            if wrist_above < 0.35:
                issues.append("Incomplete overhead lockout.")
                feedback.append("Fully extend arms overhead.")

            if wrist_drift > 0.03:
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

            breakdown = {
                "lockout": "good" if wrist_above >= 0.35 else "incomplete",
                "bar_path": "drifting" if wrist_drift > 0.03 else "good",
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
                breakdown["active_finish"] = "good" if elbow_lockout >= 165 else "soft"
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
                breakdown["torso_stack"] = "good" if torso_score <= 45 else "leaning_forward"
                breakdown["active_finish"] = "good" if elbow_lockout >= 150 else "soft"
                breakdown["knee_range"] = round(knee_range, 1)

            score = apply_coach_reward(score, issues, breakdown)

            if exercise_label == "thruster":
                score += 1.2

                if breakdown.get("bar_severity") == "moderate":
                    score += 0.8
                elif breakdown.get("bar_severity") == "severe":
                    score += 0.5

                if breakdown.get("lockout") == "good":
                    score += 0.7

                if breakdown.get("squat_depth") == "good":
                    score += 0.7

                score = min(10.0, round(score, 1))

                # Don't allow perfect scores when issues exist
                if issues:
                    score = min(score, 9.2)

            if not issues:
                score = max(score, 9.0)
                feedback = [good_rep_message]

            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(frame_numbers[start]),
                "end_frame": int(frame_numbers[end]),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback or [good_rep_message],
            })

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

    return reps, build_set_summary(reps)


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


def draw_overlay_video(
    input_path,
    output_path,
    rep_feedback,
    exercise_label,
    sample_every=1,
):
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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_output_path = output_path.replace(".mp4", "_raw.mp4")

    writer = cv2.VideoWriter(
        temp_output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("Overlay error: could not open video writer")
        cap.release()
        return None

    best_rep = max(
        rep_feedback,
        key=lambda rep: (
            float(rep.get("score", 0) or 0),
            int(rep.get("rep", 0) or 0),
        ),
    )

    exercise = str(exercise_label or "").lower().replace(" ", "_")
    score = best_rep.get("score", None)
    feedback = best_rep.get("feedback") or best_rep.get("issues") or []
    main_note = feedback[0] if feedback else "Keep the full body visible and move with control."

    frame_idx = 0
    frames_written = 0
    landmark_hits = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    ) as pose:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # Always draw a visible overlay header so we know this file is not raw video.
            cv2.rectangle(frame, (0, 0), (width, 92), (2, 6, 23), -1)

            cv2.putText(
                frame,
                "FORMCHECK AI OVERLAY",
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (134, 239, 172),
                2,
                cv2.LINE_AA,
            )

            score_text = f"{formatLabel(exercise_label)}"
            if score is not None:
                score_text += f" | Best Rep Score: {float(score):.1f}/10"

            cv2.putText(
                frame,
                score_text,
                (24, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (248, 250, 252),
                2,
                cv2.LINE_AA,
            )

            if results.pose_landmarks:
                landmark_hits += 1

                frame = draw_user_skeleton(
                    frame,
                    results.pose_landmarks,
                )

                if exercise in ["deadlift"]:
                    frame = draw_ideal_deadlift(
                        frame,
                        results.pose_landmarks,
                        width,
                        height,
                    )

                elif exercise in ["squat", "squat_back", "back_squat"]:
                    frame = draw_ideal_squat_overlay(
                        frame,
                        results.pose_landmarks,
                        width,
                        height,
                    )

                elif exercise in ["push_press", "strict_press"]:
                    frame = draw_ideal_push_press_overlay(
                        frame,
                        results.pose_landmarks,
                        width,
                        height,
                    )

                elif exercise in ["bench_press", "bench"]:
                    frame = draw_ideal_bench_press_overlay(
                        frame,
                        results.pose_landmarks,
                        width,
                        height,
                    )

                else:
                    cv2.putText(
                        frame,
                        "Pose detected. No ideal overlay for this exercise yet.",
                        (24, height - 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (251, 191, 36),
                        2,
                        cv2.LINE_AA,
                    )

            else:
                cv2.putText(
                    frame,
                    "Pose not detected on this frame",
                    (24, height - 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (239, 68, 68),
                    2,
                    cv2.LINE_AA,
                )

            cv2.rectangle(frame, (0, height - 58), (width, height), (15, 23, 42), -1)
            cv2.putText(
                frame,
                str(main_note)[:95],
                (24, height - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (226, 232, 240),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frames_written += 1
            frame_idx += 1

    cap.release()
    writer.release()

    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", temp_output_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ],
        check=True,
    )

    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    print("OVERLAY FRAMES WRITTEN:", frames_written)
    print("OVERLAY LANDMARK HITS:", landmark_hits)

    if frames_written == 0:
        print("Overlay error: no frames written")
        return None

    if landmark_hits == 0:
        print("Overlay warning: video was written but no pose landmarks were detected")

    return output_path


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

    # Rebuild phases from start/bottom/end so they are spaced correctly.
    setup_frame = start
    descent_frame = start + int((bottom - start) * 0.60)
    bottom_frame = bottom
    ascent_frame = bottom + int((end - bottom) * 0.45)
    lockout_frame = end

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


def create_push_press_phase_images(input_path, output_dir, rep, sample_every=1):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Push press phase error")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = int(rep.get("start_frame", 0))
    end = int(rep.get("end_frame", total_frames - 1))

    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames - 1))

    phase_frames = find_push_press_phase_window(start, end)

    saved = {}

    for phase, frame_idx in phase_frames.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if not ret:
            continue

        filename = f"push_press_{phase}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)
        saved[phase] = f"/outputs/{filename}"

    # One image showing all 5 phases side by side
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

    start = int(rep.get("start_frame", 0)) * sample_every
    end = int(rep.get("end_frame", total_frames - 1)) * sample_every

    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames - 1))

    duration = max(1, end - start)

    normalized_label = exercise_label.lower().replace(" ", "_")
    prefix = normalized_label

    # -----------------------------------
    # JERK / SPLIT JERK / THRUSTER
    # -----------------------------------
    if normalized_label in ["jerk", "split_jerk", "thruster"]:
        phase_frames = {
            "setup": start,
            "dip": start + int(duration * 0.20),
            "drive": start + int(duration * 0.38),
            "catch": start + int(duration * 0.62),
            "recovery": start + int(duration * 0.82),
            "finish": max(start, min(end - 1, total_frames - 1)),
        }

    # -----------------------------------
    # CLEAN / SNATCH / CLEAN & JERK
    # -----------------------------------
    else:
        phase_frames = {
            "setup": start,
            "first_pull": start + int(duration * 0.22),
            "extension": start + int(duration * 0.48),
            "catch": start + int(duration * 0.72),
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

    # Force finish fallback
    if "finish" not in saved and "recovery" in saved:
        saved["finish"] = saved["recovery"]

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
        start = int(rep.get("start_frame", 0)) * sample_every
        end = int(rep.get("end_frame", total_frames - 1)) * sample_every
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
        start = int(rep.get("start_frame", 0)) * sample_every
        end = int(rep.get("end_frame", total_frames - 1)) * sample_every
    else:
        start = 0
        end = total_frames - 1

    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames - 1))

    duration = max(1, end - start)

    phase_frames = {
        "hang": start,
        "pull": start + int(duration * 0.22),
        "transition": start + int(duration * 0.45),
        "dip": start + int(duration * 0.65),
        "lockout": start + int(duration * 0.82),
        "finish": max(start, min(end - 1, total_frames - 1)),
    }

    saved = {}
    debug_images = []

    prefix = exercise_label.lower().replace(" ", "_")

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

    if "finish" not in saved and "lockout" in saved:
        saved["finish"] = saved["lockout"]

    cap.release()

    return saved


def build_coaching_zones(exercise_label, rep_feedback):
    if not rep_feedback:
        return {}

    FEEDBACK_BY_KEY = {
        # -----------------------------
        # Shared / Squat
        # -----------------------------
        "neck": "Keep your neck neutral and eyes forward.",
        "torso": "Keep your chest up and torso controlled.",
        "squat_knees": "Drive knees out over your toes.",
        "depth": "Sink a little deeper while keeping your chest up.",
        "heels": "Keep your heels planted through the rep.",
        "front_rack": "Drive elbows higher to keep the bar secure.",
        "bar_position": "Keep the bar stacked securely over your midfoot.",

        # -----------------------------
        # Deadlift
        # -----------------------------
        "back": "Keep your back flat and brace your core.",
        "hinge": "Push your hips back and hinge before pulling.",
        "deadlift_knees": "Keep shins more vertical and hinge from the hips.",
        "deadlift_bar_path": "Keep the bar close to your body.",
        "deadlift_lockout": "Finish tall with hips fully extended.",

        # -----------------------------
        # Bench Press
        # -----------------------------
        "elbows": "Keep elbows controlled and stacked under the wrists.",
        "arch": "Keep your upper back tight and arch controlled.",
        "bench_lockout": "Fully extend your arms at the top.",
        "bench_legs_unknown": "Foot position could not be evaluated because feet were not visible.",
        "legs": "Keep your feet planted and use leg drive.",

        # -----------------------------
        # Push Press / Strict Press
        # -----------------------------
        "dip": "Use a controlled vertical dip before driving up.",
        "dip_verticality": "Keep the dip vertical with chest tall.",
        "press_timing": "Drive with your legs first, then press overhead.",
        "press_bar_path": "Keep the bar path vertical and press straight overhead.",
        "press_lockout": "Fully extend arms overhead.",
        "active_finish": "Punch to a strong, stacked lockout.",
    }

    def zone_result(label, key, good_values, feedback_key=None):
        affected = []

        for rep in rep_feedback:
            breakdown = rep.get("breakdown", {})
            value = breakdown.get(key)

            if value is None:
                continue

            if value not in good_values:
                affected.append(rep.get("rep"))

        status = "good" if not affected else "needs_work"

        if status == "good":
            message = f"{label} looks solid across the set."
        else:
            lookup_key = feedback_key or key
            message = FEEDBACK_BY_KEY.get(
                lookup_key,
                f"{label} needs attention.",
            )

        return {
            "label": label,
            "status": status,
            "message": message,
            "affected_reps": affected,
        }

    label = exercise_label.lower().replace(" ", "_")

    # -----------------------------
    # SQUAT FAMILY
    # -----------------------------
    if label in ["squat", "squat_back", "squat_front", "overhead_squat"]:
        zones = {
            "neck": zone_result("Neck", "neck", {"good"}),
            "torso": zone_result("Torso", "torso", {"good"}),
            "knees": zone_result(
                "Knees",
                "knees",
                {"good"},
                "squat_knees",
            ),
            "depth": zone_result("Depth", "depth", {"good"}),
            "heels": zone_result("Heels", "heels", {"good"}),
        }

        if label == "squat_front":
            zones["front_rack"] = zone_result(
                "Front Rack",
                "front_rack",
                {"good"},
            )
            zones["bar_position"] = zone_result(
                "Bar Position",
                "bar_position",
                {"good"},
            )

        return zones

    # -----------------------------
    # DEADLIFT
    # -----------------------------
    elif label == "deadlift":
        return {
            "neck": zone_result("Neck", "neck", {"good"}),
            "torso": zone_result("Torso", "back", {"good"}),
            "hips": zone_result("Hip Hinge", "hinge", {"good"}),
            "knees": zone_result(
                "Knees",
                "knees",
                {"good"},
                "deadlift_knees",
            ),
            "bar_path": zone_result(
                "Bar Path",
                "bar_path",
                {"good"},
                "deadlift_bar_path",
            ),
            "lockout": zone_result(
                "Lockout",
                "lockout",
                {"good"},
                "deadlift_lockout",
            ),
        }

    # -----------------------------
    # BENCH PRESS
    # -----------------------------
    elif label == "bench_press":
        return {
            "elbows": zone_result("Elbows", "elbows", {"good"}),
            "depth": zone_result("Depth", "depth", {"good"}),
            "lockout": zone_result(
                "Lockout",
                "lockout",
                {"good"},
                "bench_lockout",
            ),
            "arch": zone_result("Arch", "arch", {"controlled", "good"}),
            "legs": zone_result(
                "Leg Drive",
                "legs",
                {"good"},
                "bench_legs_unknown",
            ),
        }

    # -----------------------------
    # PUSH PRESS / STRICT PRESS
    # -----------------------------
    elif label in ["push_press", "strict_press", "thruster"]:
        return {
            "dip": zone_result("Dip", "dip", {"good"}),
            "dip_path": zone_result(
                "Dip Path",
                "dip_verticality",
                {"good"},
                "dip_verticality",
            ),
            "timing": zone_result(
                "Timing",
                "timing",
                {"good"},
                "press_timing",
            ),
            "bar_path": zone_result(
                "Bar Path",
                "bar_path",
                {"good"},
                "press_bar_path",
            ),
            "lockout": zone_result(
                "Lockout",
                "lockout",
                {"good"},
                "press_lockout",
            ),
            "finish": zone_result(
                "Finish",
                "active_finish",
                {"good"},
                "active_finish",
            ),
        }

    return {}


def analyze_video(video_path, make_visuals=True, make_overlay=True):
    try:
        olympic_labels = [
            "olympic_lift",
            "clean_and_jerk",
            "snatch",
            "clean",
            "jerk",
            "split_jerk",
            "thruster",
        ]

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {
                "exercise_label": "Unknown",
                "confidence": 0.0,
                "analysis_mode": "video_error",
                "feedback": ["Could not open uploaded video."],
                "rep_feedback": [],
                "set_summary": build_set_summary([]),
                "coaching_zones": build_coaching_zones("unknown", []),
                "overlay_video_url": None,
                "phase_images": None,
                "debug": {},
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every = max(3, min(5, total_frames // 160))

        sequence = []
        biomechanics = []
        frame_idx = 0
        pose_frames = 0

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

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if not results.pose_landmarks:
                    continue

                feats, bio = extract_features_and_biomechanics(results)

                if feats is None or bio is None:
                    continue

                pose_frames += 1
                sequence.append(feats)

                bio["frame_number"] = frame_idx
                biomechanics.append(bio)

        cap.release()

        if len(sequence) < 10:
            return {
                "exercise_label": "Unknown",
                "confidence": 0.0,
                "analysis_mode": "insufficient_data",
                "feedback": ["Not enough pose data detected."],
                "rep_feedback": [],
                "set_summary": build_set_summary([]),
                "coaching_zones": build_coaching_zones("unknown", []),
                "overlay_video_url": None,
                "phase_images": None,
                "debug": {
                    "frames_seen": total_frames,
                    "frames_processed": len(sequence),
                    "pose_frames": pose_frames,
                    "sample_every": sample_every,
                },
            }

        seq_base = pad_or_trim(np.array(sequence), target_len=30)
        seq = add_velocity(seq_base)

        probs = MODEL.predict_proba(seq)
        raw_idx = int(np.argmax(probs))

        if raw_idx >= len(CLASS_NAMES):
            raw_label = "squat"
            raw_confidence = float(np.max(probs))
        else:
            raw_label = CLASS_NAMES[raw_idx]
            raw_confidence = float(probs[raw_idx])

        summary = summarize_biomechanics(biomechanics)

        oly_sequence = [
            b["full_features"]
            for b in biomechanics
            if "full_features" in b
        ]

        oly_label, oly_confidence = predict_olympic_lift_from_sequence(
            oly_sequence
        )

        if (
            oly_label in ["clean_and_jerk", "snatch", "clean", "jerk", "split_jerk"]
            and oly_confidence >= 0.85
            and raw_label in [
                "olympic_lift",
                "deadlift",
                "squat",
                "squat_back",
                "squat_front",
                "push_press",
                "strict_press",
                "pull_up",
                "bar_muscle_up"
            ]
        ):
            raw_label = oly_label
            raw_confidence = oly_confidence

        # --------------------------------------------------
        # BAR MUSCLE-UP OVERRIDE
        # --------------------------------------------------
        if (
            raw_label in ["squat", "squat_back", "squat_front", "pull_up", "snatch"]
            and summary.get("wrist_above_shoulder_ratio", 0) > 0.40
            and summary.get("avg_torso_angle", 90) < 55
            and summary.get("avg_knee_angle", 0) > 155
            and summary.get("min_elbow_angle", 180) < 35
            and summary.get("max_elbow_angle", 0) > 160
        ):
            raw_label = "bar_muscle_up"
            raw_confidence = 0.84

        # --------------------------------------------------
        # RING MUSCLE-UP OVERRIDE
        # --------------------------------------------------
        if (
            raw_label in ["squat", "squat_back", "squat_front", "snatch", "pull_up", "bar_muscle_up"]
            and summary.get("wrist_above_shoulder_ratio", 0) > 0.60
            and summary.get("avg_torso_angle", 90) < 45
            and summary.get("max_torso_angle", 0) > 60
            and summary.get("min_elbow_angle", 180) < 75
            and summary.get("max_elbow_angle", 0) > 165
        ):
            raw_label = "ring_muscle_up"
            raw_confidence = 0.84

        # --------------------------------------------------
        # PULL-UP OVERRIDE
        # --------------------------------------------------
        if (
            raw_label in ["squat", "squat_back", "squat_front", "snatch"]
            and summary.get("wrist_above_shoulder_ratio", 0) > 0.70
            and summary.get("avg_torso_angle", 90) < 35
            and summary.get("avg_knee_angle", 0) > 145
            and summary.get("max_elbow_angle", 0) > 160
            and summary.get("min_elbow_angle", 180) < 95
        ):
            raw_label = "pull_up"
            raw_confidence = 0.84

        # SNATCH OVERRIDE
        if (
            raw_label in ["squat", "squat_back", "squat_front", "overhead_squat"]
            and summary.get("wrist_above_shoulder_ratio", 0) >= 0.22
            and summary.get("min_knee_angle", 180) < 90
            and summary.get("min_hip_angle", 180) < 90
            and summary.get("max_elbow_angle", 0) > 150
        ):
            raw_label = "snatch"
            raw_confidence = 0.82

        # CLEAN OVERRIDE
        if (
            raw_label == "deadlift"
            and summary.get("wrist_above_shoulder_ratio", 0) >= 0.10
            and summary.get("max_elbow_angle", 0) > 150
            and summary.get("min_hip_angle", 180) < 80
            and summary.get("min_knee_angle", 180) < 100
        ):
            raw_label = "clean"
            raw_confidence = 0.80

        # --------------------------------------------------
        # THRUSTER OVERRIDE
        if (
            raw_label in ["squat", "squat_back", "squat_front", "bench_press", "push_press"]
            and summary.get("wrist_above_shoulder_ratio", 0) > 0.15
            and summary.get("min_knee_angle", 180) < 100
            and summary.get("min_hip_angle", 180) < 100
            and summary.get("max_elbow_angle", 0) > 150
        ):
            raw_label = "thruster"
            raw_confidence = 0.84

        # OVERHEAD SQUAT OVERRIDE
        if (
            raw_label == "push_press"
            and summary.get("wrist_above_shoulder_ratio", 0) > 0.80
            and summary.get("min_knee_angle", 180) < 120
        ):
            raw_label = "overhead_squat"
            raw_confidence = 0.90

        overhead_router_label = None
        overhead_router_confidence = None

        if raw_label == "push_press":
            overhead_probs = OVERHEAD_ROUTER_MODEL.predict(
                np.expand_dims(seq_base, axis=0),
                verbose=0,
            )[0]

            overhead_idx = int(np.argmax(overhead_probs))
            overhead_router_label = OVERHEAD_ROUTER_LABELS[overhead_idx]
            overhead_router_confidence = float(overhead_probs[overhead_idx])

            if (
                overhead_router_label == "strict_press"
                and overhead_router_confidence > 0.85
            ):
                raw_label = "strict_press"
                raw_confidence = overhead_router_confidence

        squat_router_label = None
        squat_router_confidence = None

        # SPLIT JERK OVERRIDE
        if (
            raw_label in ["push_press", "strict_press"]
            and summary.get("wrist_above_shoulder_ratio", 0) > 0.55
            and summary.get("min_knee_angle", 180) < 115
            and summary.get("max_elbow_angle", 0) > 150
        ):
            raw_label = "split_jerk"
            raw_confidence = 0.82

        # SQUAT FAMILY ROUTER
        if "squat" in raw_label:
            squat_probs = SQUAT_ROUTER_MODEL.predict(
                np.expand_dims(seq_base, axis=0),
                verbose=0,
            )[0]

            squat_idx = int(np.argmax(squat_probs))
            squat_router_label = SQUAT_ROUTER_LABELS[squat_idx]
            squat_router_confidence = float(squat_probs[squat_idx])

            if (
                squat_router_label == "overhead_squat"
                and summary.get("wrist_above_shoulder_ratio", 0) < 0.65
            ):
                squat_router_label = "squat_front"
                squat_router_confidence = 0.81

            if (
                squat_router_label == "squat_front"
                and summary.get("wrist_above_shoulder_ratio", 0) < 0.30
            ):
                squat_router_label = "squat_back"
                squat_router_confidence = 0.80

            if squat_router_label is not None:
                raw_label = squat_router_label
                raw_confidence = squat_router_confidence

        label, confidence, override_used, reason = classify_with_biomechanics(
            raw_label,
            raw_confidence,
            summary,
            pose_frames,
        )

        if (
            raw_label in ["squat_back", "squat_front", "overhead_squat"]
            and label == "squat"
        ):
            label = raw_label
            confidence = raw_confidence

        if raw_label in olympic_labels:
            label = raw_label
            confidence = raw_confidence
            override_used = False
            reason = "olympic_router_prediction"

        analysis_mode = "classification_only"
        rep_feedback = []

        if label in ["squat", "squat_back", "squat_front", "overhead_squat"]:
            rep_feedback, _ = analyze_squat_reps(biomechanics, label)
            analysis_mode = "detailed_rep_analysis"

        elif label == "deadlift":
            rep_feedback, _ = analyze_deadlift_reps(biomechanics)
            analysis_mode = "detailed_rep_analysis"

        elif label in ["push_press", "strict_press", "thruster"]:
            rep_feedback, _ = analyze_push_press_reps(biomechanics, label)
            analysis_mode = "detailed_rep_analysis"

        elif label == "bench_press":
            rep_feedback, _ = analyze_bench_press_reps(biomechanics)
            analysis_mode = "detailed_rep_analysis"

        elif label in olympic_labels:
            analysis_mode = "classification_only"
            rep_feedback = [
                {
                    "rep": 1,
                    "start_frame": 0,
                    "end_frame": total_frames - 1,
                    "score": 10.0,
                    "grade": "Captured",
                    "issues": [],
                    "feedback": [],
                }
            ]

        set_summary = build_set_summary(rep_feedback)

        overlay_video_url = None
        phase_images = None

        if make_visuals and rep_feedback:
            phase_rep = choose_phase_rep(rep_feedback)

            if make_overlay:
                phase_rep = choose_phase_rep(rep_feedback)

                overlay_filename = f"overlay_{uuid.uuid4().hex[:8]}.mp4"
                overlay_path = os.path.join(OVERLAY_DIR, overlay_filename)

                overlay_result = draw_overlay_video(
                    video_path,
                    overlay_path,
                    rep_feedback,
                    label,
                    sample_every=sample_every,
            )

            if overlay_result:
                overlay_video_url = f"/outputs/{overlay_filename}"

            if phase_rep or label in olympic_labels:
                if label in ["squat", "squat_back", "squat_front", "overhead_squat"]:
                    phase_images = create_squat_phase_images(
                        video_path,
                        OVERLAY_DIR,
                        phase_rep,
                        sample_every=sample_every,
                    )

                elif label == "deadlift":
                    phase_images = create_deadlift_phase_images(
                        video_path,
                        OVERLAY_DIR,
                        phase_rep,
                        sample_every=sample_every,
                    )

                elif label in ["push_press", "strict_press", "thruster"]:
                    phase_images = create_push_press_phase_images(
                        video_path,
                        OVERLAY_DIR,
                        phase_rep,
                        sample_every=sample_every,
                    )

                elif label == "bench_press":
                    phase_images = create_bench_press_phase_images(
                        video_path,
                        OVERLAY_DIR,
                        phase_rep,
                        sample_every=sample_every,
                    )

                elif label in olympic_labels:
                    phase_images = create_olympic_lift_phase_images(
                        video_path,
                        OVERLAY_DIR,
                        phase_rep
                        or {
                            "rep": 1,
                            "start_frame": 0,
                            "end_frame": total_frames - 1,
                        },
                        sample_every=sample_every,
                        exercise_label=label,
                    )

        display_name = {
            "squat_front": "Front Squat",
            "squat_back": "Back Squat",
            "overhead_squat": "Overhead Squat",
            "push_press": "Push Press",
            "strict_press": "Strict Press",
            "bench_press": "Bench Press",
            "deadlift": "Deadlift",
            "squat": "Squat",
            "olympic_lift": "Olympic Lift",
            "clean_and_jerk": "Clean and Jerk",
            "snatch": "Snatch",
            "clean": "Clean",
            "jerk": "Jerk",
            "split_jerk": "Split Jerk",
            "thruster": "Thruster",
            "pull_up": "Pull-up",
            "bar_muscle_up": "Bar Muscle-up",
            "ring_muscle_up": "Ring Muscle-up",
        }.get(label, label.replace("_", " ").title())

        return {
            "exercise_label": display_name,
            "confidence": round(confidence, 2),
            "analysis_mode": analysis_mode,
            "feedback": [
                f"Predicted exercise: {display_name}.",
                f"Model confidence: {round(confidence * 100, 1)}%.",
                (
                    f"Biomechanics override applied: {reason}."
                    if override_used
                    else "Model prediction used."
                ),
            ],
            "rep_feedback": rep_feedback,
            "set_summary": set_summary,
            "coaching_zones": build_coaching_zones(label, rep_feedback),
            "overlay_video_url": overlay_video_url,
            "phase_images": phase_images,
            "debug": {
                "original_prediction": raw_label,
                "original_confidence": round(raw_confidence, 4),
                "final_prediction": label,
                "override_used": override_used,
                "classification_reason": reason,
                "raw_predictions": dict(zip(CLASS_NAMES, probs.tolist())),
                "oly_router_prediction": oly_label,
                "oly_router_confidence": round(oly_confidence, 4),
                "overhead_router_prediction": overhead_router_label,
                "overhead_router_confidence": (
                    round(overhead_router_confidence, 4)
                    if overhead_router_confidence is not None
                    else None
                ),
                "squat_router_prediction": squat_router_label,
                "squat_router_confidence": (
                    round(squat_router_confidence, 4)
                    if squat_router_confidence is not None
                    else None
                ),
                "biomechanics": summary,
                "frames_seen": total_frames,
                "frames_processed": len(sequence),
                "pose_frames": pose_frames,
                "sample_every": sample_every,
                "runtime_sequence_shape": list(seq.shape),
                "classifier_input_shape": [30, 136],
                "router_input_shape": [30, 68],
                "oly_router_input_shape": [528],
            },
        }

    except Exception as e:
        import traceback

        traceback.print_exc()

        return {
            "error": True,
            "message": str(e),
        }


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

        if not rep_json:
            return {
                "exercise_label": exercise_label or "Unknown",
                "overlay_video_url": None,
                "phase_images": None,
                "visuals_error": "Missing rep data. Analyze the video first.",
            }

        rep = json.loads(rep_json)
        if isinstance(rep, list):
            rep = rep[0] if rep else None

        if not rep:
            return {
                "exercise_label": exercise_label or "Unknown",
                "overlay_video_url": None,
                "phase_images": None,
                "visuals_error": "No usable rep found.",
            }

        label = str(exercise_label or "").lower()

        if "squat" in label:
            phase_images = create_squat_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "deadlift" in label:
            phase_images = create_deadlift_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "push press" in label:
            phase_images = create_push_press_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
            )
        elif "bench" in label:
            phase_images = create_bench_press_phase_images(
                temp_path, OVERLAY_DIR, rep, sample_every=1
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

            
def overlay_worker(job_id, temp_path, rep_feedback, exercise_label):
    try:
        overlay_jobs[job_id] = {"status": "processing"}

        overlay_filename = f"overlay_{uuid.uuid4().hex[:8]}.mp4"
        overlay_path = os.path.join(OVERLAY_DIR, overlay_filename)

        made_overlay = draw_overlay_video(
            input_path=temp_path,
            output_path=overlay_path,
            rep_feedback=rep_feedback,
            exercise_label=exercise_label or "unknown",
            sample_every=3,
        )

        if made_overlay:
            s3_key = f"overlays/{overlay_filename}"

            s3_client.upload_file(
                overlay_path,
                S3_BUCKET,
                s3_key,
                ExtraArgs={
                    "ContentType": "video/mp4",
                },
            )

            overlay_url = (
                f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
            )

            overlay_jobs[job_id] = {
                "status": "ready",
                "overlay_video_url": overlay_url,
            }
        else:
            overlay_jobs[job_id] = {
                "status": "error",
                "message": "Could not generate overlay video.",
            }

    except Exception as e:
        overlay_jobs[job_id] = {
            "status": "error",
            "message": str(e),
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if "overlay_path" in locals() and os.path.exists(overlay_path):
            os.remove(overlay_path)


@app.post("/start_overlay")
async def start_overlay(
    file: UploadFile = File(...),
    rep_json: str = Form(None),
    exercise_label: str = Form(None),
):
    import json

    suffix = os.path.splitext(file.filename or "")[1] or ".mov"
    temp_filename = f"overlay_input_{uuid.uuid4().hex[:8]}{suffix}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    rep_feedback = []

    if rep_json:
        rep = json.loads(rep_json)
        rep_feedback = [rep] if isinstance(rep, dict) else rep

    if not rep_feedback:
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        rep_feedback = [{
            "rep": 1,
            "start_frame": 0,
            "end_frame": min(90, max(1, total_frames - 1)),
            "score": 10.0,
            "grade": "Captured",
            "issues": [],
            "feedback": [],
        }]

    job_id = uuid.uuid4().hex[:12]
    overlay_jobs[job_id] = {"status": "queued"}

    threading.Thread(
        target=overlay_worker,
        args=(job_id, temp_path, rep_feedback, exercise_label),
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "status": "processing",
    }


@app.get("/overlay_status/{job_id}")
async def overlay_status(job_id: str):
    return overlay_jobs.get(job_id, {
        "status": "not_found",
        "message": "Overlay job not found.",
    })


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
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        rep_feedback = []

        if rep_json:
            try:
                rep = json.loads(rep_json)
                rep_feedback = [rep] if isinstance(rep, dict) else rep
            except Exception as e:
                print("OVERLAY REP JSON PARSE ERROR:", e)

        if not rep_feedback:
            cap = cv2.VideoCapture(temp_path)

            if not cap.isOpened():
                return {
                    "overlay_video_url": None,
                    "overlay_error": "Could not open uploaded video.",
                }

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            rep_feedback = [{
                "rep": 1,
                "start_frame": 0,
                "end_frame": min(90, max(1, total_frames - 1)),
                "score": 10.0,
                "grade": "Captured",
                "issues": [],
                "feedback": [],
            }]

        overlay_filename = f"overlay_{uuid.uuid4().hex[:8]}.mp4"
        overlay_path = os.path.join(OVERLAY_DIR, overlay_filename)

        made_overlay = draw_overlay_video(
            input_path=temp_path,
            output_path=overlay_path,
            rep_feedback=rep_feedback,
            exercise_label=exercise_label or "unknown",
            sample_every=3,
        )

        runtime = round(time.time() - started_at, 2)

        if not made_overlay:
            return {
                "overlay_video_url": None,
                "overlay_error": "Could not generate overlay video.",
                "runtime_seconds": runtime,
            }

        return {
            "overlay_video_url": f"/outputs/{overlay_filename}",
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
    suffix = os.path.splitext(file.filename or "")[1] or ".mov"
    temp_filename = f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = analyze_video(temp_path, make_visuals=False)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            "error": True,
            "message": str(e),
            "exercise_label": "Unknown",
            "confidence": 0.0,
            "analysis_mode": "error",
            "feedback": ["Analysis failed."],
            "rep_feedback": [],
            "set_summary": build_set_summary([]),
            "coaching_zones": build_coaching_zones("unknown", []),
            "overlay_video_url": None,
            "phase_images": None,
            "debug": {},
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

