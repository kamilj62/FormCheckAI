import tempfile
from pathlib import Path

import os
import uuid

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

OVERLAY_DIR = "outputs"
os.makedirs(OVERLAY_DIR, exist_ok=True)

app.mount(
    "/outputs",
    StaticFiles(directory=OVERLAY_DIR),
    name="outputs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

CLASS_NAMES = ["bench_press", "deadlift", "push_press", "squat"]
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
    }


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


def point(landmarks, landmark):
    lm = landmarks[landmark.value]
    return np.array([lm.x, lm.y], dtype=np.float32)


def extract_features_and_biomechanics(results):
    if not results.pose_landmarks:
        return None, None

    landmarks = results.pose_landmarks.landmark

    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y])

    pts = np.array([
        [lm.x, lm.y, lm.z]
        for lm in results.pose_landmarks.landmark
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
    shoulder_y = float(shoulder_mid[1])

    # Extra movement values used for deadlift vs push press detection
    hip_x = float(hip_mid[0])
    shoulder_x = float(shoulder_mid[0])
    knee_x = float(knee_mid[0])

    shoulder_hip_distance = float(np.linalg.norm(shoulder_mid - hip_mid))
    hip_knee_distance = float(np.linalg.norm(hip_mid - knee_mid))
    wrist_shoulder_distance = float(np.linalg.norm(wrist_mid - shoulder_mid))

    knee_width = abs(float(left_knee[0]) - float(right_knee[0]))
    ankle_width = abs(float(left_ankle[0]) - float(right_ankle[0]))
    valgus_ratio = knee_width / (ankle_width + 1e-6)

    wrist_x = (left_wrist[0] + right_wrist[0]) / 2
    ankle_x = (left_ankle[0] + right_ankle[0]) / 2
    bar_distance = abs(wrist_x - ankle_x)

    wrist_above_shoulder = float(wrist_y < shoulder_y)

    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2

    nose_x = nose[0]
    nose_y = nose[1]

    head_drop = nose_y - shoulder_mid_y
    head_forward = abs(nose_x - shoulder_mid_x)

    features.extend([
        knee_angle / 180.0,
        hip_angle / 180.0,
        torso_angle / 180.0,
        hip_y,
        knee_y,
    ])

    biomechanics = {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "torso_angle": torso_angle,
        "elbow_angle": elbow_angle,

        "hip_y": hip_y,
        "knee_y": knee_y,
        "wrist_x": wrist_x,
        "wrist_y": wrist_y,
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
    if pose_frames < 10:
        return raw_label, confidence, False, "low_pose_data"

    min_knee = summary.get("min_knee_angle", 0)
    max_knee = summary.get("max_knee_angle", 0)
    min_hip = summary.get("min_hip_angle", 0)
    max_hip = summary.get("max_hip_angle", 0)
    min_torso = summary.get("min_torso_angle", 0)
    max_torso = summary.get("max_torso_angle", 0)
    max_elbow = summary.get("max_elbow_angle", 0)
    min_elbow = summary.get("min_elbow_angle", 0)
    wrist_ratio = summary.get("wrist_above_shoulder_ratio", 0)

    hip_range = max_hip - min_hip
    knee_range = max_knee - min_knee
    torso_range = max_torso - min_torso
    elbow_range = max_elbow - min_elbow

    print("RAW:", raw_label, "conf:", confidence)
    print("hip_range:", hip_range)
    print("knee_range:", knee_range)
    print("torso_range:", torso_range)
    print("elbow_range:", elbow_range)
    print("wrist_ratio:", wrist_ratio)
    print("SUMMARY:", summary)

    # -----------------------------
    # TRUST MODEL: BENCH PRESS
    # -----------------------------
    if raw_label == "bench_press":
        return "bench_press", max(confidence, 0.80), True, "trusted_model_bench_press"

    # -----------------------------
    # BENCH PRESS FALLBACK
    # -----------------------------
    if (
        elbow_range >= 20
        and hip_range < 35
        and knee_range < 35
        and torso_range < 30
        and wrist_ratio < 0.70
    ):
        return "bench_press", max(confidence, 0.80), True, "bench_pattern_detected"

    # -----------------------------
    # PUSH PRESS
    # -----------------------------
    if (
        wrist_ratio > 0.55
        and knee_range > 12
        and elbow_range >= 20
    ):
        return "push_press", max(confidence, 0.78), True, "overhead_press_detected"

    # -----------------------------
    # TRUST MODEL: SQUAT
    # Put this BEFORE deadlift
    # -----------------------------
    if (
        raw_label == "squat"
        and knee_range >= 45
        and hip_range >= 25
        and min_knee < 100
    ):
        return "squat", max(confidence, 0.75), True, "trusted_squat_pattern"

    # -----------------------------
    # TRUST MODEL: DEADLIFT
    # Only reinforce deadlift
    # Never convert squat -> deadlift
    # -----------------------------
    if (
        raw_label == "deadlift"
        and wrist_ratio < 0.30
        and hip_range >= 20
        and torso_range >= 10
    ):
        return "deadlift", max(confidence, 0.85), True, "trusted_deadlift_hinge_pattern"

    # -----------------------------
    # DEFAULT = MODEL PREDICTION
    # -----------------------------
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


def build_set_summary(reps):
    if not reps:
        return {
            "detected_reps": 0,
            "avg_rep_score": 0,
            "best_rep": None,
            "worst_rep": None,
            "trend": "No clear reps detected.",
            "biggest_fix": "Record a clearer set for analysis.",
        }

    scores = [r["score"] for r in reps]

    # Count issues across reps
    issue_counts = {}
    feedback_counts = {}

    for rep in reps:
        for issue in rep.get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        for tip in rep.get("feedback", []):
            feedback_counts[tip] = feedback_counts.get(tip, 0) + 1

    # Priority coaching hierarchy
    priority_map = {
        "Bar drifts away from your body.": "Keep the bar closer — drag it up your legs.",
        "Overextending at lockout.": "Finish tall — squeeze glutes without leaning back.",
        "Incomplete lockout at the top.": "Finish tall by squeezing glutes and standing fully upright.",
        "Not enough hip hinge.": "Push your hips back more before starting the pull.",
        "Knees bend too much for a deadlift pattern.": "Keep shins more vertical and hinge from the hips.",
        "Back may be rounding during the pull.": "Brace your core and keep a neutral spine.",
        "Depth may be shallow.": "Try to reach better squat depth.",
        "Knees cave inward significantly.": "Knees are caving inward — drive them out over your toes.",
        "Mild knee cave detected.": "Keep knees tracking over your toes.",
        "Torso leaning too far forward.": "Chest is falling forward — keep torso upright.",
        "Dip is too shallow.": "Use a stronger dip to generate power.",
        "Incomplete overhead lockout.": "Fully extend arms overhead.",
        "Bar drift detected.": "Keep the bar path vertical and press straight overhead.",
        "Knees cave inward significantly during dip.": "Force knees out aggressively during the dip.",
        "Mild knee cave during dip.": "Keep knees tracking over toes.",
        "Bar not reaching full depth.": "Lower the bar fully to your chest.",
        "Incomplete lockout.": "Fully extend arms at the top.",
        "Elbows flaring excessively.": "Tuck elbows slightly and control the bar path.",
        "Elbows flaring slightly.": "Keep elbows slightly tucked.",
        "Weak leg drive.": "Keep feet planted and drive through your legs.",
    }

    biggest_fix = "Keep building consistent reps."

    # Choose highest-priority repeated issue
    for issue, fix in priority_map.items():
        if issue in issue_counts:
            biggest_fix = fix
            break

    # Fallback to most common useful feedback
    if biggest_fix == "Keep building consistent reps." and feedback_counts:
        useful_feedback = {
            tip: count
            for tip, count in feedback_counts.items()
            if "good" not in tip.lower()
            and "strong" not in tip.lower()
        }

        if useful_feedback:
            biggest_fix = max(useful_feedback, key=useful_feedback.get)

    return {
        "detected_reps": len(reps),
        "avg_rep_score": round(float(np.mean(scores)), 1),
        "best_rep": reps[int(np.argmax(scores))]["rep"],
        "worst_rep": reps[int(np.argmin(scores))]["rep"],
        "trend": (
            "Form appears to deteriorate as the set goes on."
            if len(scores) >= 2 and scores[-1] < scores[0]
            else "Form appears to improve as the set goes on."
            if len(scores) >= 2 and scores[-1] > scores[0]
            else "Form appears consistent across the set."
        ),
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


def analyze_deadlift_reps(biomechanics):
    hip_y = np.array([b["hip_y"] for b in biomechanics])
    torso = np.array([b["torso_angle"] for b in biomechanics])
    hip = np.array([b["hip_angle"] for b in biomechanics])
    knee = np.array([b["knee_angle"] for b in biomechanics])
    bar_distance = np.array([b.get("bar_distance", 0) for b in biomechanics])
    head_drop = np.array([b.get("head_drop", 0) for b in biomechanics])
    head_forward = np.array([b.get("head_forward", 0) for b in biomechanics])

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

            if end - start < 3:
                in_rep = False
                continue

            rep_signal = movement_signal[start:end + 1]
            rep_torso = torso[start:end + 1]
            rep_hip = hip[start:end + 1]
            rep_knee = knee[start:end + 1]
            rep_bar = bar_distance[start:end + 1]

            # ADD THESE
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

            # Top portion of rep (for lockout mechanics)
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

            # BACK
            if max_torso > 75:
                breakdown["back"] = "poor"
                issues.append("Back may be rounding during the pull.")
                feedback.append("Brace your core and keep a neutral spine.")
            elif max_torso > 60:
                breakdown["back"] = "fair"
                issues.append("Slight torso rounding detected.")
                feedback.append("Keep your chest proud and lats tight.")

            # HINGE
            if min_hip > 115:
                breakdown["hinge"] = "poor"
                issues.append("Not enough hip hinge.")
                feedback.append("Push your hips back more before starting the pull.")

            # KNEES
            if min_knee < 90:
                breakdown["knees"] = "fair"
                issues.append("Knees bend too much for a deadlift pattern.")
                feedback.append("Keep shins more vertical and hinge from the hips.")

            # BAR PATH
            if max_bar_distance > 0.12:
                breakdown["bar_path"] = "poor"
                issues.append("Bar drifts away from your body.")
                feedback.append("Keep the bar closer — drag it up your legs.")

            # NECK
            if max_head_drop > 0.08 or max_head_forward > 0.10:
                breakdown["neck"] = "poor"
                issues.append("Neck position may be off.")
                feedback.append("Keep your neck neutral — eyes slightly ahead on the floor.")

            # LOCKOUT — incomplete
            if end_hip < 140:
                breakdown["lockout"] = "incomplete"
                issues.append("Incomplete lockout at the top.")
                feedback.append(
                    "Finish tall by squeezing glutes and standing fully upright."
                )

            # LOCKOUT — hyperextend
            elif max_top_hip > 175 or max_top_torso > 50:
                breakdown["lockout"] = "poor"
                issues.append("Overextending at lockout.")
                feedback.append(
                    "Finish tall — squeeze glutes without leaning back."
                )

            # CONTROL
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

            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(start),
                "bottom_frame": int(start + np.argmax(rep_signal)),
                "end_frame": int(end),
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
        reps.append({
            "rep": 1,
            "start_frame": 0,
            "bottom_frame": int(np.argmax(torso)),
            "end_frame": len(biomechanics) - 1,
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


def analyze_squat_reps(biomechanics):
    knee_angles = np.array([b["knee_angle"] for b in biomechanics])
    torso_angles = np.array([b["torso_angle"] for b in biomechanics])
    valgus_ratios = np.array([b.get("valgus_ratio", 1.0) for b in biomechanics])
    heel_lifts = np.array([b.get("heel_lift", 0.0) for b in biomechanics])

    reps = []
    threshold = np.percentile(knee_angles, 35)

    in_rep = False
    start = 0

    for i, knee in enumerate(knee_angles):
        if not in_rep and knee < threshold:
            in_rep = True
            start = i

        elif in_rep and knee >= threshold:
            end = i

            if end - start < 3:
                in_rep = False
                continue

            rep_knee = knee_angles[start:end + 1]
            rep_torso = torso_angles[start:end + 1]
            rep_valgus = valgus_ratios[start:end + 1]
            rep_heel = heel_lifts[start:end + 1]

            bottom = start + int(np.argmin(rep_knee))

            # -----------------------------
            # Clean noisy MediaPipe spikes
            # -----------------------------
            clean_knee = np.clip(rep_knee, 45, 180)
            clean_torso = np.clip(rep_torso, 0, 90)
            clean_valgus = np.clip(rep_valgus, 0.75, 1.5)
            clean_heel = np.clip(rep_heel, -0.05, 0.08)

            min_knee = float(np.percentile(clean_knee, 20))
            torso_score = float(np.percentile(clean_torso, 75))
            valgus_score = float(np.percentile(clean_valgus, 25))
            max_heel_lift = float(np.percentile(clean_heel, 90))

            issues = []
            feedback = []

            # -----------------------------
            # DEPTH
            # -----------------------------
            if min_knee <= 105:
                depth_grade = "good"
            elif min_knee <= 120:
                depth_grade = "borderline"
                issues.append("Depth is close, but could be slightly lower.")
                feedback.append("Sink a little deeper while keeping your chest up.")
            else:
                depth_grade = "poor"
                issues.append("Depth may be shallow.")
                feedback.append("Try to reach better squat depth.")

            # -----------------------------
            # TORSO
            # -----------------------------
            if torso_score <= 55:
                torso_grade = "good"
            elif torso_score <= 75:
                torso_grade = "borderline"
                issues.append("Slight forward torso lean detected.")
                feedback.append("Stay braced and keep your chest proud.")
            else:
                torso_grade = "poor"
                issues.append("Torso angle could stay a little taller.")
                feedback.append("Stay braced and keep your chest proud out of the hole.")

            # -----------------------------
            # KNEES / VALGUS
            # -----------------------------
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

            # -----------------------------
            # HEELS
            # -----------------------------
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

            butt_wink_grade = "not_detected"

            # -----------------------------
            # SCORE
            # -----------------------------
            score = 10.0

            if depth_grade == "borderline":
                score -= 0.8
            elif depth_grade == "poor":
                score -= 1.8

            if torso_grade == "borderline":
                score -= 0.6
            elif torso_grade == "poor":
                score -= 1.3

            if knees_grade == "borderline":
                score -= 0.7
            elif knees_grade == "poor":
                score -= 1.3

            if heels_grade == "borderline":
                score -= 0.6
            elif heels_grade == "poor":
                score -= 1.2

            score = round(max(score, 1.0), 1)

            if not feedback:
                feedback = ["Strong squat rep. Keep bracing and driving through the floor."]

            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(start),
                "bottom_frame": int(bottom),
                "end_frame": int(end),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": {
                    "depth": depth_grade,
                    "torso": torso_grade,
                    "knees": knees_grade,
                    "heels": heels_grade,
                    "butt_wink": butt_wink_grade,
                },
                "feedback": feedback,
            })

            in_rep = False

    return reps, build_set_summary(reps)


def analyze_push_press_reps(biomechanics):
    knee = np.array([b["knee_angle"] for b in biomechanics])
    wrist_y = np.array([b["wrist_y"] for b in biomechanics])
    shoulder_y = np.array([b["shoulder_y"] for b in biomechanics])
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics])
    valgus = np.array([b.get("valgus_ratio", 1.0) for b in biomechanics])

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

            # ---- CLEAN NOISY POSE DATA ----
            clean_knee = np.clip(rep_knee, 70, 180)
            clean_valgus = np.clip(rep_valgus, 0.7, 1.5)
            clean_wrist_x = np.clip(
                rep_wrist_x,
                np.percentile(rep_wrist_x, 10),
                np.percentile(rep_wrist_x, 90),
            )

            min_knee = float(np.percentile(clean_knee, 10))
            wrist_above = float(np.mean(rep_wrist_y < rep_shoulder_y))
            min_valgus = float(np.percentile(clean_valgus, 15))
            wrist_drift = float(
                np.percentile(clean_wrist_x, 90)
                - np.percentile(clean_wrist_x, 10)
            )

            # --- BAR DRIFT ---
            if wrist_drift > 0.05:
                drift_severity = "severe"
            elif wrist_drift > 0.03:
                drift_severity = "moderate"
            else:
                drift_severity = "minor"

            issues = []
            feedback = []

            # --- DIP ---
            if min_knee > 172:
                issues.append("Dip is too shallow.")
                feedback.append("Use a stronger dip to generate power.")

            # --- LOCKOUT ---
            if wrist_above < 0.35:
                issues.append("Incomplete overhead lockout.")
                feedback.append("Fully extend arms overhead.")

            # --- BAR PATH ---
            if wrist_drift > 0.03:
                issues.append("Bar drift detected.")
                feedback.append(
                    "Keep the bar path vertical and press straight overhead."
                )

            # --- KNEES ---
            if min_valgus < 0.65:
                issues.append("Knees cave inward significantly during dip.")
                feedback.append("Force knees out aggressively during the dip.")
            elif min_valgus < 0.8:
                issues.append("Mild knee cave during dip.")
                feedback.append("Keep knees tracking over toes.")

            # --- SCORING ---
            base_score = 10.0
            penalty = 0

            for issue in issues:
                if "Bar drift" in issue:
                    penalty += 2.0
                elif "lockout" in issue.lower():
                    penalty += 1.5
                elif "knees" in issue.lower():
                    penalty += 1.2
                elif "dip" in issue.lower():
                    penalty += 1.0
                else:
                    penalty += 1.0

            if drift_severity == "severe":
                penalty += 1.0
            elif drift_severity == "moderate":
                penalty += 0.5

            score = base_score - penalty
            score = max(1.0, round(score, 1))

            # --- GOOD REP FLOOR ---
            if (
                min_knee <= 172
                and wrist_above >= 0.35
                and drift_severity == "minor"
                and min_valgus >= 0.8
            ):
                score = max(score, 9.0)

            breakdown = {
                "dip": "good" if min_knee <= 172 else "shallow",
                "lockout": "good" if wrist_above >= 0.35 else "incomplete",
                "bar_path": "drifting" if wrist_drift > 0.03 else "good",
                "bar_severity": drift_severity,
                "knees": (
                    "poor" if min_valgus < 0.65
                    else "borderline" if min_valgus < 0.8
                    else "good"
                ),
            }

            score = apply_coach_reward(score, issues, breakdown)

            if not issues:
                score = max(score, 9.0)

            # --- CLEAN UP BORDERLINE FLAGS ON EXCELLENT REPS ---
            if score >= 9.0 and issues == ["Mild knee cave during dip."]:
                issues = []
                feedback = ["Good push press rep."]
                breakdown["knees"] = "good"

            if score >= 9.0 and not issues:
                feedback = ["Good push press rep."]

            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(start),
                "end_frame": int(end),
                "score": round(score, 1),
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback or ["Good push press rep."],
            })

            in_rep = False

    return reps, build_set_summary(reps)


def draw_ideal_push_press_overlay(frame, pose_landmarks, width, height):
    """
    Draw ideal push press path:
    - cyan corridor
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

    # use midpoint of left/right joints
    left_shoulder = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)

    left_hip = pt(mp_pose.PoseLandmark.LEFT_HIP.value)
    right_hip = pt(mp_pose.PoseLandmark.RIGHT_HIP.value)

    left_wrist = pt(mp_pose.PoseLandmark.LEFT_WRIST.value)
    right_wrist = pt(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    shoulder = (left_shoulder + right_shoulder) / 2
    hip = (left_hip + right_hip) / 2
    wrist = (left_wrist + right_wrist) / 2

    # ideal vertical bar path = stacked over shoulders / midfoot line
    ideal_x = int(shoulder[0])

    top_y = int(min(wrist[1], shoulder[1]) - 60)
    bottom_y = int(hip[1] + 30)

    top_y = max(20, top_y)
    bottom_y = min(height - 20, bottom_y)

    start_pt = (ideal_x, bottom_y)
    end_pt = (ideal_x, top_y)

    # translucent corridor
    overlay = frame.copy()
    corridor_width = 26

    cv2.rectangle(
        overlay,
        (ideal_x - corridor_width, top_y),
        (ideal_x + corridor_width, bottom_y),
        (255, 255, 0),
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        0.18,
        frame,
        0.82,
        0,
    )

    # glow
    cv2.line(
        frame,
        start_pt,
        end_pt,
        (180, 255, 255),
        18,
        cv2.LINE_AA,
    )

    # center line
    cv2.line(
        frame,
        start_pt,
        end_pt,
        (255, 255, 0),
        7,
        cv2.LINE_AA,
    )

    # endpoints
    cv2.circle(
        frame,
        start_pt,
        11,
        (255, 255, 0),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        end_pt,
        11,
        (255, 255, 0),
        -1,
        cv2.LINE_AA,
    )

    # label
    label = "IDEAL BAR PATH"
    label_x = max(20, ideal_x - 90)
    label_y = max(40, top_y - 12)

    cv2.rectangle(
        frame,
        (label_x - 8, label_y - 28),
        (label_x + 195, label_y + 8),
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


def analyze_bench_press_reps(biomechanics):
    elbow = np.array([b["elbow_angle"] for b in biomechanics])
    wrist_y = np.array([b["wrist_y"] for b in biomechanics])
    shoulder_y = np.array([b["shoulder_y"] for b in biomechanics])
    hip_y = np.array([b["hip_y"] for b in biomechanics])
    knee = np.array([b["knee_angle"] for b in biomechanics])

    reps = []
    threshold = np.percentile(elbow, 35)

    in_rep = False
    start = 0

    for i, e in enumerate(elbow):
        if not in_rep and e < threshold:
            in_rep = True
            start = i

        elif in_rep and e >= threshold:
            end = i

            if end - start < 3:
                in_rep = False
                continue

            # Expand rep window so we don't miss true lockout/depth
            pad = 4
            s = max(0, start - pad)
            e2 = min(len(elbow), end + pad + 1)

            rep_elbow = elbow[s:e2]
            rep_wrist_y = wrist_y[s:e2]
            rep_shoulder_y = shoulder_y[s:e2]
            rep_hip_y = hip_y[s:e2]
            rep_knee = knee[s:e2]

            clean_elbow = np.clip(rep_elbow, 45, 180)
            clean_wrist_y = np.clip(rep_wrist_y, 0.0, 1.0)
            clean_shoulder_y = np.clip(rep_shoulder_y, 0.0, 1.0)
            clean_hip_y = np.clip(rep_hip_y, 0.0, 1.0)
            clean_knee = np.clip(rep_knee, 45, 180)

            max_elbow = float(np.percentile(clean_elbow, 92))
            elbow_p75 = float(np.percentile(clean_elbow, 75))

            wrist_range = float(
                np.percentile(clean_wrist_y, 90)
                - np.percentile(clean_wrist_y, 10)
            )

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

            # RANGE / DEPTH
            if wrist_range < 0.025:
                depth_status = "limited_range"
                issues.append("Range of motion may be limited.")
                feedback.append("Use a full, controlled press from chest to lockout.")
            elif bar_depth < -0.06:
                depth_status = "possibly_shallow"
                issues.append("Bar may not be reaching full depth.")
                feedback.append("Lower the bar under control toward your chest.")
            else:
                depth_status = "good"

            # LOCKOUT
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

            # ELBOW FLARE
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

            # ARCH
            arch_delta = shoulder_level - hip_level

            if arch_delta > 0.20:
                arch_status = "excessive"
                issues.append("Back arch may be excessive.")
                feedback.append("Keep a controlled arch without losing ribcage position.")
            else:
                arch_status = "controlled"

            # LEG DRIVE
            if feet_visible:
                if avg_knee < 95:
                    leg_status = "weak"
                    issues.append("Leg drive may be weak.")
                    feedback.append("Keep feet planted and drive through your legs.")
                else:
                    leg_status = "good"
            else:
                leg_status = "unknown"

            breakdown = {
                "depth": depth_status,
                "lockout": lockout_status,
                "elbows": elbow_status,
                "arch": arch_status,
                "legs": leg_status,
                "wrist_range": round(wrist_range, 3),
                "max_elbow": round(max_elbow, 1),
                "bar_depth": round(bar_depth, 3),
            }

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
                score = max(score, 9.0)
                feedback = ["Strong bench press rep. Maintain control and consistency."]

            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(start),
                "end_frame": int(end),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback,
                "visibility_notes": visibility_notes,
            })

            in_rep = False

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

    # Bench ideal path is a slightly diagonal line:
    # chest/lower position -> stacked lockout over shoulder
    lockout_pt = (
        int(shoulder[0]),
        int(min(wrist[1], elbow[1]) - 35),
    )

    chest_pt = (
        int(shoulder[0] - 45),
        int(shoulder[1] + 55),
    )

    # keep points inside frame
    lockout_pt = (
        max(20, min(width - 20, lockout_pt[0])),
        max(20, min(height - 20, lockout_pt[1])),
    )

    chest_pt = (
        max(20, min(width - 20, chest_pt[0])),
        max(20, min(height - 20, chest_pt[1])),
    )

    # translucent corridor
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

    # glow
    cv2.line(
        frame,
        chest_pt,
        lockout_pt,
        (180, 255, 255),
        18,
        cv2.LINE_AA,
    )

    # main line
    cv2.line(
        frame,
        chest_pt,
        lockout_pt,
        (255, 255, 0),
        7,
        cv2.LINE_AA,
    )

    # endpoint circles
    cv2.circle(frame, chest_pt, 11, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, lockout_pt, 11, (255, 255, 0), -1, cv2.LINE_AA)

    # label
    label = "IDEAL PRESS PATH"
    label_x = max(20, min(chest_pt[0], lockout_pt[0]) - 30)
    label_y = max(40, min(chest_pt[1], lockout_pt[1]) - 20)

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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("Overlay error: could not open video writer")
        cap.release()
        return None

    frame_idx = 0
    exercise = exercise_label.lower().replace(" ", "_")

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
            processed_idx = frame_idx // sample_every

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            draw_overlay = False

            for rep in rep_feedback:
                highlight_start = max(0, rep["start_frame"] - 30)
                highlight_end = rep["end_frame"] + 10

                if highlight_start <= processed_idx <= highlight_end:
                    draw_overlay = True
                    break

            if draw_overlay and results.pose_landmarks:

                # Same green skeleton for every lift
                frame = draw_user_skeleton(
                    frame,
                    results.pose_landmarks,
                )

                # Blue/cyan ideal guide on top
                if exercise == "deadlift":
                    frame = draw_ideal_deadlift(
                        frame,
                        results.pose_landmarks,
                        width,
                        height,
                    )

                elif exercise == "squat":
                    frame = draw_ideal_squat_overlay(
                        frame,
                        results.pose_landmarks,
                        width,
                        height,
                    )

                elif exercise == "push_press":
                    frame = draw_ideal_push_press_overlay(
                        frame,
                        results.pose_landmarks,
                        width,
                        height,
                    )

                elif exercise == "bench_press":
                    frame = draw_ideal_bench_press_overlay(
                        frame,
                        results.pose_landmarks,
                        width,
                        height,
                    )

            writer.write(frame)

    cap.release()
    writer.release()

    print("Overlay saved:", output_path)
    return output_path


def draw_ideal_deadlift(frame, pose_landmarks, width, height):
    lm = pose_landmarks.landmark

    def pt(idx):
        p = lm[idx]
        return np.array([p.x * width, p.y * height], dtype=np.float32)

    # use real body joints
    shoulder = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    hip = pt(mp_pose.PoseLandmark.RIGHT_HIP.value)
    knee = pt(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    ankle = pt(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    heel = pt(mp_pose.PoseLandmark.RIGHT_HEEL.value)
    foot = pt(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value)

    # reshape toward ideal deadlift
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
        cv2.line(
            frame,
            tuple(joints[i]),
            tuple(joints[i + 1]),
            blue,
            4,
            cv2.LINE_AA,
        )

    for p in joints:
        cv2.circle(
            frame,
            tuple(p),
            6,
            blue,
            -1,
            cv2.LINE_AA,
        )

    return frame


def draw_ideal_squat_overlay(frame, pose_landmarks, width, height):
    landmarks = pose_landmarks.landmark

    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * width, lm.y * height], dtype=np.float32)

    def vis(idx):
        return landmarks[idx].visibility

    left_ids = {
        "shoulder": 11,
        "hip": 23,
        "knee": 25,
        "ankle": 27,
    }

    right_ids = {
        "shoulder": 12,
        "hip": 24,
        "knee": 26,
        "ankle": 28,
    }

    left_vis = sum(vis(i) for i in left_ids.values())
    right_vis = sum(vis(i) for i in right_ids.values())

    ids = left_ids if left_vis >= right_vis else right_ids

    shoulder = pt(ids["shoulder"])
    hip = pt(ids["hip"])
    knee = pt(ids["knee"])
    ankle = pt(ids["ankle"])

    femur_len = np.linalg.norm(hip - knee)
    torso_len = np.linalg.norm(shoulder - hip)

    if femur_len < 5 or torso_len < 5:
        return frame

    # Direction knees travel relative to ankle
    forward_sign = np.sign(knee[0] - ankle[0])
    if forward_sign == 0:
        forward_sign = np.sign(shoulder[0] - hip[0])
    if forward_sign == 0:
        forward_sign = 1

    # Squat phase: lower hip = deeper squat
    phase = np.clip((ankle[1] - hip[1]) / max(femur_len * 1.6, 1), 0.0, 1.0)

    ideal_ankle = ankle.copy()

    # Knee tracks forward over toes
    ideal_knee = np.array([
        ankle[0] + forward_sign * femur_len * (0.28 + 0.18 * phase),
        ankle[1] - femur_len * (0.92 - 0.10 * phase),
    ])

    # Hip sits back opposite knee direction
    ideal_hip = np.array([
        ankle[0] - forward_sign * femur_len * (0.28 + 0.12 * phase),
        ankle[1] - femur_len * (1.38 - 0.22 * phase),
    ])

    # Shoulder stays over midfoot-ish with slight torso lean
    ideal_shoulder = np.array([
        ideal_hip[0] + forward_sign * torso_len * (0.55 + 0.10 * phase),
        ideal_hip[1] - torso_len * 0.95,
    ])

    blue = (255, 0, 0)

    pts = [
        tuple(ideal_shoulder.astype(int)),
        tuple(ideal_hip.astype(int)),
        tuple(ideal_knee.astype(int)),
        tuple(ideal_ankle.astype(int)),
    ]

    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(frame, a, b, blue, 5, cv2.LINE_AA)

    for p in pts:
        cv2.circle(frame, p, 7, blue, -1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "Blue = ideal squat guide",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        blue,
        2,
        cv2.LINE_AA,
    )

        # Bar guide for back squat: bar should stay over midfoot
    bar_point = ideal_shoulder.copy()
    midfoot = ideal_ankle.copy()

    cv2.circle(
        frame,
        tuple(bar_point.astype(int)),
        8,
        blue,
        -1,
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        tuple(bar_point.astype(int)),
        tuple(midfoot.astype(int)),
        blue,
        3,
        cv2.LINE_AA,
    )

    return frame


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "exercise_label": "Unknown",
            "confidence": 0.0,
            "analysis_mode": "video_error",
            "feedback": ["Could not open uploaded video."],
            "rep_feedback": [],
            "set_summary": build_set_summary([]),
            "overlay_video_url": None,
            "debug": {},
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(total_frames // 120, 1)

    sequence = []
    biomechanics = []
    frame_idx = 0
    pose_frames = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
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
            "overlay_video_url": None,
            "debug": {},
        }

    seq = pad_or_trim(np.array(sequence), target_len=30)
    seq = add_velocity(seq)

    probs = MODEL.predict_proba(seq)

    raw_idx = int(np.argmax(probs))
    raw_label = CLASS_NAMES[raw_idx]
    raw_confidence = float(probs[raw_idx])

    summary = summarize_biomechanics(biomechanics)

    label, confidence, override_used, reason = classify_with_biomechanics(
        raw_label,
        raw_confidence,
        summary,
        pose_frames,
    )

    analysis_mode = "classification_only"
    rep_feedback = []
    set_summary = build_set_summary([])

    if label == "squat":
        rep_feedback, set_summary = analyze_squat_reps(biomechanics)
        analysis_mode = "detailed_rep_analysis"

    elif label == "deadlift":
        rep_feedback, set_summary = analyze_deadlift_reps(biomechanics)
        analysis_mode = "detailed_rep_analysis"

    elif label == "push_press":
        rep_feedback, set_summary = analyze_push_press_reps(biomechanics)
        analysis_mode = "detailed_rep_analysis"

    elif label == "bench_press":
        rep_feedback, set_summary = analyze_bench_press_reps(biomechanics)
        analysis_mode = "detailed_rep_analysis"

    overlay_video_url = None

    if rep_feedback:
        overlay_filename = f"overlay_{uuid.uuid4().hex[:8]}.mp4"
        overlay_path = os.path.join(OVERLAY_DIR, overlay_filename)

        draw_overlay_video(
            input_path=video_path,
            output_path=overlay_path,
            rep_feedback=rep_feedback,
            exercise_label=label,
            sample_every=sample_every,
        )

        overlay_video_url = f"/outputs/{overlay_filename}"

    # Rebuild summary from final rep scores so overall score matches the rep cards
    set_summary = build_set_summary(rep_feedback)

    return {
        "exercise_label": label.replace("_", " ").title(),
        "confidence": round(confidence, 2),
        "analysis_mode": analysis_mode,
        "feedback": [
            f"Predicted exercise: {label.replace('_', ' ').title()}.",
            f"Model confidence: {round(confidence * 100, 1)}%.",
            f"Biomechanics override applied: {reason}."
            if override_used
            else "Model prediction used.",
        ],
        "rep_feedback": rep_feedback,
        "set_summary": set_summary,
        "overlay_video_url": overlay_video_url,
        "debug": {
            "original_prediction": raw_label,
            "original_confidence": round(raw_confidence, 4),
            "final_prediction": label,
            "override_used": override_used,
            "classification_reason": reason,
            "raw_predictions": dict(zip(CLASS_NAMES, probs.tolist())),
            "biomechanics": summary,
            "frames_seen": total_frames,
            "frames_processed": len(sequence),
            "pose_frames": pose_frames,
            "sample_every": sample_every,
            "model_input_shape": list(seq.shape),
        },
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename or "upload.mov").suffix or ".mov"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        result = analyze_video(video_path)

        print("\n====================")
        print("BACKEND RESULT")
        print(result)
        print("====================\n")

        return result

    except Exception as e:
        return {
            "error": True,
            "message": str(e),
        }