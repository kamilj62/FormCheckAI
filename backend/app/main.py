import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FormCheck AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path
from .model_runtime import NumpyFormCheckModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL = NumpyFormCheckModel(MODEL_DIR)

CLASS_NAMES = ["bench_press", "deadlift", "push_press", "squat"]
mp_pose = mp.solutions.pose


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": str(MODEL_PATH),
        "classes": CLASS_NAMES,
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

    wrist_above_shoulder = float(wrist_y < shoulder_y)

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
    print("min_knee:", min_knee)
    print("wrist_ratio:", wrist_ratio)

    # Push press override
    if wrist_ratio > 0.3:
        return "push_press", 0.9, True, "overhead_arm_position_detected"

    # PUSH PRESS DETECTION (override squat)
    wrist_ratio = summary.get("wrist_above_shoulder_ratio", 0)

    if wrist_ratio > 0.3:
        return "push_press", 0.9, True, "overhead_arm_position_detected"

    # DEADLIFT
    if (
        wrist_ratio < 0.20
        and hip_range >= 30
        and torso_range >= 12
        and min_knee >= 90
    ):
        return "deadlift", max(confidence, 0.85), True, "deadlift_hinge_pattern_detected"

    # BENCH PRESS
    if (
        wrist_ratio < 0.30
        and elbow_range >= 25
        and hip_range < 25
        and knee_range < 25
        and torso_range < 18
    ):
        return "bench_press", max(confidence, 0.80), True, "bench_pattern_detected"

    # PUSH PRESS
    if (
        wrist_ratio > 0.35
        and knee_range > 10
        and torso_range < 25
    ):
        return "push_press", max(confidence, 0.78), True, "overhead_press_detected"

    # SQUAT
    if (
        min_knee < 95
        and knee_range >= 40
        and hip_range >= 20
        and wrist_ratio < 0.25
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


def build_set_summary(reps):
    if not reps:
        return {
            "detected_reps": 0,
            "avg_rep_score": 0,
            "best_rep": None,
            "worst_rep": None,
            "trend": "No clear reps detected.",
        }

    scores = [r["score"] for r in reps]

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
    }


def apply_coach_reward(score, issues, breakdown):
    if len(issues) == 0:
        return max(score, 9.2)

    if len(issues) == 1:
        return max(score, 8.0)

    if len(issues) == 2:
        return max(score, 7.2)

    return score


def analyze_squat_reps(biomechanics):
    hip_y = np.array([b["hip_y"] for b in biomechanics])
    knee_angles = np.array([b["knee_angle"] for b in biomechanics])
    torso_angles = np.array([b["torso_angle"] for b in biomechanics])
    valgus_ratios = np.array([b.get("valgus_ratio", 1.0) for b in biomechanics])

    reps = []
    threshold = np.percentile(hip_y, 65)

    in_rep = False
    start = 0

    for i, y in enumerate(hip_y):
        if not in_rep and y > threshold:
            in_rep = True
            start = i

        elif in_rep and y <= threshold:
            end = i

            if end - start < 3:
                in_rep = False
                continue

            rep_hip_y = hip_y[start:end + 1]
            rep_knees = knee_angles[start:end + 1]
            rep_torso = torso_angles[start:end + 1]
            rep_valgus = valgus_ratios[start:end + 1]

            bottom = start + int(np.argmax(rep_hip_y))

            min_knee = float(np.min(rep_knees))
            max_torso = float(np.max(rep_torso))
            min_torso = float(np.min(rep_torso))
            min_valgus = float(np.min(rep_valgus))

            torso_change = max_torso - min_torso
            hip_change = float(np.max(rep_hip_y) - np.min(rep_hip_y))

            butt_wink_detected = (
                min_knee < 110
                and torso_change > 12
                and hip_change > 0.04
            )

            issues = []
            feedback = []

            if min_knee > 105:
                issues.append("Depth may be shallow.")
                feedback.append("Try to reach better squat depth.")

            if max_torso > 35:
                issues.append("Torso leaning too far forward.")
                feedback.append("Chest is falling forward — keep torso upright.")

            if min_valgus < 0.85:
                issues.append("Knees cave inward significantly.")
                feedback.append("Knees are caving inward — drive them out over your toes.")
            elif min_valgus < 1.0:
                issues.append("Mild knee cave detected.")
                feedback.append("Keep knees tracking over your toes.")

            if butt_wink_detected:
                issues.append("Possible butt wink detected at the bottom.")
                feedback.append("Brace harder and stop depth before your pelvis tucks under.")

            breakdown = {
                "depth": "good" if min_knee <= 105 else "needs_work",
                "torso": "good" if max_torso <= 35 else "poor",
                "knees": (
                    "poor" if min_valgus < 0.85
                    else "borderline" if min_valgus < 1.0
                    else "good"
                ),
                "butt_wink": "possible" if butt_wink_detected else "not_detected",
            }

            score = compute_rep_score(issues)
            score = apply_coach_reward(score, issues, breakdown)
            
            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(start),
                "bottom_frame": int(bottom),
                "end_frame": int(end),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback or ["Good squat rep."],
            })

            in_rep = False

    return reps, build_set_summary(reps)


def analyze_deadlift_reps(biomechanics):
    hip_y = np.array([b["hip_y"] for b in biomechanics])
    torso = np.array([b["torso_angle"] for b in biomechanics])
    hip = np.array([b["hip_angle"] for b in biomechanics])
    knee = np.array([b["knee_angle"] for b in biomechanics])

    reps = []

    # Deadlift bottom usually has hips lower / torso more folded.
    movement_signal = torso + (hip_y * 100)
    threshold = np.percentile(movement_signal, 60)

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

            rep_torso = torso[start:end + 1]
            rep_hip = hip[start:end + 1]
            rep_knee = knee[start:end + 1]

            max_torso = float(np.max(rep_torso))
            min_hip = float(np.min(rep_hip))
            min_knee = float(np.min(rep_knee))
            end_hip = float(rep_hip[-1])
            torso_change = float(np.max(rep_torso) - np.min(rep_torso))

            issues = []
            feedback = []

            breakdown = {
                "setup": "good",
                "back": "good",
                "hinge": "good",
                "lockout": "good",
                "knees": "good",
                "control": "good",
            }

            if max_torso > 65:
                breakdown["back"] = "poor"
                issues.append("Back may be rounding during the pull.")
                feedback.append("Brace your core and keep a neutral spine.")
            elif max_torso > 55:
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

            if end_hip < 145:
                breakdown["lockout"] = "incomplete"
                issues.append("Incomplete lockout at the top.")
                feedback.append("Finish tall by squeezing glutes and standing fully upright.")

            if torso_change < 8:
                breakdown["control"] = "poor"
                issues.append("Movement range was too small to confidently score.")
                feedback.append("Record the full rep from setup to lockout.")

            score = compute_rep_score(issues)
            score = apply_coach_reward(score, issues, breakdown)

            if not issues:
                score = 8.5

            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(start),
                "bottom_frame": int(start + np.argmax(movement_signal[start:end + 1])),
                "end_frame": int(end),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback or ["Strong rep. Focus on keeping the bar even closer and finishing explosively."],
            })

            in_rep = False

    # Fallback: if no reps detected, still return one coaching card
    if not reps and len(biomechanics) >= 10:
        max_torso = float(np.max(torso))
        min_hip = float(np.min(hip))
        min_knee = float(np.min(knee))

        issues = []
        feedback = []

        breakdown = {
            "setup": "review",
            "back": "good",
            "hinge": "good",
            "lockout": "review",
            "knees": "good",
            "control": "review",
        }

        if max_torso > 65:
            breakdown["back"] = "poor"
            issues.append("Back may be rounding during the pull.")
            feedback.append("Brace hard and keep your spine neutral.")
        elif max_torso > 55:
            breakdown["back"] = "fair"
            issues.append("Slight back rounding detected.")
            feedback.append("Keep your chest up and lats tight.")

        if min_hip > 115:
            breakdown["hinge"] = "poor"
            issues.append("Not enough hip hinge.")
            feedback.append("Push hips back more and keep the bar close.")

        if min_knee < 90:
            breakdown["knees"] = "fair"
            issues.append("Knees bend too much for a deadlift.")
            feedback.append("Keep shins more vertical and hinge through the hips.")

        score = compute_rep_score(issues, base_score=8.0)

        reps.append({
            "rep": 1,
            "start_frame": 0,
            "bottom_frame": int(np.argmax(torso)),
            "end_frame": len(biomechanics) - 1,
            "score": score,
            "grade": grade_score(score),
            "issues": issues or ["Could not clearly segment individual reps."],
            "breakdown": breakdown,
            "feedback": feedback or ["Deadlift detected, but the rep was hard to segment. Record the full body from the side for better scoring."],
        })

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

            min_knee = float(np.min(rep_knee))
            wrist_above = float(np.mean(rep_wrist_y < rep_shoulder_y))
            min_valgus = float(np.min(rep_valgus))

            # --- BAR DRIFT ---
            wrist_drift = float(np.max(rep_wrist_x) - np.min(rep_wrist_x))

            if wrist_drift > 0.04:
                drift_severity = "severe"
            elif wrist_drift > 0.02:
                drift_severity = "moderate"
            else:
                drift_severity = "minor"

            issues = []
            feedback = []

            # --- DIP ---
            if min_knee > 140:
                issues.append("Dip is too shallow.")
                feedback.append("Use a stronger dip to generate power.")

            # --- LOCKOUT ---
            if wrist_above < 0.35:
                issues.append("Incomplete overhead lockout.")
                feedback.append("Fully extend arms overhead.")

            # --- BAR PATH ---
            if wrist_drift > 0.02:
                issues.append("Bar drift detected.")
                feedback.append("Keep the bar path vertical and press straight overhead.")

            # --- KNEES ---
            if min_valgus < 0.75:
                issues.append("Knees cave inward significantly during dip.")
                feedback.append("Force knees out aggressively during the dip.")
            elif min_valgus < 0.9:
                issues.append("Knees cave inward during dip.")
                feedback.append("Drive knees out during the dip phase.")
            elif min_valgus < 1.05:
                issues.append("Mild knee cave during dip.")
                feedback.append("Keep knees tracking over toes.")

            # --- SCORING (BALANCED) ---
            base_score = 10.0
            penalty = 0

            for issue in issues:
                if "Bar drift" in issue:
                    penalty += 2.0
                elif "lockout" in issue.lower():
                    penalty += 1.5
                elif "knees" in issue.lower():
                    penalty += 1.5
                elif "dip" in issue.lower():
                    penalty += 1.0
                else:
                    penalty += 1.0

            # extra severity penalty (small, not brutal)
            if drift_severity == "severe":
                penalty += 1.5
            elif drift_severity == "moderate":
                penalty += 0.75

            score = base_score - penalty
            score = max(1.0, round(score, 1))

            # --- GOOD REP FLOOR ---
            if (
                min_knee <= 140
                and wrist_above >= 0.35
                and drift_severity in ["minor"]
                and min_valgus >= 1.05
            ):
                score = max(score, 9.0)

                if score >= 7:
                    issues = []
                    feedback = ["Good push press rep."]

            breakdown = {
                "dip": "good" if min_knee <= 140 else "shallow",
                "lockout": "good" if wrist_above >= 0.35 else "incomplete",
                "bar_path": "drifting" if wrist_drift > 0.02 else "good",
                "bar_severity": drift_severity,
                "knees": (
                    "poor" if min_valgus < 0.75
                    else "borderline" if min_valgus < 1.05
                    else "good"
                ),
            }

            score = apply_coach_reward(score, issues, breakdown)

            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(start),
                "end_frame": int(end),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": breakdown,
                "feedback": feedback or ["Good push press rep."],
            })           

            in_rep = False

    return reps, build_set_summary(reps)


def analyze_bench_press_reps(biomechanics):
    elbow = np.array([b["elbow_angle"] for b in biomechanics])
    wrist = np.array([b["wrist_y"] for b in biomechanics])
    shoulder = np.array([b["shoulder_y"] for b in biomechanics])
    hip = np.array([b["hip_y"] for b in biomechanics])
    knee = np.array([b["knee_angle"] for b in biomechanics])

    reps = []
    threshold = np.percentile(elbow, 30)

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

            rep_elbow = elbow[start:end + 1]
            rep_wrist = wrist[start:end + 1]
            rep_shoulder = shoulder[start:end + 1]
            rep_hip = hip[start:end + 1]
            rep_knee = knee[start:end + 1]

            min_elbow = float(np.min(rep_elbow))
            max_elbow = float(np.max(rep_elbow))
            elbow_p75 = float(np.percentile(rep_elbow, 75))

            bar_depth = float(np.mean(rep_wrist > rep_shoulder))
            arch_ratio = float(np.mean(rep_hip < rep_shoulder))
            avg_knee = float(np.mean(rep_knee))

            issues = []
            feedback = []

            # -------------------
            # DEPTH (FIXED)
            # -------------------
            if bar_depth < 0.10:
                issues.append("Bar not reaching full depth.")
                feedback.append("Lower the bar fully to your chest.")

            # -------------------
            # LOCKOUT (FIXED)
            # -------------------
            if max_elbow < 130:
                issues.append("Incomplete lockout.")
                feedback.append("Fully extend arms at the top.")

            # -------------------
            # ELBOWS
            # -------------------
            if elbow_p75 > 145:
                issues.append("Elbows flaring excessively.")
                feedback.append("Tuck elbows slightly and control the bar path.")
                elbow_status = "severe_flare"
            elif elbow_p75 > 130:
                issues.append("Elbows flaring slightly.")
                feedback.append("Keep elbows slightly tucked.")
                elbow_status = "flared"
            else:
                elbow_status = "good"

            # -------------------
            # ARCH (FIXED)
            # -------------------
            if arch_ratio > 0.85:
                issues.append("Excessive back arch.")
                feedback.append("Keep ribcage down and maintain controlled arch.")
                arch_status = "excessive"
            else:
                arch_status = "controlled"

            # -------------------
            # LEG DRIVE
            # -------------------
            if avg_knee < 110:
                issues.append("Weak leg drive.")
                feedback.append("Keep feet planted and drive through your legs.")
                leg_status = "none"
            else:
                leg_status = "good"

            # -------------------
            # SCORING (FIXED)
            # -------------------
            score = compute_rep_score(issues)
            score = apply_coach_reward(score, issues, breakdown)
            
            if not issues:
                score = 8.5
            elif score >= 8 and len(issues) >= 2:
                score = 7.0

            # -------------------
            # OUTPUT
            # -------------------
            reps.append({
                "rep": len(reps) + 1,
                "start_frame": int(start),
                "end_frame": int(end),
                "score": score,
                "grade": grade_score(score),
                "issues": issues,
                "breakdown": {
                    "depth": "good" if bar_depth >= 0.10 else "shallow",
                    "lockout": "good" if max_elbow >= 130 else "incomplete",
                    "elbows": elbow_status,
                    "arch": arch_status,
                    "legs": leg_status,
                },
                "feedback": feedback or ["Strong bench press rep. Maintain control and consistency."],
            })

            in_rep = False

    return reps, build_set_summary(reps)    


def is_video_usable(biomechanics):
    if len(biomechanics) < 10:
        return False

    # Check how often key points are visible
    visible_frames = 0

    for b in biomechanics:
        if (
            b["hip_y"] is not None
            and b["shoulder_y"] is not None
            and b["wrist_y"] is not None
        ):
            visible_frames += 1

    visibility_ratio = visible_frames / len(biomechanics)

    return visibility_ratio > 0.6


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
            "debug": {},
        }

    seq = pad_or_trim(np.array(sequence), target_len=30)
    seq = add_velocity(seq)

    probs = MODEL.predict_proba(seq)

    raw_idx = int(np.argmax(probs))
    raw_label = CLASS_NAMES[raw_idx]
    raw_confidence = float(probs[raw_idx])

    summary = summarize_biomechanics(biomechanics)

    # Smart video quality check
    
    torso_range = summary.get("max_torso_angle", 0) - summary.get("min_torso_angle", 0)
    hip_range = summary.get("max_hip_angle", 0) - summary.get("min_hip_angle", 0)
    pose_ratio = pose_frames / max(total_frames, 1)

    issues = []

    if pose_ratio < 0.08:
        issues.append("Pose detection is unstable")

    if len(sequence) < 8:
        issues.append("Not enough movement captured")

    # Only apply strict camera-angle checks to bench press.
    # Squat/deadlift/push press naturally have large torso/hip movement.
    if raw_label == "bench_press":
        if torso_range > 20:
            issues.append("Camera too close or upper body not fully visible")

        if hip_range > 40:
            issues.append("Lower body not visible or angle is unclear")
        
        if len(issues) > 0:
            return {
                "exercise_label": "Unknown",
                "confidence": 0.0,
                "analysis_mode": "poor_video_quality",
                "feedback": issues,
                "rep_feedback": [],
                "set_summary": build_set_summary([]),
                "debug": {
                    "reason": "dynamic_video_quality_check",
                    "issues": issues,
                    "pose_ratio": round(pose_ratio, 2),
                    "torso_range": round(torso_range, 1),
                    "hip_range": round(hip_range, 1),
                    "original_prediction": raw_label,
                    "original_confidence": round(raw_confidence, 4),
                    "frames_seen": total_frames,
                    "pose_frames": pose_frames,
                },
            }

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

    return {
        "exercise_label": label.replace("_", " ").title(),
        "confidence": round(confidence, 2),
        "analysis_mode": analysis_mode,
        "feedback": [
            f"Predicted exercise: {label.replace('_', ' ').title()}.",
            f"Model confidence: {round(confidence * 100, 1)}%.",
            f"Biomechanics override applied: {reason}." if override_used else "Model prediction used.",
        ],
        "rep_feedback": rep_feedback,
        "set_summary": set_summary,
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

        return analyze_video(video_path)

    except Exception as e:
        return {
            "error": True,
            "message": str(e),
        }