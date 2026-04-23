import os
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .model_runtime import (
    LABELS,
    LABEL_DISPLAY,
    NumpyFormCheckModel,
    SEQ_LEN,
    extract_features,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = FastAPI(title="FormCheck Real Inference Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = NumpyFormCheckModel(MODEL_DIR)
mp_pose = mp.solutions.pose


def build_coaching_notes(label, probs):
    return [
        f"Predicted exercise: {LABEL_DISPLAY.get(label, label)}.",
        f"Model confidence: {float(np.max(probs)):.1%}.",
    ]


def generate_squat_feedback(metrics):
    if not metrics:
        return []

    hip = [m["hip_depth"] for m in metrics]
    knees = [m["knee_track_mean"] for m in metrics]
    torso = [abs(m["torso_angle"]) for m in metrics]

    feedback = []

    if min(hip) > -0.15:
        feedback.append("Go deeper — hips are not reaching parallel.")

    if np.mean(knees) < -0.04:
        feedback.append("Knees are caving inward — push them out.")

    if max(torso) > 40:
        feedback.append("Chest is falling forward — keep torso upright.")

    if not feedback:
        feedback.append("Squat mechanics look stable.")

    return feedback


def score_squat_rep(metrics):
    if not metrics:
        return 0.0, "Needs Work", ["No usable rep data."], {}

    hip = [m["hip_depth"] for m in metrics]
    knees = [m["knee_track_mean"] for m in metrics]
    torso = [abs(m["torso_angle"]) for m in metrics]

    score = 10.0
    issues = []
    breakdown = {}

    if min(hip) > -0.15:
        score -= 2.0
        issues.append("Depth is shallow.")
        breakdown["depth"] = "poor"
    else:
        breakdown["depth"] = "good"

    avg_knees = float(np.mean(knees))
    if avg_knees < -0.08:
        score -= 3.0
        issues.append("Knees cave inward significantly.")
        breakdown["knees"] = "poor"
    elif avg_knees < -0.04:
        score -= 1.5
        issues.append("Mild knee cave detected.")
        breakdown["knees"] = "borderline"
    else:
        breakdown["knees"] = "good"

    max_torso = max(torso)
    if max_torso > 55:
        score -= 3.0
        issues.append("Torso leaning too far forward.")
        breakdown["torso"] = "poor"
    elif max_torso > 40:
        score -= 1.5
        issues.append("Some forward chest collapse.")
        breakdown["torso"] = "borderline"
    else:
        breakdown["torso"] = "good"

    score = round(max(score, 0.0), 1)

    grade = (
        "Excellent" if score >= 9 else
        "Good" if score >= 7 else
        "Fair" if score >= 5 else
        "Needs Work"
    )

    return score, grade, issues, breakdown


def smooth_signal(x, window=7):
    x = np.array(x, dtype=np.float32)
    if len(x) < max(3, window):
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same")


def detect_squat_reps(metrics):
    """
    Looser balanced squat rep detector:
    - aims to detect more real reps
    - still tries to suppress duplicate bottoms
    """
    if len(metrics) < 18:
        return []

    hip = np.array([m["hip_depth"] for m in metrics], dtype=np.float32)
    knee = np.array([m["knee_angle_mean"] for m in metrics], dtype=np.float32)

    hip_s = smooth_signal(hip, window=7)
    knee_s = smooth_signal(knee, window=7)

    hip_n = (hip_s - np.mean(hip_s)) / (np.std(hip_s) + 1e-6)
    knee_n = (knee_s - np.mean(knee_s)) / (np.std(knee_s) + 1e-6)

    rep_signal = (-0.68 * hip_n) + (-0.32 * knee_n)

    peak_threshold = max(0.38, float(np.percentile(rep_signal, 68)))
    top_level = float(np.percentile(rep_signal, 35))

    min_spacing = max(16, int(len(rep_signal) * 0.06))
    min_rep_len = 8
    max_rep_len = 170
    min_signal_range = 0.32

    candidate_bottoms = []
    for i in range(1, len(rep_signal) - 1):
        if rep_signal[i] > rep_signal[i - 1] and rep_signal[i] >= rep_signal[i + 1]:
            if rep_signal[i] >= peak_threshold:
                candidate_bottoms.append(i)

    if not candidate_bottoms:
        return []

    bottoms = []
    for b in candidate_bottoms:
        if not bottoms:
            bottoms.append(b)
            continue

        if b - bottoms[-1] < min_spacing:
            if rep_signal[b] > rep_signal[bottoms[-1]]:
                bottoms[-1] = b
        else:
            bottoms.append(b)

    reps = []
    for b in bottoms:
        s = b
        while s > 0 and rep_signal[s] > top_level:
            s -= 1

        e = b
        while e < len(rep_signal) - 1 and rep_signal[e] > top_level:
            e += 1

        rep_len = e - s
        if rep_len < min_rep_len or rep_len > max_rep_len:
            continue

        signal_range = float(np.max(rep_signal[s:e + 1]) - np.min(rep_signal[s:e + 1]))
        if signal_range < min_signal_range:
            continue

        reps.append((s, b, e))

    if not reps:
        return []

    filtered = [reps[0]]
    for curr in reps[1:]:
        prev = filtered[-1]

        prev_s, prev_b, prev_e = prev
        curr_s, curr_b, curr_e = curr

        overlap = min(prev_e, curr_e) - max(prev_s, curr_s)
        prev_len = prev_e - prev_s
        curr_len = curr_e - curr_s

        if overlap > 0:
            overlap_ratio = overlap / max(1, min(prev_len, curr_len))
            if overlap_ratio > 0.60:
                if rep_signal[curr_b] > rep_signal[prev_b]:
                    filtered[-1] = curr
            else:
                filtered.append(curr)
        else:
            filtered.append(curr)

    return filtered


def build_rep_feedback(metrics, reps):
    results = []

    for (s, b, e) in reps:
        segment = metrics[s:e + 1]
        if len(segment) < 6:
            continue

        score, grade, issues, breakdown = score_squat_rep(segment)
        coaching = generate_squat_feedback(segment)

        results.append(
            {
                "rep": len(results) + 1,
                "start_frame": s,
                "bottom_frame": b,
                "end_frame": e,
                "score": score,
                "grade": grade,
                "issues": issues,
                "breakdown": breakdown,
                "feedback": coaching,
            }
        )

    return results


def summarize_rep_trends(rep_feedback):
    if not rep_feedback:
        return {
            "detected_reps": 0,
            "avg_rep_score": None,
            "best_rep": None,
            "worst_rep": None,
            "trend": "No reps detected.",
        }

    scores = [r["score"] for r in rep_feedback]
    best_rep = max(rep_feedback, key=lambda r: r["score"])["rep"]
    worst_rep = min(rep_feedback, key=lambda r: r["score"])["rep"]

    split_idx = max(1, len(scores) // 2)
    first_half = scores[:split_idx]
    second_half = scores[split_idx:] if split_idx < len(scores) else []

    trend = "Form stayed fairly consistent."
    if second_half:
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        if second_avg < first_avg - 0.3:
            trend = "Form appears to deteriorate as the set goes on."
        elif second_avg > first_avg + 0.3:
            trend = "Form appears to improve as the set goes on."

    return {
        "detected_reps": len(rep_feedback),
        "avg_rep_score": round(float(np.mean(scores)), 1),
        "best_rep": best_rep,
        "worst_rep": worst_rep,
        "trend": trend,
    }


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video")

    frame_buffer = []
    probs = []
    metrics = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                continue

            feat, _ = extract_features(res.pose_landmarks.landmark)
            if feat is None:
                continue

            metrics.append(
                {
                    "torso_angle": float(feat[51]),
                    "hip_depth": float(feat[52]),
                    "knee_track_mean": float(feat[55]),
                    "knee_angle_mean": float(feat[67]),
                }
            )

            frame_buffer.append(feat)
            if len(frame_buffer) > SEQ_LEN:
                frame_buffer.pop(0)

            if len(frame_buffer) >= SEQ_LEN:
                seq = frame_buffer[-SEQ_LEN:]
                vel = [np.zeros_like(seq[0])] + [seq[i] - seq[i - 1] for i in range(1, len(seq))]
                seq142 = np.stack(
                    [np.concatenate([seq[i], vel[i]]) for i in range(len(seq))]
                ).astype(np.float32)
                probs.append(MODEL.predict_proba(seq142))

    cap.release()

    if not probs:
        raise HTTPException(400, "No predictions")

    mean = np.mean(np.stack(probs), axis=0)
    idx = int(np.argmax(mean))
    label = LABELS[idx]

    feedback = build_coaching_notes(label, mean)
    rep_feedback = []
    set_summary = {
        "detected_reps": 0,
        "avg_rep_score": None,
        "best_rep": None,
        "worst_rep": None,
        "trend": "No reps detected.",
    }
    analysis_mode = "classification_only"

    if label == "squat":
        reps = detect_squat_reps(metrics)
        rep_feedback = build_rep_feedback(metrics, reps)

        if rep_feedback:
            analysis_mode = "detailed_rep_analysis"
            set_summary = summarize_rep_trends(rep_feedback)
            feedback.append("Detailed rep analysis completed.")
        else:
            feedback.append(
                "Exercise detected, but detailed rep analysis was limited for this clip."
            )
            feedback.append(
                "For best results, record from the side with your full body visible and complete 2-5 clear reps."
            )

    return {
        "exercise_label": LABEL_DISPLAY[label],
        "confidence": float(mean[idx]),
        "analysis_mode": analysis_mode,
        "feedback": feedback,
        "rep_feedback": rep_feedback,
        "set_summary": set_summary,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    print("UPLOAD RECEIVED")
    print("filename:", file.filename)
    print("content_type:", file.content_type)

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    if not suffix:
        suffix = ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name

        total_bytes = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            tmp.write(chunk)

    print("saved temp path:", tmp_path)
    print("total bytes:", total_bytes)

    try:
        result = analyze_video(tmp_path)
        print("ANALYSIS RESULT:", result)
        return result
    finally:
        os.remove(tmp_path)