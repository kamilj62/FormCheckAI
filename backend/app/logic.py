import numpy as np

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

    # IMPORTANT:
    # If the model already predicts bench press, trust it.
    # Bench angles can make wrist_above_shoulder_ratio falsely high.
    obvious_push_up = (
        wrist_ratio < 0.10
        and min_torso >= 45
        and max_torso <= 85
        and elbow_range >= 60
        and min_elbow >= 70
    )

    if raw_label == "bench_press" and not obvious_push_up:
        return "bench_press", max(confidence, 0.80), True, "trusted_model_bench_press"

    if raw_label == "bench_press" and obvious_push_up:
        return "push_up", max(confidence, 0.86), True, "push_up_bodyweight_pattern"

    # BENCH PRESS fallback:
    # Use this only when the model did NOT predict bench,
    # but biomechanics still look like a bench press.
    # Disabled: this broad fallback misclassifies curls and chest-fly movements
    # as bench press. Real bench predictions are still preserved above by
    # trusted_model_bench_press.

    # PUSH PRESS:
    # Needs overhead position AND knee dip/drive.
    if (
        wrist_ratio > 0.55
        and knee_range > 12
        and elbow_range >= 20
    ):
        return "push_press", max(confidence, 0.78), True, "overhead_press_detected"

    # DEADLIFT:
    # Hinge pattern with limited wrist-over-shoulder time.
    if (
        wrist_ratio < 0.20
        and hip_range >= 30
        and torso_range >= 12
        and min_knee >= 90
    ):
        return "deadlift", max(confidence, 0.85), True, "deadlift_hinge_pattern_detected"

    # SQUAT:
    # Deep knee bend with hip/knee movement.
    if (
        min_knee < 95
        and knee_range >= 40
        and hip_range >= 20
        and wrist_ratio < 0.25
    ):
        return "squat", max(confidence, 0.75), True, "squat_pattern_detected"

    return raw_label, confidence, False, "model_prediction"

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

def apply_coach_reward(score, issues, breakdown):
    if len(issues) == 0:
        return max(score, 9.2)

    if len(issues) == 1:
        return max(score, 8.0)

    if len(issues) == 2:
        return max(score, 7.2)

    return score
