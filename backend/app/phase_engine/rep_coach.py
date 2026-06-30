import numpy as np


# -------------------------------------------------------
# MAIN COACHING FUNCTION (IMPROVED SIGNAL VERSION)
# -------------------------------------------------------

def coach_rep(records, start, bottom, end):

    rep = records[start:end + 1]

    if len(rep) == 0:
        return {
            "score": 0,
            "strengths": [],
            "issues": ["Empty rep"],
            "phase_feedback": {},
            "priority_fix": "data issue"
        }

    # ---------------------------------------------------
    # EXTRACT SIGNALS
    # ---------------------------------------------------
    knee = np.array([r.get("knee", 0) for r in rep])
    hip = np.array([r.get("hip", 0) for r in rep])

    # ---------------------------------------------------
    # DEPTH (RANGE OF MOTION)
    # ---------------------------------------------------
    depth = np.clip((180 - np.min(knee)) / 90, 0, 1)

    # ---------------------------------------------------
    # STABILITY (VELOCITY BASED — IMPROVED)
    # ---------------------------------------------------
    velocity = np.gradient(knee)
    stability = np.clip(1 - np.std(velocity) / 5, 0, 1)

    # ---------------------------------------------------
    # VALGUS / ASYMMETRY (IMPROVED SIGNAL)
    # ---------------------------------------------------
    valgus = np.mean([
        abs(r.get("knee", 0) - r.get("hip", 0))
        for r in rep
    ]) / 10.0
    valgus = np.clip(valgus, 0, 1)

    # ---------------------------------------------------
    # SCORE
    # ---------------------------------------------------
    score_raw = (
        0.5 * depth +
        0.3 * stability +
        0.2 * (1 - valgus)
    )

    score = 2 + 8 * (score_raw ** 1.2)

    score = max(0, min(10, round(score, 1)))

    # ---------------------------------------------------
    # COACHING LOGIC
    # ---------------------------------------------------
    strengths = []
    issues = []
    phase = {}

    if depth > 0.7:
        strengths.append("Good squat depth")
        phase["bottom"] = "strong"
    else:
        issues.append("Insufficient depth")
        phase["bottom"] = "shallow"

    if stability < 0.6:
        issues.append("Unstable knee tracking")
        phase["descent"] = "unstable"
    else:
        strengths.append("Good control")

    if valgus > 0.6:
        issues.append("Knee valgus detected")
        priority = "knee alignment"
    else:
        priority = "depth control"

    return {
        "score": round(float(score), 2),
        "strengths": strengths,
        "issues": issues,
        "phase_feedback": phase,
        "priority_fix": priority
    }