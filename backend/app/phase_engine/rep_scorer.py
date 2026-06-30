import numpy as np


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def _smooth(x, k=5):
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def _angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cos = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# -------------------------------------------------------
# CORE SCORING
# -------------------------------------------------------

def score_rep(records, start, bottom, end):
    """
    Returns:
    {
        score: float,
        breakdown: dict,
        feedback: list
    }
    """

    rep = records[start:end + 1]

    knee_angles = []
    hip_angles = []

    for r in rep:
        if "knee" in r and "hip" in r:
            knee_angles.append(r["knee"])
            hip_angles.append(r["hip"])

    if len(knee_angles) < 3:
        return {
            "score": 5.0,
            "breakdown": {},
            "feedback": ["Not enough data for scoring"]
        }

    knee_angles = np.array(knee_angles)
    hip_angles = np.array(hip_angles)

    # -------------------------------------------------------
    # METRICS
    # -------------------------------------------------------

    depth_score = np.clip((180 - np.min(knee_angles)) / 90, 0, 1)

    knee_stability = 1 - np.std(knee_angles) / 30
    hip_stability = 1 - np.std(hip_angles) / 30

    stability_score = np.clip((knee_stability + hip_stability) / 2, 0, 1)

    # tempo (smooth reps get better score)
    duration = len(rep)
    tempo_score = np.clip(duration / 60, 0, 1)

    # -------------------------------------------------------
    # FINAL SCORE
    # -------------------------------------------------------

    score = (
        0.5 * depth_score +
        0.3 * stability_score +
        0.2 * tempo_score
    ) * 10

    # -------------------------------------------------------
    # FEEDBACK
    # -------------------------------------------------------

    feedback = []

    if depth_score < 0.5:
        feedback.append("Go deeper in the squat")

    if stability_score < 0.6:
        feedback.append("Knee/hip stability needs improvement")

    if tempo_score < 0.4:
        feedback.append("Control your descent and avoid rushing")

    if not feedback:
        feedback.append("Good rep")

    return {
        "score": round(score, 2),
        "breakdown": {
            "depth": round(depth_score, 2),
            "stability": round(stability_score, 2),
            "tempo": round(tempo_score, 2),
        },
        "feedback": feedback
    }