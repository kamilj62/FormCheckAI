import json
import os
import numpy as np

PROFILE_DIR = "profiles"


def load_profile(user_id):
    os.makedirs(PROFILE_DIR, exist_ok=True)
    path = f"{PROFILE_DIR}/{user_id}.json"

    if not os.path.exists(path):
        return {
            "user_id": user_id,
            "sessions": 0,
            "metrics": {
                "depth": 0.5,
                "stability": 0.5,
                "valgus": 0.5
            },
            "trend": {
                "depth": [],
                "stability": [],
                "valgus": []
            },
            "flags": []
        }

    with open(path, "r") as f:
        return json.load(f)


def save_profile(user_id, profile):
    path = f"{PROFILE_DIR}/{user_id}.json"
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)

def update_profile(profile, rep_results):

    if "trend" not in profile:
        profile["trend"] = {}

    if "score" not in profile["trend"]:
        profile["trend"]["score"] = []

    if "fatigue" not in profile["trend"]:
        profile["trend"]["fatigue"] = []

        if len(rep_results) == 0:
            return profile

    # ---------------------------------------------------
    # SAFE FEATURE EXTRACTION
    # (no dependency on removed debug system)
    # ---------------------------------------------------

    # use proxy signals from scores + fatigue context
    avg_score = np.mean([r.get("score", 0) for r in rep_results])
    avg_adjusted = np.mean([r.get("adjusted_score", r.get("score", 0)) for r in rep_results])

    avg_fatigue = np.mean([
        r.get("set_context", {}).get("fatigue_index", 0)
        for r in rep_results
    ])

    # ---------------------------------------------------
    # INITIALIZE SAFETY
    # ---------------------------------------------------
    if "metrics" not in profile:
        profile["metrics"] = {"depth": 0.5, "stability": 0.5, "valgus": 0.5}

    if "trend" not in profile:
        profile["trend"] = {"score": [], "fatigue": []}

    # ---------------------------------------------------
    # SMOOTH UPDATE (LEARNING SYSTEM)
    # ---------------------------------------------------
    alpha = 0.2

    profile["metrics"]["depth"] = (1 - alpha) * profile["metrics"]["depth"] + alpha * avg_score
    profile["metrics"]["stability"] = (1 - alpha) * profile["metrics"]["stability"] + alpha * avg_adjusted
    profile["metrics"]["valgus"] = (1 - alpha) * profile["metrics"]["valgus"] + alpha * (1 - avg_fatigue)

    # ---------------------------------------------------
    # TREND TRACKING
    # ---------------------------------------------------
    profile["trend"].setdefault("score", []).append(avg_score)
    profile["trend"].setdefault("fatigue", []).append(avg_fatigue)

    profile["sessions"] = profile.get("sessions", 0) + 1

    # ---------------------------------------------------
    # SIMPLE FLAGS (NOW STABLE)
    # ---------------------------------------------------
    profile["flags"] = []

    if profile["metrics"]["valgus"] < 0.4:
        profile["flags"].append("fatigue accumulation detected")

    if profile["metrics"]["depth"] < 0.5:
        profile["flags"].append("low performance baseline")

    if profile["metrics"]["stability"] < 0.5:
        profile["flags"].append("inconsistent movement pattern")

    return profile
