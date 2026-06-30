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

    if len(rep_results) == 0:
        return profile

    avg_depth = np.mean([r["debug"]["depth"] for r in rep_results])
    avg_stability = np.mean([r["debug"]["stability"] for r in rep_results])
    avg_valgus = np.mean([r["debug"]["valgus"] for r in rep_results])

    # exponential moving average (smooth learning)
    alpha = 0.2

    profile["metrics"]["depth"] = (1 - alpha) * profile["metrics"]["depth"] + alpha * avg_depth
    profile["metrics"]["stability"] = (1 - alpha) * profile["metrics"]["stability"] + alpha * avg_stability
    profile["metrics"]["valgus"] = (1 - alpha) * profile["metrics"]["valgus"] + alpha * avg_valgus

    profile["trend"]["depth"].append(avg_depth)
    profile["trend"]["stability"].append(avg_stability)
    profile["trend"]["valgus"].append(avg_valgus)

    profile["sessions"] += 1

    # simple flag system
    profile["flags"] = []

    if profile["metrics"]["valgus"] > 0.6:
        profile["flags"].append("knee valgus tendency")

    if profile["metrics"]["depth"] < 0.5:
        profile["flags"].append("shallow squat pattern")

    if profile["metrics"]["stability"] < 0.5:
        profile["flags"].append("unstable movement pattern")

    return profile