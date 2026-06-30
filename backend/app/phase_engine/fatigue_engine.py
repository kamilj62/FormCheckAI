import numpy as np


# -------------------------------------------------------
# REP FEATURE EXTRACTION
# -------------------------------------------------------

def extract_rep_features(records, rep):
    start, bottom, end = rep["start"], rep["bottom"], rep["end"]
    seg = records[start:end + 1]

    knee = np.array([r.get("knee", 0) for r in seg])

    if len(knee) < 3:
        return {"rom": 0, "stability": 0, "velocity": 0}

    rom = np.min(knee)
    stability = np.std(knee)
    velocity = np.mean(np.abs(np.gradient(knee)))

    return {
        "rom": rom,
        "stability": stability,
        "velocity": velocity
    }


# -------------------------------------------------------
# FATIGUE CURVE (SET-LEVEL LOGIC)
# -------------------------------------------------------

def compute_fatigue_curve(rep_features):
    rom = np.array([f["rom"] for f in rep_features])
    stability = np.array([f["stability"] for f in rep_features])
    velocity = np.array([f["velocity"] for f in rep_features])

    fatigue_curve = []

    for i in range(len(rep_features)):
        fatigue = 0

        # ROM degradation
        fatigue += (rom[0] - rom[i]) * 0.02

        # stability degradation
        fatigue += (stability[i] - stability[0]) * 0.5

        # velocity slowdown
        fatigue += (velocity[0] - velocity[i]) * 0.5

        fatigue_curve.append(fatigue)

    return fatigue_curve