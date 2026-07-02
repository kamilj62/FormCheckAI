import numpy as np


def detect_movement_events(biomechanics, exercise_label):
    if not biomechanics:
        return {}

    n = len(biomechanics)

    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=float)
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics], dtype=float)
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics], dtype=float)

    events = {}

    events["setup"] = 0
    events["bottom"] = int(np.argmin(knee))
    events["extension"] = int(np.argmax(hip))
    events["lockout"] = int(np.argmin(wrist_y))  # lower y = higher on screen

    return events
