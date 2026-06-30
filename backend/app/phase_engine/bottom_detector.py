import numpy as np


def find_bottom_v1(records, approx_bottom, radius=25):

    if not records:
        return None

    pool = [
        r for r in records
        if abs(r["frame"] - approx_bottom) <= radius
    ]

    if len(pool) < 5:
        pool = records

    knee = np.array([r.get("knee", 0.0) for r in pool])
    hip = np.array([r.get("hip", 0.0) for r in pool])

    stability = -(np.abs(np.gradient(knee)) + np.abs(np.gradient(hip)))

    kernel = np.ones(5) / 5
    stability = np.convolve(stability, kernel, mode="same")

    threshold = np.percentile(stability, 20)
    candidates = np.where(stability <= threshold)[0]

    if len(candidates) == 0:
        idx = int(np.argmax(stability))
    else:
        idx = int(np.median(candidates)) + 2

    idx = max(0, min(idx, len(pool) - 1))

    return pool[idx]