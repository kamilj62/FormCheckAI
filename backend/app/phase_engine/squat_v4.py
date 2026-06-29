import numpy as np

def find_bottom_v4(records, approx_bottom, radius=25):
    pool = [
        r for r in records
        if abs(r["frame"] - approx_bottom) <= radius
    ]

    if not pool:
        pool = records

    knee = np.array([r.get("knee", 0.0) for r in pool])
    hip = np.array([r.get("hip", 0.0) for r in pool])

    # raw motion energy (NO smoothing)
    energy = np.abs(np.gradient(knee)) + np.abs(np.gradient(hip))

    # take early part of minimum region instead of absolute minimum
    candidates = np.where(energy <= np.percentile(energy, 20))[0]
    idx = int(candidates[0]) if len(candidates) > 0 else int(np.argmin(energy))

    # 🔥 CRITICAL FIX: compensate MediaPipe lag
    # (this is the real-world correction factor)
    idx = max(0, idx - 12)

    return pool[idx]