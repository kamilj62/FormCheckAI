import numpy as np

def find_bottom_v4(records, approx_bottom, radius=25):

    pool = [
        r for r in records
        if abs(r["frame"] - approx_bottom) <= radius
    ]

    if len(pool) < 5:
        pool = records

    knee = np.array([r["knee"] for r in pool])
    hip = np.array([r["hip"] for r in pool])

    # stability signal (low movement = bottom region)
    stability = -(np.abs(np.gradient(knee)) + np.abs(np.gradient(hip)))

    # smooth signal a bit
    kernel = np.ones(5) / 5
    stability = np.convolve(stability, kernel, mode="same")

    # -------------------------------------------------------
    # YOUR DECISION BLOCK (THIS IS WHAT YOU ASKED ABOUT)
    # -------------------------------------------------------
    if len(candidates := np.where(stability <= np.percentile(stability, 20))[0]) == 0:
        idx = int(np.argmax(stability))
    else:
        idx = int(np.median(candidates)) + 2

    return pool[min(idx, len(pool) - 1)]