import numpy as np

def find_bottom_v4(records, approx_bottom, radius=25):
    pool = [
        r for r in records
        if abs(r["frame"] - approx_bottom) <= radius
    ]

    if not pool:
        pool = records

    knee = np.array([r.get("knee", 0.0) for r in pool])
    k = np.convolve(knee, np.ones(5)/5, mode="same")

    # plateau definition = stable low region
    threshold = np.percentile(k, 25)

    for i in range(len(k)):
        if k[i] <= threshold:
            return pool[i]

    return pool[int(np.argmin(k))]