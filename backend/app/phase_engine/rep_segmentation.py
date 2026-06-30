import numpy as np


def segment_reps(records, knee_key="knee", rest_pct=85, min_len=20):

    if not records:
        return []

    frames = np.array([r["frame"] for r in records])

    knee = np.array([
        r.get(knee_key, np.nan) for r in records
    ])

    # fix missing values
    if np.isnan(knee).any():
        for i in range(1, len(knee)):
            if np.isnan(knee[i]):
                knee[i] = knee[i - 1]

        if np.isnan(knee[0]):
            knee[0] = np.nanmean(knee)

    # smooth signal
    k = np.convolve(knee, np.ones(5)/5, mode="same")

    # rest detection (now k exists BEFORE use)
    rest_threshold = np.percentile(k, rest_pct)
    is_rest = k >= rest_threshold

    reps = []
    state = "rest"
    start = None

    for i in range(1, len(k)):

        if state == "rest" and not is_rest[i]:
            state = "down"
            start = i

        if state == "down" and k[i] < k[i - 1]:
            continue

        if state == "down" and k[i] > k[i - 1]:
            bottom = i - 1
            state = "up"

        if state == "up" and is_rest[i]:
            end = i

            if start is not None and end - start >= min_len:
                reps.append({
                    "start": int(frames[start]),
                    "bottom": int(frames[bottom]),
                    "end": int(frames[end])
                })

            state = "rest"
            start = None

    return reps