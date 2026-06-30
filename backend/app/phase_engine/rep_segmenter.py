import numpy as np


def segment_reps(records):
    """
    Returns list of:
    {start, bottom, end}
    """

    if len(records) < 10:
        return []

    knee = np.array([r.get("knee", 0.0) for r in records])
    hip = np.array([r.get("hip", 0.0) for r in records])

    signal = knee + hip

    # smooth signal
    kernel = np.ones(5) / 5
    signal = np.convolve(signal, kernel, mode="same")

    velocity = np.gradient(signal)

    bottoms = []

    # detect local minima + low velocity
    for i in range(2, len(signal) - 2):
        if (
            signal[i] < signal[i - 1]
            and signal[i] < signal[i + 1]
            and abs(velocity[i]) < np.percentile(np.abs(velocity), 30)
        ):
            bottoms.append(i)

    if not bottoms:
        return []

    # cluster nearby bottoms
    clustered = []
    group = [bottoms[0]]

    for b in bottoms[1:]:
        if b - group[-1] < 10:
            group.append(b)
        else:
            clustered.append(int(np.mean(group)))
            group = [b]

    clustered.append(int(np.mean(group)))

    bottoms = clustered

    reps = []

    for i, b in enumerate(bottoms):

        if i == 0:
            start = max(0, b - 30)
        else:
            start = int((bottoms[i - 1] + b) / 2)

        if i == len(bottoms) - 1:
            end = min(len(records) - 1, b + 30)
        else:
            end = int((b + bottoms[i + 1]) / 2)

        reps.append({
            "start": start,
            "bottom": b,
            "end": end
        })

    return reps