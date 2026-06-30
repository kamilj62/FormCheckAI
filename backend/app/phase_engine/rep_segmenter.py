import numpy as np


def segment_reps(records):

    signal = np.array([r["knee"] + r["hip"] for r in records])
    signal = np.convolve(signal, np.ones(7)/7, mode="same")
    velocity = np.gradient(signal)

    reps = []

    state = "idle"
    start = None
    bottom = None

    for i in range(1, len(signal)-1):

        going_down = velocity[i] < 0
        going_up = velocity[i] > 0

        # START REP
        if state == "idle" and going_down:
            state = "down"
            start = i
            bottom = i

        # TRACK BOTTOM
        elif state == "down":
            if signal[i] < signal[bottom]:
                bottom = i

            if going_up:
                state = "up"

        # END REP
        elif state == "up":

            if going_down:
                # reject micro oscillation
                continue

            # rep ends when we have moved enough upward
            if signal[i] > signal[bottom]:

                end = i

                if end - start > 25:
                    reps.append({
                        "start": start,
                        "bottom": bottom,
                        "end": end
                    })

                state = "idle"

    return reps