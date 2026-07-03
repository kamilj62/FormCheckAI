import numpy as np


def build_oly_router_features(biomechanics, events):
    """
    Shared video-level Olympic routing features.

    Returns a dict of interpretable movement features.
    """

    if not biomechanics or not events:
        return {}

    n = len(biomechanics)

    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics], dtype=float)
    wrist_y = np.array([b.get("wrist_y", 0.0) for b in biomechanics], dtype=float)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=float)
    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=float)

    def idx(name, default=0):
        return max(0, min(int(events.get(name, default)), n - 1))

    setup = idx("setup")
    extension = idx("extension", events.get("clean_extension", 0))
    catch = idx("catch", events.get("clean_catch", 0))
    finish = idx("finish", events.get("lockout", n - 1))

    overhead = wrist_y < shoulder_y

    features = {
        "has_overhead": float(np.any(overhead)),
        "catch_overhead": float(overhead[catch]),
        "catch_depth": float(knee[catch]),
        "extension_to_catch": float(catch - extension),
        "catch_to_finish": float(finish - catch),
        "extension_height_change":
            float(hip_y[catch] - hip_y[extension]),
        "lockout_duration":
            float(np.sum(overhead[catch:finish + 1])),
    }

    return features
