import numpy as np

FEATURE_NAMES = [
    "knee_mean", "knee_std", "knee_min", "knee_max", "knee_delta",
    "hip_mean", "hip_std", "hip_min", "hip_max", "hip_delta",
    "elbow_mean", "elbow_std", "elbow_min", "elbow_max", "elbow_delta",
    "shoulder_mean", "shoulder_std", "shoulder_min", "shoulder_max", "shoulder_delta",
    "wrist_y_mean", "wrist_y_std", "wrist_y_min", "wrist_y_max", "wrist_y_delta",
    "hip_y_mean", "hip_y_std", "hip_y_min", "hip_y_max", "hip_y_delta",
    "wrist_shoulder_distance_mean", "wrist_shoulder_distance_std", "wrist_shoulder_distance_min", "wrist_shoulder_distance_max", "wrist_shoulder_distance_delta",
    "overhead_ratio", "has_overhead", "first_overhead_pct", "last_overhead_pct", "overhead_span_pct",
    "min_knee_angle", "min_hip_angle", "max_elbow_angle", "max_shoulder_angle",
    "min_knee_time_pct", "min_hip_time_pct",
    "max_wrist_motion", "mean_wrist_motion", "max_hip_motion", "mean_hip_motion",
    "overhead_near_bottom", "late_overhead_flag", "early_min_knee", "late_min_knee",
] + [f"pad_{i}" for i in range(80 - 53)]


def f(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def stats(arr):
    arr = np.array(arr, dtype=np.float32)
    if len(arr) == 0:
        return [0, 0, 0, 0, 0]
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(arr[-1] - arr[0]),
    ]

def build_movement_video_features(biomechanics):
    if not biomechanics:
        return np.zeros(80, dtype=np.float32)

    knee = [f(b.get("knee_angle", 180)) for b in biomechanics]
    hip = [f(b.get("hip_angle", 180)) for b in biomechanics]
    elbow = [f(b.get("elbow_angle", 0)) for b in biomechanics]
    shoulder = [f(b.get("shoulder_angle", 0)) for b in biomechanics]

    wrist_y = [f(b.get("wrist_y", 1)) for b in biomechanics]
    shoulder_y = [f(b.get("shoulder_y", 0)) for b in biomechanics]
    hip_y = [f(b.get("hip_y", 0)) for b in biomechanics]

    wrist_shoulder_distance = [
        f(b.get("wrist_shoulder_distance", 0)) for b in biomechanics
    ]

    overhead = np.array([
        1.0 if wrist_y[i] < shoulder_y[i] else 0.0
        for i in range(len(biomechanics))
    ], dtype=np.float32)

    n = len(biomechanics)

    overhead_idxs = np.where(overhead > 0.5)[0]
    if len(overhead_idxs):
        first_overhead = int(overhead_idxs[0]) / max(1, n)
        last_overhead = int(overhead_idxs[-1]) / max(1, n)
        overhead_span = (int(overhead_idxs[-1]) - int(overhead_idxs[0]) + 1) / max(1, n)
    else:
        first_overhead = 1.0
        last_overhead = 0.0
        overhead_span = 0.0

    min_knee_idx = int(np.argmin(knee)) if knee else 0
    min_hip_idx = int(np.argmin(hip)) if hip else 0

    wrist_motion = np.diff(np.array(wrist_y, dtype=np.float32))
    hip_motion = np.diff(np.array(hip_y, dtype=np.float32))

    feats = []

    feats += stats(knee)
    feats += stats(hip)
    feats += stats(elbow)
    feats += stats(shoulder)
    feats += stats(wrist_y)
    feats += stats(hip_y)
    feats += stats(wrist_shoulder_distance)

    feats += [
        float(np.mean(overhead)),
        float(np.max(overhead)),
        float(first_overhead),
        float(last_overhead),
        float(overhead_span),

        float(np.min(knee)),
        float(np.min(hip)),
        float(np.max(elbow)),
        float(np.max(shoulder)),

        float(min_knee_idx / max(1, n)),
        float(min_hip_idx / max(1, n)),

        float(np.max(np.abs(wrist_motion))) if len(wrist_motion) else 0.0,
        float(np.mean(np.abs(wrist_motion))) if len(wrist_motion) else 0.0,
        float(np.max(np.abs(hip_motion))) if len(hip_motion) else 0.0,
        float(np.mean(np.abs(hip_motion))) if len(hip_motion) else 0.0,

        # snatch signal: deep catch while overhead appears
        float(np.mean(overhead[min_knee_idx:min(n, min_knee_idx + 8)])) if n else 0.0,

        # C&J signal: overhead happens late after a deep catch
        float(first_overhead > 0.45),
        float(np.min(knee[:max(1, int(n * 0.55))])),
        float(np.min(knee[max(1, int(n * 0.55)):])) if n > 2 else 180.0,
    ]

    feats = np.array(feats, dtype=np.float32)

    if len(feats) < 80:
        feats = np.pad(feats, (0, 80 - len(feats)))
    else:
        feats = feats[:80]

    return feats


# Backward-compatible alias
build_oly_video_features = build_movement_video_features
