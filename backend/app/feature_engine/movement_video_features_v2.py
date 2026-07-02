import numpy as np

from app.feature_engine.feature_names_v2 import FEATURE_NAMES


def f(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def velocity_stats(values):
    """Return mean and peak absolute frame-to-frame velocity."""
    arr = np.array(values, dtype=np.float32)
    if len(arr) < 2:
        return 0.0, 0.0

    vel = np.abs(np.diff(arr))
    return float(np.mean(vel)), float(np.max(vel))


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

    min_knee_time_pct = float(min_knee_idx / max(1, n))
    min_hip_time_pct = float(min_hip_idx / max(1, n))
    bottom_to_overhead_time = float(first_overhead - min_knee_time_pct)
    early_late_overhead_delta = float(last_overhead - first_overhead)

    wrist_motion = np.diff(np.array(wrist_y, dtype=np.float32))
    hip_motion = np.diff(np.array(hip_y, dtype=np.float32))

    wrist_vel_mean, wrist_vel_peak = velocity_stats(wrist_y)
    hip_vel_mean, hip_vel_peak = velocity_stats(hip_y)
    knee_vel_mean, knee_vel_peak = velocity_stats(knee)
    elbow_vel_mean, elbow_vel_peak = velocity_stats(elbow)

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

        min_knee_time_pct,
        min_hip_time_pct,

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

    wrist_arr = np.array(wrist_y, dtype=np.float32)
    hip_arr = np.array(hip_y, dtype=np.float32)

    wrist_path_length = float(np.sum(np.abs(np.diff(wrist_arr)))) if len(wrist_arr) > 1 else 0.0
    hip_path_length = float(np.sum(np.abs(np.diff(hip_arr)))) if len(hip_arr) > 1 else 0.0
    wrist_vertical_range = float(np.max(wrist_arr) - np.min(wrist_arr)) if len(wrist_arr) else 0.0
    hip_vertical_range = float(np.max(hip_arr) - np.min(hip_arr)) if len(hip_arr) else 0.0

    overhead_indices = np.where(overhead > 0.5)[0]
    overhead_jitter = float(np.std(wrist_arr[overhead_indices])) if len(overhead_indices) > 2 else 0.0

    late_start = int(len(wrist_arr) * 0.65)
    late_wrist_y_std = float(np.std(wrist_arr[late_start:])) if len(wrist_arr[late_start:]) > 2 else 0.0

    feats += [
        wrist_vel_mean,
        wrist_vel_peak,
        hip_vel_mean,
        hip_vel_peak,
        knee_vel_mean,
        knee_vel_peak,
        elbow_vel_mean,
        elbow_vel_peak,
        bottom_to_overhead_time,
        early_late_overhead_delta,

        wrist_path_length,
        hip_path_length,
        wrist_vertical_range,
        hip_vertical_range,
        overhead_jitter,
        late_wrist_y_std,
    ]

    feats = np.array(feats, dtype=np.float32)

    # Keep feature vector aligned with FEATURE_NAMES.
    if len(feats) != len(FEATURE_NAMES):
        raise ValueError(f"Feature length mismatch: got {len(feats)}, expected {len(FEATURE_NAMES)}")

    return feats


# Backward-compatible alias
build_oly_video_features = build_movement_video_features
