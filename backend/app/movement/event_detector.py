import numpy as np


def _detect_squat_like_events(biomechanics):
    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=float)
    n = len(knee)

    velocity = np.gradient(knee)
    velocity_std = float(np.std(velocity)) or 1.0
    low_knee_threshold = np.percentile(knee, 20)

    bottom_candidates = np.where(
        (knee <= low_knee_threshold) &
        (np.abs(velocity) <= velocity_std * 0.5)
    )[0]

    bottom = int(bottom_candidates[len(bottom_candidates) // 2]) if len(bottom_candidates) else int(np.argmin(knee))

    after_bottom = velocity[bottom:]
    extension = bottom + int(np.argmax(after_bottom)) if len(after_bottom) else bottom

    post_extension = np.abs(velocity[extension:])
    stable_candidates = np.where(post_extension <= velocity_std * 0.3)[0]
    lockout = extension + int(stable_candidates[-1]) if len(stable_candidates) else n - 1

    return {
        "setup": 0,
        "bottom": bottom,
        "extension": extension,
        "lockout": lockout,
    }


def _detect_snatch_events(biomechanics):
    n = len(biomechanics)
    if n < 10:
        return {}

    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics], dtype=np.float32)
    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)
    hip_angle = np.array([b.get("hip_angle", 180.0) for b in biomechanics], dtype=np.float32)
    knee_angle = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=np.float32)

    overhead = wrist_y < shoulder_y

    if not np.any(overhead):
        return _detect_squat_like_events(biomechanics)

    overhead_idxs = np.where(overhead)[0]

    clusters = []
    current = [int(overhead_idxs[0])]
    max_gap = max(8, n // 60)

    for idx in overhead_idxs[1:]:
        idx = int(idx)
        if idx - current[-1] <= max_gap:
            current.append(idx)
        else:
            if len(current) >= 4:
                clusters.append(current)
            current = [idx]

    if len(current) >= 4:
        clusters.append(current)

    if not clusters:
        return _detect_squat_like_events(biomechanics)

    cluster = clusters[0]
    cluster_start = cluster[0]
    cluster_end = cluster[-1]

    local_end = min(cluster_end, cluster_start + max(8, n // 20))
    catch_window = np.arange(cluster_start, local_end + 1)
    catch_idx = int(catch_window[np.argmax(hip_y[catch_window])])
    catch_idx = max(3, min(catch_idx, n - 2))

    start_idx = max(0, catch_idx - max(30, n // 4))

    pre_catch_start = max(start_idx + 1, int(start_idx + (catch_idx - start_idx) * 0.35))
    pre_catch_end = max(pre_catch_start + 1, catch_idx)

    extension_score = hip_angle[pre_catch_start:pre_catch_end] + knee_angle[pre_catch_start:pre_catch_end]
    extension_idx = pre_catch_start + int(np.argmax(extension_score))
    extension_idx = max(start_idx + 2, min(extension_idx, catch_idx - 1))

    first_pull_idx = max(start_idx + 1, int(start_idx + (extension_idx - start_idx) * 0.45))

    search_end = min(n - 1, catch_idx + max(20, n // 10))
    finish_candidates = [i for i in range(catch_idx + 1, search_end + 1) if overhead[i]]

    if finish_candidates:
        finish_candidates = np.array(finish_candidates, dtype=int)
        finish_idx = int(finish_candidates[np.argmin(hip_y[finish_candidates])])
    else:
        finish_idx = min(n - 1, catch_idx + max(10, n // 25))

    finish_idx = max(catch_idx + 1, min(finish_idx, n - 1))

    return {
        "setup": start_idx,
        "first_pull": first_pull_idx,
        "extension": extension_idx,
        "catch": catch_idx,
        "lockout": finish_idx,
        "finish": finish_idx,
    }


def detect_movement_events(biomechanics, exercise_label=None):
    if not biomechanics:
        return {}

    label = str(exercise_label or "").lower().replace(" ", "_")

    if "snatch" in label:
        return _detect_snatch_events(biomechanics)

    return _detect_squat_like_events(biomechanics)
