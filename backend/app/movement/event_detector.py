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



def _detect_snatch_fallback_events(biomechanics):
    """Return an ordered snatch schema when overhead landmarks are unavailable."""
    n = len(biomechanics)
    if n < 10:
        return {}

    base = _detect_squat_like_events(biomechanics)

    setup_idx = max(0, min(int(base.get("setup", 0)), n - 6))
    catch_idx = max(
        setup_idx + 3,
        min(int(base.get("bottom", int(n * 0.70))), n - 3),
    )

    extension_idx = setup_idx + int((catch_idx - setup_idx) * 0.75)
    extension_idx = max(
        setup_idx + 2,
        min(extension_idx, catch_idx - 1),
    )

    first_pull_idx = setup_idx + int(
        (extension_idx - setup_idx) * 0.45
    )
    first_pull_idx = max(
        setup_idx + 1,
        min(first_pull_idx, extension_idx - 1),
    )

    finish_idx = max(
        catch_idx + 1,
        min(int(base.get("lockout", n - 1)), n - 1),
    )

    return {
        "setup": setup_idx,
        "first_pull": first_pull_idx,
        "extension": extension_idx,
        "catch": catch_idx,
        "lockout": finish_idx,
        "finish": finish_idx,
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
        return _detect_snatch_fallback_events(biomechanics)

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
        return _detect_snatch_fallback_events(biomechanics)

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



def _detect_clean_and_jerk_events(biomechanics):
    n = len(biomechanics)
    if n < 10:
        return {}

    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=float)
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics], dtype=float)
    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics], dtype=float)
    shoulder_y = np.array([b.get("shoulder_y", 0.5) for b in biomechanics], dtype=float)
    elbow = np.array([b.get("elbow_angle", 180.0) for b in biomechanics], dtype=float)

    clean_search_start = max(1, int(n * 0.12))
    clean_search_end = max(clean_search_start + 2, int(n * 0.65))

    knee_norm = (knee - np.min(knee)) / (np.ptp(knee) + 1e-6)
    hip_norm = (hip - np.min(hip)) / (np.ptp(hip) + 1e-6)
    extension_signal = 0.60 * knee_norm + 0.40 * hip_norm

    # Temporary extension estimate only for locating first overhead.
    clean_extension_idx = clean_search_start + int(
        np.argmax(extension_signal[clean_search_start:clean_search_end])
    )

    overhead = (wrist_y < shoulder_y) & (elbow > 145)
    overhead_candidates = np.where(overhead & (np.arange(n) > clean_extension_idx))[0]
    first_overhead_idx = int(overhead_candidates[0]) if len(overhead_candidates) else n - 1

    catch_start = max(1, int(n * 0.20))
    catch_end = max(catch_start + 2, min(int(n * 0.45), first_overhead_idx))

    rack_like = np.abs(wrist_y - shoulder_y) < 0.24

    candidates = [
        i for i in range(catch_start, catch_end)
        if rack_like[i] and knee[i] < 145
    ]

    if candidates:
        clean_catch_idx = int(candidates[np.argmin(knee[candidates])])
    else:
        clean_catch_idx = catch_start + int(np.argmin(knee[catch_start:catch_end]))

    # Final clean extension must occur before the clean catch.
    pre_catch_start = max(1, int(clean_catch_idx * 0.35))
    pre_catch_end = max(pre_catch_start + 2, clean_catch_idx)
    clean_extension_idx = pre_catch_start + int(
        np.argmax(extension_signal[pre_catch_start:pre_catch_end])
    )
    clean_extension_idx = max(1, min(clean_extension_idx, clean_catch_idx - 1))

    recovery_search_start = min(n - 2, clean_catch_idx + max(6, int(n * 0.05)))
    recovery_search_end = max(
        recovery_search_start + 2,
        min(first_overhead_idx - 2, clean_catch_idx + max(18, int(n * 0.30)), n - 1)
    )

    recovery_candidates = [
        i for i in range(recovery_search_start, recovery_search_end)
        if rack_like[i] and knee[i] > 130 and hip[i] > 125
    ]

    if recovery_candidates:
        clean_recovery_idx = int(max(recovery_candidates, key=lambda i: extension_signal[i]))
    else:
        clean_recovery_idx = recovery_search_start + int(
            np.argmax(extension_signal[recovery_search_start:recovery_search_end])
        )

    jerk_start = min(n - 2, max(clean_recovery_idx + 1, clean_catch_idx + max(4, int(n * 0.04))))
    jerk_end = max(jerk_start + 2, min(first_overhead_idx, n - 1))

    jerk_dip_idx = jerk_start + int(np.argmin(knee[jerk_start:jerk_end]))

    drive_start = min(n - 2, jerk_dip_idx + 1)
    drive_end = max(drive_start + 1, first_overhead_idx)
    jerk_drive_idx = drive_start + int(np.argmax(extension_signal[drive_start:drive_end]))

    jerk_catch_idx = max(first_overhead_idx, jerk_drive_idx + 1)
    stable_needed = 3

    for i in range(jerk_drive_idx + 1, n - stable_needed):
        stable_overhead = all(overhead[i:i + stable_needed])
        stable_lockout = np.mean(elbow[i:i + stable_needed]) > 150
        if stable_overhead and stable_lockout:
            jerk_catch_idx = i
            break

    jerk_catch_idx = max(jerk_catch_idx, jerk_drive_idx + 1)
    jerk_catch_idx = min(jerk_catch_idx, n - 2)

    min_finish_gap = max(12, int(n * 0.12))
    finish_idx = min(n - 1, jerk_catch_idx + max(min_finish_gap, int(n * 0.35)))

    for i in range(jerk_catch_idx + min_finish_gap, n - stable_needed):
        stable_overhead = all(overhead[i:i + stable_needed])
        stable_lockout = np.mean(elbow[i:i + stable_needed]) > 150
        standing = knee[i] > 145 and hip[i] > 140
        if stable_overhead and stable_lockout and standing:
            finish_idx = i
            break

    return {
        "setup": 0,
        "clean_extension": clean_extension_idx,
        "clean_catch": clean_catch_idx,
        "clean_recovery": clean_recovery_idx,
        "jerk_dip": jerk_dip_idx,
        "jerk_drive": jerk_drive_idx,
        "jerk_catch": jerk_catch_idx,
        "lockout": finish_idx,
        "finish": finish_idx,
    }


def _detect_clean_events(biomechanics):
    n = len(biomechanics)
    if n < 10:
        return {}

    hip_y = np.array([b.get("hip_y", 0.0) for b in biomechanics], dtype=np.float32)
    wrist_y = np.array([b.get("wrist_y", 1.0) for b in biomechanics], dtype=np.float32)
    shoulder_y = np.array([b.get("shoulder_y", 0.0) for b in biomechanics], dtype=np.float32)
    wrist_x = np.array([b.get("wrist_x", 0.0) for b in biomechanics], dtype=np.float32)
    shoulder_x = np.array([b.get("shoulder_x", 0.0) for b in biomechanics], dtype=np.float32)
    knee = np.array([b.get("knee_angle", 180.0) for b in biomechanics], dtype=np.float32)
    hip = np.array([b.get("hip_angle", 180.0) for b in biomechanics], dtype=np.float32)

    rack_distance = np.abs(wrist_x - shoulder_x)

    front_rack = (
        (np.abs(wrist_y - shoulder_y) < 0.22)
        & (rack_distance < 0.32)
        & (knee < 155)
    )

    idxs = np.where(front_rack)[0]
    if len(idxs) == 0:
        return {}

    clusters = []
    current = [int(idxs[0])]
    max_gap = max(8, n // 60)

    for idx in idxs[1:]:
        idx = int(idx)
        if idx - current[-1] <= max_gap:
            current.append(idx)
        else:
            if len(current) >= 3:
                clusters.append(current)
            current = [idx]

    if len(current) >= 3:
        clusters.append(current)

    for cluster in clusters:
        cluster_start = cluster[0]
        cluster_end = cluster[-1]

        if cluster_start < max(60, int(n * 0.18)):
            continue

        pre_ext_start = max(0, cluster_start - max(70, n // 5))
        pre_ext_end = max(pre_ext_start + 1, cluster_start)

        ext_score = hip[pre_ext_start:pre_ext_end] + knee[pre_ext_start:pre_ext_end]
        if len(ext_score) == 0 or float(np.max(ext_score)) < 300:
            continue

        local_end = min(cluster_end, cluster_start + max(8, n // 20))
        catch_window = np.arange(cluster_start, local_end + 1)
        catch_idx = int(catch_window[np.argmax(hip_y[catch_window])])
        catch_idx = max(3, min(catch_idx, n - 2))

        setup_idx = max(0, catch_idx - max(45, n // 4))

        pre_start = max(setup_idx + 1, int(setup_idx + (catch_idx - setup_idx) * 0.35))
        pre_end = max(pre_start + 1, catch_idx)

        extension_score = hip[pre_start:pre_end] + knee[pre_start:pre_end]
        true_extension_idx = pre_start + int(np.argmax(extension_score))
        true_extension_idx = max(setup_idx + 2, min(true_extension_idx, catch_idx - 1))

        first_pull_idx = max(
            setup_idx + 1,
            int(setup_idx + (true_extension_idx - setup_idx) * 0.45)
        )

        pull_under_idx = first_pull_idx + int((catch_idx - first_pull_idx) * 0.40)
        pull_under_idx = max(first_pull_idx + 1, min(pull_under_idx, catch_idx - 1))

        search_end = min(n - 1, catch_idx + max(20, n // 8))
        standing_candidates = [
            i for i in range(catch_idx + 1, search_end + 1)
            if hip_y[i] < hip_y[catch_idx] - 0.03
        ]

        if standing_candidates:
            finish_idx = int(standing_candidates[min(5, len(standing_candidates) - 1)])
        else:
            finish_idx = min(n - 1, catch_idx + max(10, n // 25))

        finish_idx = max(catch_idx + 1, min(finish_idx, n - 1))

        return {
            "setup": setup_idx,
            "first_pull": first_pull_idx,
            "extension": pull_under_idx,
            "clean_extension": true_extension_idx,
            "catch": catch_idx,
            "clean_catch": catch_idx,
            "finish": finish_idx,
            "lockout": finish_idx,
        }

    return {}

def detect_movement_events(biomechanics, exercise_label=None):
    if not biomechanics:
        return {}

    label = str(exercise_label or "").lower().replace(" ", "_")

    if "clean_and_jerk" in label or "clean jerk" in label:
        return _detect_clean_and_jerk_events(biomechanics)

    if label == "clean" or "clean" in label:
        return _detect_clean_events(biomechanics)

    if "snatch" in label:
        return _detect_snatch_events(biomechanics)

    return _detect_squat_like_events(biomechanics)
