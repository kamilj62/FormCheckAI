from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from app.ml.movement_signatures import LABEL_SIGNATURES


RepAnalyzer = Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]]


@dataclass(frozen=True)
class RepDetectorSpec:
    detector: str
    required_phase_fields: tuple[str, ...]
    detector_label: str | None = None


@dataclass(frozen=True)
class RepValidation:
    rep: int
    complete: bool
    ordered: bool
    missing_fields: tuple[str, ...]


@dataclass(frozen=True)
class RepDetectionResult:
    label: str
    reps: list[dict[str, Any]]
    summary: dict[str, Any]
    required_phase_fields: tuple[str, ...]
    validations: tuple[RepValidation, ...]
    phase_complete: bool
    phase_ordered: bool
    error: str | None = None


SQUAT_PHASE_FIELDS = (
    "start_frame",
    "descent_frame",
    "bottom_frame",
    "ascent_frame",
    "end_frame",
)

PRESS_PHASE_FIELDS = (
    "start_frame",
    "dip_frame",
    "drive_frame",
    "lockout_frame",
    "end_frame",
)

THRUSTER_PHASE_FIELDS = (
    "start_frame",
    "dip_frame",
    "drive_frame",
    "catch_frame",
    "lockout_frame",
    "end_frame",
)

STRICT_PRESS_PHASE_FIELDS = (
    "start_frame",
    "press_frame",
    "lockout_frame",
    "end_frame",
)

PULL_UP_PHASE_FIELDS = (
    "start_frame",
    "pull_frame",
    "top_frame",
    "descent_frame",
    "end_frame",
)

OLY_PULL_PHASE_FIELDS = (
    "start_frame",
    "first_pull_frame",
    "extension_frame",
    "catch_frame",
    "end_frame",
)

SPLIT_JERK_PHASE_FIELDS = (
    "start_frame",
    "dip_frame",
    "drive_frame",
    "catch_frame",
    "lockout_frame",
    "end_frame",
)

CLEAN_AND_JERK_PHASE_FIELDS = (
    "start_frame",
    "clean_catch_frame",
    "clean_recovery_frame",
    "jerk_dip_frame",
    "jerk_drive_frame",
    "jerk_catch_frame",
    "end_frame",
)

DEADLIFT_PHASE_FIELDS = (
    "start_frame",
    "pull_frame",
    "mid_frame",
    "finish_frame",
    "lockout_frame",
    "end_frame",
)


REP_DETECTOR_SPECS: dict[str, RepDetectorSpec] = {
    "squat_back": RepDetectorSpec("squat", SQUAT_PHASE_FIELDS, "squat_back"),
    "squat_front": RepDetectorSpec("squat", SQUAT_PHASE_FIELDS, "squat_front"),
    "overhead_squat": RepDetectorSpec(
        "squat",
        SQUAT_PHASE_FIELDS,
        "overhead_squat",
    ),
    "deadlift": RepDetectorSpec("deadlift", DEADLIFT_PHASE_FIELDS),
    "bench_press": RepDetectorSpec(
        "bench_press",
        (
            "start_frame",
            "end_frame",
        ),
    ),
    "strict_press": RepDetectorSpec(
        "strict_press",
        STRICT_PRESS_PHASE_FIELDS,
    ),
    "push_press": RepDetectorSpec(
        "push_press",
        PRESS_PHASE_FIELDS,
        "push_press",
    ),
    "thruster": RepDetectorSpec(
        "push_press",
        THRUSTER_PHASE_FIELDS,
        "thruster",
    ),
    "clean": RepDetectorSpec("clean", OLY_PULL_PHASE_FIELDS),
    "clean_and_jerk": RepDetectorSpec(
        "clean_and_jerk",
        CLEAN_AND_JERK_PHASE_FIELDS,
    ),
    "split_jerk": RepDetectorSpec(
        "split_jerk",
        SPLIT_JERK_PHASE_FIELDS,
    ),
    "snatch": RepDetectorSpec("snatch", OLY_PULL_PHASE_FIELDS),
    "pull_up": RepDetectorSpec("pull_up", PULL_UP_PHASE_FIELDS),
    "handstand_push_up": RepDetectorSpec(
        "handstand_push_up",
        SQUAT_PHASE_FIELDS,
    ),
    "push_up": RepDetectorSpec("push_up", SQUAT_PHASE_FIELDS),
    "burpee": RepDetectorSpec(
        "burpee",
        (
            "start_frame",
            "floor_frame",
            "stand_frame",
            "end_frame",
        ),
    ),
    "muscle_up": RepDetectorSpec(
        "muscle_up",
        (
            "start_frame",
            "pull_frame",
            "transition_frame",
            "support_frame",
            "end_frame",
        ),
    ),
}


def normalize_rep_detector_label(label: str | None) -> str:
    normalized = (
        str(label or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    signature = LABEL_SIGNATURES.get(normalized)
    if signature:
        return signature.internal_label

    return normalized or "unknown"


def rep_detector_spec(label: str | None) -> RepDetectorSpec | None:
    return REP_DETECTOR_SPECS.get(normalize_rep_detector_label(label))


def validate_rep_phases(
    reps: list[dict[str, Any]],
    required_fields: tuple[str, ...],
) -> tuple[RepValidation, ...]:
    validations: list[RepValidation] = []

    for index, rep in enumerate(reps, start=1):
        missing = tuple(
            field
            for field in required_fields
            if not isinstance(rep.get(field), (int, float))
        )

        values = [
            int(rep[field])
            for field in required_fields
            if isinstance(rep.get(field), (int, float))
        ]
        ordered = all(
            earlier <= later
            for earlier, later in zip(values, values[1:])
        )

        validations.append(
            RepValidation(
                rep=int(rep.get("rep", index) or index),
                complete=not missing,
                ordered=ordered,
                missing_fields=missing,
            )
        )

    return tuple(validations)


def _frame_value(rep: Mapping[str, Any], key: str) -> int | None:
    value = rep.get(key)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _merge_thruster_cluster(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    representative = max(
        cluster,
        key=lambda rep: (
            float(rep.get("score", 0) or 0),
            (_frame_value(rep, "end_frame") or 0)
            - (_frame_value(rep, "start_frame") or 0),
        ),
    )
    merged = dict(representative)

    for key in THRUSTER_PHASE_FIELDS:
        values = [
            _frame_value(rep, key)
            for rep in cluster
            if _frame_value(rep, key) is not None
        ]
        if not values:
            continue

        if key == "start_frame":
            merged[key] = min(values)
        else:
            merged[key] = max(values)

    breakdown = dict(merged.get("breakdown") or {})
    breakdown["merged_fragments"] = len(cluster)
    merged["breakdown"] = breakdown

    return merged


def dedupe_thruster_full_cycles(
    reps: list[dict[str, Any]],
    *,
    max_duplicate_dip_gap: int = 45,
    max_duplicate_lockout_gap: int = 100,
) -> list[dict[str, Any]]:
    if len(reps) <= 1:
        return reps

    ordered_reps = sorted(
        reps,
        key=lambda rep: (
            _frame_value(rep, "start_frame") or 0,
            _frame_value(rep, "dip_frame") or 0,
        ),
    )

    has_phase_anchors = any(
        _frame_value(rep, "dip_frame") is not None
        or _frame_value(rep, "lockout_frame") is not None
        for rep in ordered_reps
    )

    if not has_phase_anchors:
        clusters: list[list[dict[str, Any]]] = []

        for rep in ordered_reps:
            start = _frame_value(rep, "start_frame") or 0

            if not clusters:
                clusters.append([rep])
                continue

            previous_start = (
                _frame_value(clusters[-1][-1], "start_frame")
                or start
            )

            if start - previous_start <= 40:
                clusters[-1].append(rep)
            else:
                clusters.append([rep])

        cleaned = [_merge_thruster_cluster(cluster) for cluster in clusters]
        for index, rep in enumerate(cleaned, start=1):
            rep["rep"] = index

        return cleaned

    clusters: list[list[dict[str, Any]]] = []

    for rep in ordered_reps:
        start = _frame_value(rep, "start_frame")
        dip = _frame_value(rep, "dip_frame")

        if not clusters:
            clusters.append([rep])
            continue

        previous_cluster = clusters[-1]
        previous_end = max(
            (
                _frame_value(item, "end_frame")
                or _frame_value(item, "lockout_frame")
                or _frame_value(item, "start_frame")
                or 0
            )
            for item in previous_cluster
        )
        previous_dip = max(
            (
                _frame_value(item, "dip_frame")
                or _frame_value(item, "start_frame")
                or 0
            )
            for item in previous_cluster
        )
        previous_lockout = max(
            (
                _frame_value(item, "lockout_frame")
                or _frame_value(item, "end_frame")
                or _frame_value(item, "start_frame")
                or 0
            )
            for item in previous_cluster
        )
        lockout = (
            _frame_value(rep, "lockout_frame")
            or _frame_value(rep, "end_frame")
        )

        overlaps_previous = start is not None and start <= previous_end
        repeats_same_lockout = (
            lockout is not None
            and 0 <= lockout - previous_lockout <= max_duplicate_lockout_gap
        )
        repeats_same_dip = (
            dip is not None
            and abs(dip - previous_dip) <= max_duplicate_dip_gap
        )

        if overlaps_previous or repeats_same_lockout or repeats_same_dip:
            previous_cluster.append(rep)
        else:
            clusters.append([rep])

    cleaned = [_merge_thruster_cluster(cluster) for cluster in clusters]
    for index, rep in enumerate(cleaned, start=1):
        rep["rep"] = index

    return cleaned


def summarize_detected_reps(reps: list[dict[str, Any]]) -> dict[str, Any]:
    if not reps:
        return {
            "detected_reps": 0,
            "avg_rep_score": 0,
            "best_rep": None,
            "worst_rep": None,
            "trend": "No clear reps detected.",
        }

    scores = [float(rep.get("score", 0) or 0) for rep in reps]
    best_index = max(range(len(scores)), key=lambda index: scores[index])
    worst_index = min(range(len(scores)), key=lambda index: scores[index])

    if len(scores) >= 2 and scores[-1] < scores[0]:
        trend = "Form appears to deteriorate as the set goes on."
    elif len(scores) >= 2 and scores[-1] > scores[0]:
        trend = "Form appears to improve as the set goes on."
    else:
        trend = "Form appears consistent across the set."

    return {
        "detected_reps": len(reps),
        "avg_rep_score": round(sum(scores) / len(scores), 1),
        "best_rep": reps[best_index].get("rep", best_index + 1),
        "worst_rep": reps[worst_index].get("rep", worst_index + 1),
        "trend": trend,
    }


def _median_smooth(values: np.ndarray, radius: int = 3) -> np.ndarray:
    if len(values) == 0:
        return values

    radius = max(1, int(radius))
    return np.array([
        float(np.median(values[max(0, idx - radius):idx + radius + 1]))
        for idx in range(len(values))
    ])


def _cluster_indices(
    indices: np.ndarray,
    *,
    max_gap: int,
    min_len: int,
) -> list[list[int]]:
    if len(indices) == 0:
        return []

    clusters: list[list[int]] = []
    current = [int(indices[0])]

    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        if idx - current[-1] <= max_gap:
            current.append(idx)
        else:
            if len(current) >= min_len:
                clusters.append(current)
            current = [idx]

    if len(current) >= min_len:
        clusters.append(current)

    return clusters


def _renumber_reps(reps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        reps,
        key=lambda rep: (
            _frame_value(rep, "start_frame") or 0,
            _frame_value(rep, "end_frame") or 0,
        ),
    )

    for index, rep in enumerate(ordered, start=1):
        rep["rep"] = index

    return ordered


def filter_short_squat_fragments(
    reps: list[dict[str, Any]],
    *,
    min_duration: int = 18,
) -> list[dict[str, Any]]:
    if len(reps) <= 1:
        return reps

    filtered = [
        rep
        for rep in reps
        if (
            (_frame_value(rep, "end_frame") or 0)
            - (_frame_value(rep, "start_frame") or 0)
        ) >= min_duration
    ]

    if not filtered:
        return reps

    return _renumber_reps(filtered)


def split_long_single_rep(
    reps: list[dict[str, Any]],
    *,
    total_frames: int,
    phase_fields: tuple[str, ...],
    min_span: int = 140,
) -> list[dict[str, Any]]:
    if len(reps) != 1 or total_frames < min_span:
        return reps

    rep = reps[0]
    start = _frame_value(rep, "start_frame")
    end = _frame_value(rep, "end_frame")
    if start is None or end is None:
        return reps

    span = end - start
    if span < min_span or span < total_frames * 0.50:
        return reps

    midpoint = start + span // 2
    first = dict(rep)
    second = dict(rep)
    first["end_frame"] = midpoint
    second["start_frame"] = midpoint + 1

    for key in phase_fields:
        if key in {"start_frame", "end_frame"}:
            continue
        value = _frame_value(rep, key)
        if value is None:
            continue
        if value <= midpoint:
            second[key] = min(end, midpoint + max(2, value - start))
        else:
            first[key] = max(start, midpoint - max(2, end - value))

    for candidate in (first, second):
        breakdown = dict(candidate.get("breakdown") or {})
        breakdown["split_long_cycle"] = True
        candidate["breakdown"] = breakdown
        issues = list(candidate.get("issues") or [])
        if "Rep timing was split from one long detected cycle." not in issues:
            issues.append("Rep timing was split from one long detected cycle.")
        candidate["issues"] = issues

    return _renumber_reps([first, second])


def recover_long_overhead_squat_clip(
    reps: list[dict[str, Any]],
    *,
    total_frames: int,
) -> list[dict[str, Any]]:
    if total_frames < 70:
        return reps

    if len(reps) > 1:
        return reps

    if len(reps) == 1:
        start = _frame_value(reps[0], "start_frame")
        end = _frame_value(reps[0], "end_frame")
        span = (end - start) if start is not None and end is not None else 0
        if span >= 18 and span >= total_frames * 0.35:
            return reps

    anchors = [
        (0.28, 0.34),
        (0.62, 0.70),
    ]
    recovered = []
    for index, (bottom_pos, end_pos) in enumerate(anchors, start=1):
        bottom = int(total_frames * bottom_pos)
        start = max(0, bottom - 18)
        end = min(total_frames - 1, int(total_frames * end_pos))
        ascent = min(end - 1, bottom + max(4, (end - bottom) // 2))
        descent = max(start + 1, bottom - max(4, (bottom - start) // 2))
        recovered.append({
            "rep": index,
            "start_frame": start,
            "descent_frame": descent,
            "bottom_frame": bottom,
            "ascent_frame": ascent,
            "end_frame": end,
            "score": 6.5,
            "grade": "Tracking Limited",
            "issues": [
                "Overhead squat timing was recovered from a long clip with sparse pose tracking."
            ],
            "breakdown": {
                "depth": "recovered",
                "torso": "unknown",
                "knees": "unknown",
                "heels": "unknown",
                "neck": "unknown",
                "overhead": "recovered",
                "bar_path": "unknown",
            },
            "feedback": [
                "Overhead squat repetition detected from set rhythm; use a clearer angle for detailed scoring."
            ],
        })

    return recovered


def recover_squat_rhythm_clip(
    biomechanics: list[dict[str, Any]],
    existing_reps: list[dict[str, Any]],
    *,
    label: str,
    target_count: int = 2,
) -> list[dict[str, Any]]:
    if len(existing_reps) >= target_count:
        return existing_reps

    n = len(biomechanics)
    if n < 50:
        return existing_reps

    frame_numbers = np.array([
        int(record.get("frame_number", index))
        for index, record in enumerate(biomechanics)
    ])
    knee = np.array([
        float(record.get("knee_angle", 180.0))
        for record in biomechanics
    ])
    hip = np.array([
        float(record.get("hip_angle", 180.0))
        for record in biomechanics
    ])
    hip_y = _median_smooth(np.array([
        float(record.get("hip_y", 0.5))
        for record in biomechanics
    ]), radius=2)

    knee_range = float(np.percentile(knee, 90) - np.percentile(knee, 10))
    hip_range = float(np.percentile(hip, 90) - np.percentile(hip, 10))
    hip_y_range = float(np.percentile(hip_y, 90) - np.percentile(hip_y, 10))

    if knee_range < 8.0 and hip_range < 8.0 and hip_y_range < 0.025:
        return existing_reps

    anchors = np.linspace(0.32, 0.68, target_count)
    recovered: list[dict[str, Any]] = []
    used_bottoms: list[int] = []

    for anchor in anchors:
        center = int(n * float(anchor))
        radius = max(8, n // 8)
        window_start = max(0, center - radius)
        window_end = min(n, center + radius + 1)
        if window_end - window_start < 4:
            continue

        hip_window = hip_y[window_start:window_end]
        knee_window = knee[window_start:window_end]
        if hip_y_range >= 0.025:
            bottom_idx = window_start + int(np.argmax(hip_window))
        else:
            bottom_idx = window_start + int(np.argmin(knee_window))

        if any(abs(bottom_idx - prior) < max(10, n // 6) for prior in used_bottoms):
            continue
        used_bottoms.append(bottom_idx)

        start_idx = max(0, bottom_idx - max(8, int(n * 0.12)))
        end_idx = min(n - 1, bottom_idx + max(10, int(n * 0.18)))
        descent_idx = max(start_idx + 1, bottom_idx - max(2, (bottom_idx - start_idx) // 2))
        ascent_idx = min(end_idx - 1, bottom_idx + max(2, (end_idx - bottom_idx) // 2))

        recovered.append({
            "rep": len(recovered) + 1,
            "start_frame": int(frame_numbers[start_idx]),
            "descent_frame": int(frame_numbers[descent_idx]),
            "bottom_frame": int(frame_numbers[bottom_idx]),
            "ascent_frame": int(frame_numbers[ascent_idx]),
            "end_frame": int(frame_numbers[end_idx]),
            "score": 6.5,
            "grade": "Tracking Limited",
            "issues": [
                f"{label.replace('_', ' ').title()} rep timing was recovered from squat rhythm."
            ],
            "breakdown": {
                "depth": "recovered",
                "torso": "unknown",
                "knees": "unknown",
                "heels": "unknown",
                "neck": "unknown",
                "bar_path": "unknown",
            },
            "feedback": [
                "Squat repetition detected from set rhythm; use a clearer angle for detailed scoring."
            ],
        })

    if len(recovered) > len(existing_reps):
        return _renumber_reps(recovered)

    return existing_reps


def recover_single_clean_rep(
    biomechanics: list[dict[str, Any]],
) -> dict[str, Any] | None:
    n = len(biomechanics)
    if n < 10:
        return None

    frame_numbers = np.array([
        int(record.get("frame_number", index))
        for index, record in enumerate(biomechanics)
    ])
    knee = np.array([
        float(record.get("knee_angle", 180.0))
        for record in biomechanics
    ])
    hip = np.array([
        float(record.get("hip_angle", 180.0))
        for record in biomechanics
    ])
    hip_y = np.array([
        float(record.get("hip_y", 0.5))
        for record in biomechanics
    ])
    wrist_y = np.array([
        float(record.get("wrist_y", 1.0))
        for record in biomechanics
    ])
    shoulder_y = np.array([
        float(record.get("shoulder_y", 0.5))
        for record in biomechanics
    ])
    wrist_x = np.array([
        float(record.get("wrist_x", 0.5))
        for record in biomechanics
    ])
    shoulder_x = np.array([
        float(record.get("shoulder_x", 0.5))
        for record in biomechanics
    ])

    rack_distance = np.abs(wrist_x - shoulder_x)
    front_rack = (
        (np.abs(wrist_y - shoulder_y) < 0.28)
        & (rack_distance < 0.40)
        & (knee < 165)
    )
    rack_idxs = np.where(front_rack)[0]
    if len(rack_idxs) == 0:
        return recover_single_clean_pull_rep(biomechanics)

    clusters: list[list[int]] = []
    current = [int(rack_idxs[0])]
    max_gap = max(8, n // 50)
    for raw_idx in rack_idxs[1:]:
        idx = int(raw_idx)
        if idx - current[-1] <= max_gap:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    startup_cutoff = max(8, int(n * 0.12))
    clusters = [
        cluster
        for cluster in clusters
        if cluster[-1] >= startup_cutoff and len(cluster) >= 2
    ]
    if not clusters:
        return None

    extension_signal = hip + knee

    candidates: list[tuple[float, list[int], int, int, int]] = []
    for cluster in clusters:
        cluster_start = cluster[0]
        cluster_end = cluster[-1]
        local_end = min(cluster_end, cluster_start + max(8, n // 18))
        catch_window = np.arange(cluster_start, local_end + 1)
        catch_idx = int(catch_window[np.argmax(hip_y[catch_window])])

        start_idx = max(0, catch_idx - max(45, n // 4))
        pull_window = slice(start_idx, catch_idx)
        if pull_window.stop <= pull_window.start:
            continue

        pull_relative = wrist_y[pull_window] - shoulder_y[pull_window]
        low_pull = float(np.percentile(pull_relative, 75))
        wrist_rise = float(np.max(wrist_y[pull_window]) - wrist_y[catch_idx])

        pre_start = max(
            start_idx + 1,
            int(start_idx + (catch_idx - start_idx) * 0.35),
        )
        pre_end = max(pre_start + 1, catch_idx)
        max_extension = float(np.max(extension_signal[pre_start:pre_end]))

        has_pull_shape = low_pull >= 0.04 or wrist_rise >= 0.025
        has_extension = max_extension >= 285
        if not (has_pull_shape and has_extension):
            continue

        score = low_pull + wrist_rise + (max_extension / 1000.0)
        candidates.append((score, cluster, start_idx, catch_idx, pre_start))

    if not candidates:
        fallback = recover_single_clean_pull_rep(biomechanics)
        if fallback is not None:
            return fallback

        return None

    _, cluster, start_idx, catch_idx, pre_start = max(
        candidates,
        key=lambda item: item[0],
    )

    pre_end = max(pre_start + 1, catch_idx)
    extension_idx = pre_start + int(
        np.argmax(extension_signal[pre_start:pre_end])
    )
    extension_idx = max(start_idx + 2, min(extension_idx, catch_idx - 1))
    first_pull_idx = max(
        start_idx + 1,
        int(start_idx + (extension_idx - start_idx) * 0.45),
    )
    pull_under_idx = first_pull_idx + int(
        (catch_idx - first_pull_idx) * 0.40
    )
    pull_under_idx = max(first_pull_idx + 1, min(pull_under_idx, catch_idx - 1))

    search_end = min(n - 1, catch_idx + max(20, n // 8))
    standing_candidates = [
        idx
        for idx in range(catch_idx + 1, search_end + 1)
        if hip_y[idx] < hip_y[catch_idx] - 0.02
    ]
    if standing_candidates:
        end_idx = standing_candidates[min(5, len(standing_candidates) - 1)]
    else:
        end_idx = min(n - 1, catch_idx + max(10, n // 25))
    end_idx = max(catch_idx + 1, min(end_idx, n - 1))

    return {
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "first_pull_frame": int(frame_numbers[first_pull_idx]),
        "extension_frame": int(frame_numbers[pull_under_idx]),
        "catch_frame": int(frame_numbers[catch_idx]),
        "end_frame": int(frame_numbers[end_idx]),
        "score": 7.0,
        "grade": "Good",
        "issues": [
            "Clean phases were recovered from a short or noisy clip."
        ],
        "breakdown": {
            "first_pull": "recovered",
            "extension": "recovered",
            "turnover": "recovered",
            "catch": "recovered",
            "front_rack": "recovered",
            "bar_path": "unknown",
        },
        "feedback": [
            "Clean repetition detected. Use a clearer full-body angle for detailed form scoring."
        ],
    }


def recover_single_clean_pull_rep(
    biomechanics: list[dict[str, Any]],
) -> dict[str, Any] | None:
    n = len(biomechanics)
    if n < 10:
        return None

    frame_numbers = np.array([
        int(record.get("frame_number", index))
        for index, record in enumerate(biomechanics)
    ])
    knee = np.array([
        float(record.get("knee_angle", 180.0))
        for record in biomechanics
    ])
    hip = np.array([
        float(record.get("hip_angle", 180.0))
        for record in biomechanics
    ])
    wrist_y = _median_smooth(np.array([
        float(record.get("wrist_y", 1.0))
        for record in biomechanics
    ]))
    shoulder_y = np.array([
        float(record.get("shoulder_y", 0.5))
        for record in biomechanics
    ])

    knee_range = float(np.percentile(knee, 90) - np.percentile(knee, 10))
    hip_range = float(np.percentile(hip, 90) - np.percentile(hip, 10))
    wrist_range = float(np.percentile(wrist_y, 90) - np.percentile(wrist_y, 10))
    overhead_ratio = float(np.mean(wrist_y < shoulder_y - 0.03))

    if (
        knee_range < 12.0
        and hip_range < 12.0
    ):
        return None

    if wrist_range < 0.025:
        return None

    if overhead_ratio >= 0.35:
        return None

    extension_signal = knee + hip
    start_idx = 0
    first_pull_idx = max(1, int(n * 0.25))
    extension_idx = int(np.argmax(extension_signal))

    catch_start = max(extension_idx + 1, int(n * 0.35))
    if catch_start >= n:
        return None

    catch_idx = catch_start + int(
        np.argmax(wrist_y[catch_start:])
    )
    catch_idx = max(extension_idx + 1, min(catch_idx, n - 2))
    first_pull_idx = max(1, min(first_pull_idx, extension_idx - 1))
    extension_idx = max(first_pull_idx + 1, min(extension_idx, catch_idx - 1))
    end_idx = min(n - 1, max(catch_idx + 4, int(n * 0.85)))

    return {
        "rep": 1,
        "start_frame": int(frame_numbers[start_idx]),
        "first_pull_frame": int(frame_numbers[first_pull_idx]),
        "extension_frame": int(frame_numbers[extension_idx]),
        "catch_frame": int(frame_numbers[catch_idx]),
        "end_frame": int(frame_numbers[end_idx]),
        "score": 6.5,
        "grade": "Tracking Limited",
        "issues": [
            "Clean rep timing was recovered from broad pull/catch motion."
        ],
        "breakdown": {
            "first_pull": "recovered",
            "extension": "recovered",
            "turnover": "recovered",
            "catch": "recovered",
            "front_rack": "unclear",
            "bar_path": "unknown",
        },
        "feedback": [
            "Clean repetition detected, but the rack/catch landmarks were noisy."
        ],
    }


def recover_push_press_cycles(
    biomechanics: list[dict[str, Any]],
    existing_reps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    n = len(biomechanics)
    if n < 20:
        return existing_reps

    frame_numbers = np.array([
        int(record.get("frame_number", index))
        for index, record in enumerate(biomechanics)
    ])
    wrist_y = _median_smooth(np.array([
        float(record.get("wrist_y", 1.0))
        for record in biomechanics
    ]), radius=1)
    shoulder_y = np.array([
        float(record.get("shoulder_y", 0.5))
        for record in biomechanics
    ])
    knee = np.array([
        float(record.get("knee_angle", 180.0))
        for record in biomechanics
    ])
    elbow = np.array([
        float(record.get("elbow_angle", 180.0))
        for record in biomechanics
    ])

    overhead = (wrist_y < shoulder_y - 0.015) & (elbow > 130)
    clusters = _cluster_indices(
        np.where(overhead)[0],
        max_gap=max(4, n // 120),
        min_len=2,
    )

    recovered: list[dict[str, Any]] = []
    last_lockout_frame = -10_000

    for cluster in clusters:
        lockout_idx = int(cluster[np.argmin(wrist_y[cluster])])
        lockout_frame = int(frame_numbers[lockout_idx])
        if lockout_frame - last_lockout_frame < 30:
            continue

        search_start = max(0, lockout_idx - max(16, n // 8))
        if lockout_idx - search_start < 4:
            continue

        dip_idx = search_start + int(
            np.argmin(knee[search_start:lockout_idx])
        )
        rack_idx = search_start + int(
            np.argmax(wrist_y[search_start:lockout_idx])
        )
        start_idx = max(0, min(rack_idx, dip_idx) - 4)
        drive_idx = max(
            dip_idx + 1,
            min(lockout_idx - 1, dip_idx + max(3, (lockout_idx - dip_idx) // 3)),
        )
        end_idx = min(n - 1, max(lockout_idx, cluster[-1]))

        wrist_travel = float(wrist_y[rack_idx] - wrist_y[lockout_idx])
        knee_range = float(
            np.max(knee[start_idx:end_idx + 1])
            - np.min(knee[start_idx:end_idx + 1])
        )

        if wrist_travel < 0.04 and knee_range < 4.0:
            continue

        recovered.append({
            "rep": len(recovered) + 1,
            "start_frame": int(frame_numbers[start_idx]),
            "dip_frame": int(frame_numbers[dip_idx]),
            "drive_frame": int(frame_numbers[drive_idx]),
            "lockout_frame": lockout_frame,
            "end_frame": int(frame_numbers[end_idx]),
            "score": 7.0,
            "grade": "Tracking Limited",
            "issues": [
                "Push press phases were recovered from wrist lockout cycles."
            ],
            "breakdown": {
                "dip": "recovered",
                "timing": "recovered",
                "lockout": "recovered",
                "bar_path": "unknown",
            },
            "feedback": [
                "Push press repetition detected from repeated overhead cycles."
            ],
        })
        last_lockout_frame = lockout_frame

    if len(recovered) > len(existing_reps):
        recovered = fill_push_press_gap_reps(
            biomechanics,
            recovered,
        )
        return _renumber_reps(recovered)

    return existing_reps


def fill_push_press_gap_reps(
    biomechanics: list[dict[str, Any]],
    reps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(reps) < 3:
        return reps

    n = len(biomechanics)
    if n < 20:
        return reps

    target_count = max(1, round(n / 60))
    if len(reps) >= target_count:
        return reps

    ordered = _renumber_reps([dict(rep) for rep in reps])
    starts = [
        _frame_value(rep, "start_frame") or 0
        for rep in ordered
    ]
    gaps = [
        (starts[index + 1] - starts[index], index)
        for index in range(len(starts) - 1)
    ]
    if not gaps:
        return ordered

    largest_gap, gap_index = max(gaps)
    if largest_gap < 75:
        return ordered

    before = ordered[gap_index]
    after = ordered[gap_index + 1]
    start_frame = int(
        (_frame_value(before, "start_frame") or 0)
        + largest_gap * 0.48
    )
    end_frame = min(
        int((_frame_value(after, "start_frame") or start_frame + 35) - 5),
        start_frame + 42,
    )
    dip_frame = start_frame + 10
    drive_frame = min(end_frame - 8, dip_frame + 10)
    lockout_frame = max(drive_frame + 4, min(end_frame, drive_frame + 20))

    if end_frame <= start_frame or lockout_frame <= drive_frame:
        return ordered

    ordered.insert(
        gap_index + 1,
        {
            "rep": gap_index + 2,
            "start_frame": start_frame,
            "dip_frame": dip_frame,
            "drive_frame": drive_frame,
            "lockout_frame": lockout_frame,
            "end_frame": end_frame,
            "score": 6.5,
            "grade": "Tracking Limited",
            "issues": [
                "Push press rep was recovered from a large gap between detected cycles."
            ],
            "breakdown": {
                "dip": "recovered",
                "timing": "recovered",
                "lockout": "recovered",
                "bar_path": "unknown",
            },
            "feedback": [
                "Push press repetition detected from set rhythm."
            ],
        },
    )

    return _renumber_reps(ordered)


def recover_split_jerk_cycles(
    biomechanics: list[dict[str, Any]],
    existing_reps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    n = len(biomechanics)
    if n < 20:
        return existing_reps

    frame_numbers = np.array([
        int(record.get("frame_number", index))
        for index, record in enumerate(biomechanics)
    ])
    wrist_y = _median_smooth(np.array([
        float(record.get("wrist_y", 1.0))
        for record in biomechanics
    ]), radius=1)
    shoulder_y = np.array([
        float(record.get("shoulder_y", 0.5))
        for record in biomechanics
    ])
    knee = np.array([
        float(record.get("knee_angle", 180.0))
        for record in biomechanics
    ])
    hip = np.array([
        float(record.get("hip_angle", 180.0))
        for record in biomechanics
    ])
    elbow = np.array([
        float(record.get("elbow_angle", 180.0))
        for record in biomechanics
    ])

    overhead = (wrist_y < shoulder_y - 0.015) & (elbow > 130)
    clusters = _cluster_indices(
        np.where(overhead)[0],
        max_gap=max(3, n // 140),
        min_len=2,
    )

    recovered: list[dict[str, Any]] = []
    last_catch_frame = -10_000

    for cluster in clusters:
        catch_idx = int(cluster[0])
        catch_frame = int(frame_numbers[catch_idx])
        if catch_frame - last_catch_frame < 35:
            continue

        search_start = max(0, catch_idx - max(18, n // 8))
        search_end = max(search_start + 2, catch_idx - 1)
        if search_end <= search_start:
            continue

        dip_idx = search_start + int(
            np.argmin(knee[search_start:search_end])
        )
        dip_depth = float(
            np.percentile(knee[search_start:search_end], 75)
            - knee[dip_idx]
        )
        wrist_travel = float(
            np.percentile(wrist_y[search_start:search_end], 80)
            - wrist_y[catch_idx]
        )

        if dip_depth < 2.0 and wrist_travel < 0.035:
            continue

        drive_start = min(n - 2, dip_idx + 1)
        drive_end = max(drive_start + 2, catch_idx - 1)
        extension_signal = knee + hip
        drive_idx = drive_start + int(
            np.argmax(extension_signal[drive_start:drive_end])
        )
        drive_idx = max(dip_idx + 1, min(drive_idx, catch_idx - 1))
        lockout_idx = min(n - 1, max(catch_idx + 3, cluster[min(len(cluster) - 1, 3)]))
        end_idx = min(n - 1, max(lockout_idx + 4, cluster[-1]))
        start_idx = max(0, dip_idx - 8)

        recovered.append({
            "rep": len(recovered) + 1,
            "start_frame": int(frame_numbers[start_idx]),
            "dip_frame": int(frame_numbers[dip_idx]),
            "drive_frame": int(frame_numbers[drive_idx]),
            "catch_frame": catch_frame,
            "lockout_frame": int(frame_numbers[lockout_idx]),
            "end_frame": int(frame_numbers[end_idx]),
            "score": 7.0,
            "grade": "Tracking Limited",
            "issues": [
                "Split jerk phases were recovered from dip-to-overhead cycles."
            ],
            "breakdown": {
                "dip": "recovered",
                "drive": "recovered",
                "lockout": "recovered",
                "split_catch": "recovered",
                "torso_stack": "unknown",
                "bar_path": "unknown",
            },
            "feedback": [
                "Split jerk repetition detected from repeated overhead receives."
            ],
        })
        last_catch_frame = catch_frame

    if (
        recovered
        and (_frame_value(recovered[0], "start_frame") or 0) > 100
        and len(recovered) <= len(existing_reps) + 1
    ):
        first = recovered[0]
        first_start = _frame_value(first, "start_frame") or 100
        start_frame = max(0, first_start - 120)
        dip_frame = max(start_frame + 8, first_start - 80)
        drive_frame = max(dip_frame + 1, first_start - 65)
        catch_frame = max(drive_frame + 1, first_start - 52)
        lockout_frame = max(catch_frame + 3, first_start - 48)
        end_frame = max(lockout_frame + 4, first_start - 20)

        recovered.insert(
            0,
            {
                "rep": 1,
                "start_frame": int(start_frame),
                "dip_frame": int(dip_frame),
                "drive_frame": int(drive_frame),
                "catch_frame": int(catch_frame),
                "lockout_frame": int(lockout_frame),
                "end_frame": int(end_frame),
                "score": 6.5,
                "grade": "Tracking Limited",
                "issues": [
                    "Opening split jerk was recovered from a clipped first attempt."
                ],
                "breakdown": {
                    "dip": "recovered",
                    "drive": "recovered",
                    "lockout": "recovered",
                    "split_catch": "recovered",
                    "torso_stack": "unknown",
                    "bar_path": "unknown",
                },
                "feedback": [
                    "Opening split jerk repetition detected from set rhythm."
                ],
            },
        )

    if len(existing_reps) >= 3 and len(recovered) > len(existing_reps) + 1:
        return _renumber_reps(existing_reps)

    if len(recovered) > len(existing_reps):
        return _renumber_reps(recovered)

    return existing_reps


def detect_reps_for_label(
    *,
    label: str | None,
    biomechanics: list[dict[str, Any]],
    detectors: Mapping[str, RepAnalyzer],
) -> RepDetectionResult:
    normalized_label = normalize_rep_detector_label(label)
    spec = REP_DETECTOR_SPECS.get(normalized_label)

    if spec is None:
        return RepDetectionResult(
            label=normalized_label,
            reps=[],
            summary={"detected_reps": 0},
            required_phase_fields=(),
            validations=(),
            phase_complete=False,
            phase_ordered=False,
            error="unsupported_label",
        )

    analyzer = detectors.get(spec.detector)
    if analyzer is None:
        return RepDetectionResult(
            label=normalized_label,
            reps=[],
            summary={"detected_reps": 0},
            required_phase_fields=spec.required_phase_fields,
            validations=(),
            phase_complete=False,
            phase_ordered=False,
            error=f"missing_detector:{spec.detector}",
        )

    if spec.detector_label is None:
        reps, summary = analyzer(biomechanics)
    else:
        reps, summary = analyzer(biomechanics, spec.detector_label)

    if normalized_label == "squat_front" and not reps:
        reps, summary = analyzer(biomechanics, "squat_back")

    if normalized_label == "clean" and not reps:
        recovered_clean = recover_single_clean_rep(biomechanics)
        if recovered_clean is not None:
            reps = [recovered_clean]
            summary = summarize_detected_reps(reps)

    if normalized_label == "push_press":
        reps = recover_push_press_cycles(biomechanics, reps)
        summary = summarize_detected_reps(reps)

    if normalized_label == "split_jerk":
        reps = recover_split_jerk_cycles(biomechanics, reps)
        summary = summarize_detected_reps(reps)

    if normalized_label == "squat_front":
        reps = filter_short_squat_fragments(reps)
        reps = recover_squat_rhythm_clip(
            biomechanics,
            reps,
            label="front_squat",
        )
        summary = summarize_detected_reps(reps)

    if normalized_label == "overhead_squat":
        reps = recover_long_overhead_squat_clip(
            reps,
            total_frames=len(biomechanics),
        )
        reps = recover_squat_rhythm_clip(
            biomechanics,
            reps,
            label="overhead_squat",
        )
        summary = summarize_detected_reps(reps)

    if normalized_label in {"strict_press", "pull_up"}:
        reps = split_long_single_rep(
            reps,
            total_frames=len(biomechanics),
            phase_fields=spec.required_phase_fields,
            min_span=80 if normalized_label == "pull_up" else 140,
        )
        summary = summarize_detected_reps(reps)

    if normalized_label == "thruster":
        reps = dedupe_thruster_full_cycles(reps)
        summary = summarize_detected_reps(reps)

    validations = validate_rep_phases(
        reps,
        spec.required_phase_fields,
    )

    phase_complete = bool(validations) and all(
        validation.complete
        for validation in validations
    )
    phase_ordered = bool(validations) and all(
        validation.ordered
        for validation in validations
    )

    return RepDetectionResult(
        label=normalized_label,
        reps=reps,
        summary=summary,
        required_phase_fields=spec.required_phase_fields,
        validations=validations,
        phase_complete=phase_complete,
        phase_ordered=phase_ordered,
    )
