from __future__ import annotations

import numpy as np

from app.feature_engine.movement_video_features_v3 import (
    FEATURE_NAMES as V3_FEATURE_NAMES,
    build_movement_video_features_v3,
    safe_float,
)


EXTRA_FEATURE_NAMES = [
    "overhead_head_ratio",
    "overhead_tail_ratio",
    "max_overhead_run_pct",
    "overhead_run_count",
    "knee_at_first_overhead",
    "hip_at_first_overhead",
    "first_overhead_minus_min_knee_pct",
    "first_overhead_minus_min_hip_pct",
    "max_upward_wrist_velocity_pct",
    "max_upward_hip_velocity_pct",
    "wrist_hip_upward_peak_delay_pct",
    "pre_overhead_knee_range",
    "post_overhead_knee_range",
    "pre_overhead_hip_range",
    "post_overhead_hip_range",
    "deep_knee_dip_count",
    "deep_hip_dip_count",
]

FEATURE_NAMES = V3_FEATURE_NAMES + EXTRA_FEATURE_NAMES


def _range(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0

    return float(np.max(values) - np.min(values))


def _overhead_runs(overhead: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None

    for index, value in enumerate(overhead):
        active = bool(value > 0.5)

        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index - 1))
            start = None

    if start is not None:
        runs.append((start, len(overhead) - 1))

    return runs


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)

    if len(values) < 3:
        return values.copy()

    window = max(3, min(int(window), len(values)))

    if window % 2 == 0:
        window -= 1

    if window < 3:
        return values.copy()

    kernel = np.ones(window, dtype=np.float32) / float(window)

    return np.convolve(
        values,
        kernel,
        mode="same",
    ).astype(np.float32)


def _count_deep_dips(
    values: np.ndarray,
    threshold: float,
    min_separation_pct: float = 0.08,
) -> int:
    values = np.asarray(values, dtype=np.float32)

    if len(values) < 3:
        return 0

    smoothed = _smooth(values, window=5)

    candidates = [
        index
        for index in range(1, len(smoothed) - 1)
        if (
            smoothed[index] <= smoothed[index - 1]
            and smoothed[index] < smoothed[index + 1]
            and smoothed[index] < threshold
        )
    ]

    if not candidates:
        return 0

    minimum_separation = max(
        2,
        int(round(len(values) * min_separation_pct)),
    )

    selected: list[int] = []

    for index in sorted(
        candidates,
        key=lambda candidate: float(smoothed[candidate]),
    ):
        if all(
            abs(index - existing) >= minimum_separation
            for existing in selected
        ):
            selected.append(index)

    return int(len(selected))


def build_movement_video_features_v4(biomechanics):
    base_features = build_movement_video_features_v3(
        biomechanics
    )

    if not biomechanics:
        return np.concatenate(
            [
                base_features,
                np.zeros(
                    len(EXTRA_FEATURE_NAMES),
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    n = len(biomechanics)
    denominator = max(1, n - 1)

    wrist_y = np.asarray(
        [
            safe_float(frame.get("wrist_y"), 1.0)
            for frame in biomechanics
        ],
        dtype=np.float32,
    )
    shoulder_y = np.asarray(
        [
            safe_float(frame.get("shoulder_y"), 0.0)
            for frame in biomechanics
        ],
        dtype=np.float32,
    )
    hip_y = np.asarray(
        [
            safe_float(frame.get("hip_y"), 0.0)
            for frame in biomechanics
        ],
        dtype=np.float32,
    )
    knee = np.asarray(
        [
            safe_float(frame.get("knee_angle"), 180.0)
            for frame in biomechanics
        ],
        dtype=np.float32,
    )
    hip = np.asarray(
        [
            safe_float(frame.get("hip_angle"), 180.0)
            for frame in biomechanics
        ],
        dtype=np.float32,
    )

    overhead = (wrist_y < shoulder_y).astype(np.float32)
    overhead_indices = np.where(overhead > 0.5)[0]

    head_size = max(1, int(round(n * 0.20)))
    tail_size = max(1, int(round(n * 0.20)))

    overhead_head_ratio = float(
        np.mean(overhead[:head_size])
    )
    overhead_tail_ratio = float(
        np.mean(overhead[-tail_size:])
    )

    runs = _overhead_runs(overhead)

    max_overhead_run_pct = (
        max(
            end - start + 1
            for start, end in runs
        )
        / max(1, n)
        if runs
        else 0.0
    )

    overhead_run_count = float(len(runs))

    min_knee_index = int(np.argmin(knee))
    min_hip_index = int(np.argmin(hip))

    if len(overhead_indices):
        first_overhead_index = int(overhead_indices[0])

        knee_at_first_overhead = float(
            knee[first_overhead_index]
        )
        hip_at_first_overhead = float(
            hip[first_overhead_index]
        )

        first_overhead_pct = (
            first_overhead_index / denominator
        )

        pre_knee = knee[: first_overhead_index + 1]
        post_knee = knee[first_overhead_index:]
        pre_hip = hip[: first_overhead_index + 1]
        post_hip = hip[first_overhead_index:]
    else:
        first_overhead_index = n - 1
        first_overhead_pct = 1.0
        knee_at_first_overhead = 180.0
        hip_at_first_overhead = 180.0

        pre_knee = knee
        post_knee = np.asarray([], dtype=np.float32)
        pre_hip = hip
        post_hip = np.asarray([], dtype=np.float32)

    min_knee_pct = min_knee_index / denominator
    min_hip_pct = min_hip_index / denominator

    wrist_velocity = np.diff(wrist_y)
    hip_velocity = np.diff(hip_y)

    if len(wrist_velocity):
        wrist_upward_peak_index = int(
            np.argmin(wrist_velocity)
        )
        wrist_upward_peak_pct = (
            wrist_upward_peak_index
            / max(1, len(wrist_velocity) - 1)
        )
    else:
        wrist_upward_peak_pct = 0.0

    if len(hip_velocity):
        hip_upward_peak_index = int(
            np.argmin(hip_velocity)
        )
        hip_upward_peak_pct = (
            hip_upward_peak_index
            / max(1, len(hip_velocity) - 1)
        )
    else:
        hip_upward_peak_pct = 0.0

    extra_features = np.asarray(
        [
            overhead_head_ratio,
            overhead_tail_ratio,
            float(max_overhead_run_pct),
            overhead_run_count,
            knee_at_first_overhead,
            hip_at_first_overhead,
            float(first_overhead_pct - min_knee_pct),
            float(first_overhead_pct - min_hip_pct),
            float(wrist_upward_peak_pct),
            float(hip_upward_peak_pct),
            float(
                wrist_upward_peak_pct
                - hip_upward_peak_pct
            ),
            _range(pre_knee),
            _range(post_knee),
            _range(pre_hip),
            _range(post_hip),
            float(
                _count_deep_dips(
                    knee,
                    threshold=150.0,
                )
            ),
            float(
                _count_deep_dips(
                    hip,
                    threshold=145.0,
                )
            ),
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [base_features, extra_features]
    ).astype(np.float32)

    if len(features) != len(FEATURE_NAMES):
        raise ValueError(
            f"V4 feature mismatch: got {len(features)}, "
            f"expected {len(FEATURE_NAMES)}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "V4 features contain NaN or infinity"
        )

    return features


build_movement_video_features = (
    build_movement_video_features_v4
)
