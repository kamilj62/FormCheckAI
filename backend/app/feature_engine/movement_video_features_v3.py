from __future__ import annotations

import numpy as np


SIGNALS = [
    "knee_angle",
    "hip_angle",
    "elbow_angle",
    "torso_angle",
    "wrist_y",
    "shoulder_y",
    "hip_y",
    "wrist_shoulder_distance",
]

N_BINS = 10

GLOBAL_FEATURE_NAMES = [
    "frame_count",
    "overhead_ratio",
    "first_overhead_pct",
    "last_overhead_pct",
    "overhead_span_pct",
    "overhead_transition_count",
    "min_knee_time_pct",
    "min_hip_time_pct",
    "max_wrist_motion",
    "mean_wrist_motion",
    "max_hip_motion",
    "mean_hip_motion",
]

TEMPORAL_FEATURE_NAMES = [
    f"{signal}_bin_{bin_index:02d}"
    for signal in SIGNALS
    for bin_index in range(N_BINS)
]

OVERHEAD_BIN_NAMES = [
    f"overhead_ratio_bin_{bin_index:02d}"
    for bin_index in range(N_BINS)
]

FEATURE_NAMES = (
    GLOBAL_FEATURE_NAMES
    + TEMPORAL_FEATURE_NAMES
    + OVERHEAD_BIN_NAMES
)


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def resample_mean(values, n_bins=N_BINS):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return np.zeros(n_bins, dtype=np.float32)

    if len(values) == 1:
        return np.repeat(values[0], n_bins).astype(np.float32)

    edges = np.linspace(0, len(values), n_bins + 1)

    result = []

    for index in range(n_bins):
        start = int(np.floor(edges[index]))
        end = int(np.floor(edges[index + 1]))

        start = min(start, len(values) - 1)
        end = max(start + 1, end)
        end = min(end, len(values))

        result.append(float(np.mean(values[start:end])))

    return np.asarray(result, dtype=np.float32)


def build_movement_video_features_v3(biomechanics):
    if not biomechanics:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    n = len(biomechanics)

    signal_arrays = {}

    defaults = {
        "knee_angle": 180.0,
        "hip_angle": 180.0,
        "elbow_angle": 0.0,
        "torso_angle": 0.0,
        "wrist_y": 1.0,
        "shoulder_y": 0.0,
        "hip_y": 0.0,
        "wrist_shoulder_distance": 0.0,
    }

    for signal in SIGNALS:
        signal_arrays[signal] = np.asarray(
            [
                safe_float(
                    frame.get(signal, defaults[signal]),
                    defaults[signal],
                )
                for frame in biomechanics
            ],
            dtype=np.float32,
        )

    wrist_y = signal_arrays["wrist_y"]
    shoulder_y = signal_arrays["shoulder_y"]
    hip_y = signal_arrays["hip_y"]
    knee = signal_arrays["knee_angle"]
    hip = signal_arrays["hip_angle"]

    overhead = (wrist_y < shoulder_y).astype(np.float32)
    overhead_indices = np.where(overhead > 0.5)[0]

    if len(overhead_indices):
        first_overhead_pct = float(overhead_indices[0] / max(1, n - 1))
        last_overhead_pct = float(overhead_indices[-1] / max(1, n - 1))
        overhead_span_pct = float(
            (overhead_indices[-1] - overhead_indices[0] + 1)
            / max(1, n)
        )
    else:
        first_overhead_pct = 1.0
        last_overhead_pct = 0.0
        overhead_span_pct = 0.0

    overhead_transition_count = int(
        np.sum(np.abs(np.diff(overhead)) > 0.5)
    ) if n > 1 else 0

    wrist_motion = np.diff(wrist_y)
    hip_motion = np.diff(hip_y)

    global_features = [
        float(n),
        float(np.mean(overhead)),
        first_overhead_pct,
        last_overhead_pct,
        overhead_span_pct,
        float(overhead_transition_count),
        float(np.argmin(knee) / max(1, n - 1)),
        float(np.argmin(hip) / max(1, n - 1)),
        float(np.max(np.abs(wrist_motion))) if len(wrist_motion) else 0.0,
        float(np.mean(np.abs(wrist_motion))) if len(wrist_motion) else 0.0,
        float(np.max(np.abs(hip_motion))) if len(hip_motion) else 0.0,
        float(np.mean(np.abs(hip_motion))) if len(hip_motion) else 0.0,
    ]

    temporal_features = []

    for signal in SIGNALS:
        temporal_features.extend(
            resample_mean(signal_arrays[signal]).tolist()
        )

    overhead_bins = resample_mean(overhead).tolist()

    features = np.asarray(
        global_features
        + temporal_features
        + overhead_bins,
        dtype=np.float32,
    )

    if len(features) != len(FEATURE_NAMES):
        raise ValueError(
            f"V3 feature mismatch: got {len(features)}, "
            f"expected {len(FEATURE_NAMES)}"
        )

    if not np.isfinite(features).all():
        raise ValueError("V3 features contain NaN or infinity")

    return features


build_movement_video_features = build_movement_video_features_v3
