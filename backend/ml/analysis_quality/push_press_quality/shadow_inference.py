from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
MODELS_DIR = MODULE_DIR / "results" / "models"

MODEL_FILES = {
    "elbow_error": "elbow_error_candidate.joblib",
    "knee_error": "knee_error_candidate.joblib",
}

SERIES_NAMES = (
    "left_elbow_angle",
    "right_elbow_angle",
    "elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "knee_angle",
    "wrist_y",
    "wrist_x",
    "shoulder_y",
    "hip_y",
    "torso_lean",
    "wrist_above_shoulder",
    "wrist_shoulder_offset_x",
)

ELBOW_VISIBILITY_KEYS = (
    "visibility_left_shoulder",
    "visibility_right_shoulder",
    "visibility_left_elbow",
    "visibility_right_elbow",
    "visibility_left_wrist",
    "visibility_right_wrist",
    "visibility_left_hip",
    "visibility_right_hip",
)

KNEE_VISIBILITY_KEYS = (
    "visibility_left_shoulder",
    "visibility_right_shoulder",
    "visibility_left_hip",
    "visibility_right_hip",
    "visibility_left_knee",
    "visibility_right_knee",
    "visibility_left_ankle",
    "visibility_right_ankle",
)


def safe_float(value, default=0.0):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def summarize(values, prefix):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_robust_range": 0.0,
            f"{prefix}_start": 0.0,
            f"{prefix}_end": 0.0,
            f"{prefix}_delta": 0.0,
        }

    p10, p25, p50, p75, p90 = np.percentile(
        array,
        [10, 25, 50, 75, 90],
    )

    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_range": float(np.max(array) - np.min(array)),
        f"{prefix}_p10": float(p10),
        f"{prefix}_p25": float(p25),
        f"{prefix}_median": float(p50),
        f"{prefix}_p75": float(p75),
        f"{prefix}_p90": float(p90),
        f"{prefix}_iqr": float(p75 - p25),
        f"{prefix}_robust_range": float(p90 - p10),
        f"{prefix}_start": float(array[0]),
        f"{prefix}_end": float(array[-1]),
        f"{prefix}_delta": float(array[-1] - array[0]),
    }


@lru_cache(maxsize=2)
def load_candidate(target):
    if target not in MODEL_FILES:
        raise ValueError(f"Unsupported target: {target}")

    path = MODELS_DIR / MODEL_FILES[target]

    if not path.exists():
        raise FileNotFoundError(f"Candidate model not found: {path}")

    package = joblib.load(path)

    required = {
        "target",
        "model",
        "feature_columns",
        "threshold",
        "model_version",
    }

    missing = required - set(package)

    if missing:
        raise ValueError(
            f"{path.name} missing package keys: {sorted(missing)}"
        )

    return package


def build_live_feature_row(biomechanics_window, target):
    if target not in MODEL_FILES:
        raise ValueError(f"Unsupported target: {target}")

    if not biomechanics_window:
        raise ValueError("Biomechanics window is empty")

    rows = sorted(
        biomechanics_window,
        key=lambda item: int(item.get("frame_number", 0)),
    )

    series = {name: [] for name in SERIES_NAMES}

    for row in rows:
        left_elbow = safe_float(row.get("left_elbow_angle"))
        right_elbow = safe_float(row.get("right_elbow_angle"))
        left_knee = safe_float(row.get("left_knee_angle"))
        right_knee = safe_float(row.get("right_knee_angle"))

        wrist_y = safe_float(row.get("wrist_y"))
        wrist_x = safe_float(row.get("wrist_x"))
        shoulder_y = safe_float(row.get("shoulder_y"))
        shoulder_x = safe_float(row.get("shoulder_x"))
        hip_y = safe_float(row.get("hip_y"))

        series["left_elbow_angle"].append(left_elbow)
        series["right_elbow_angle"].append(right_elbow)
        series["elbow_angle"].append(
            (left_elbow + right_elbow) / 2.0
        )

        series["left_knee_angle"].append(left_knee)
        series["right_knee_angle"].append(right_knee)
        series["knee_angle"].append(
            (left_knee + right_knee) / 2.0
        )

        series["wrist_y"].append(wrist_y)
        series["wrist_x"].append(wrist_x)
        series["shoulder_y"].append(shoulder_y)
        series["hip_y"].append(hip_y)
        series["torso_lean"].append(
            safe_float(row.get("torso_lean"))
        )
        series["wrist_above_shoulder"].append(
            shoulder_y - wrist_y
        )
        series["wrist_shoulder_offset_x"].append(
            abs(wrist_x - shoulder_x)
        )

    frame_numbers = np.asarray(
        [
            int(row.get("frame_number", index))
            for index, row in enumerate(rows)
        ],
        dtype=int,
    )

    if len(frame_numbers) >= 2:
        positive_steps = np.diff(frame_numbers)
        positive_steps = positive_steps[positive_steps > 0]
        frame_step = (
            int(round(float(np.median(positive_steps))))
            if positive_steps.size
            else 1
        )
    else:
        frame_step = 1

    processed_frames = (
        int(round(
            (frame_numbers[-1] - frame_numbers[0])
            / max(frame_step, 1)
        ))
        + 1
    )

    visibility_keys = (
        ELBOW_VISIBILITY_KEYS
        if target == "elbow_error"
        else KNEE_VISIBILITY_KEYS
    )

    visibility_values = [
        safe_float(row.get(key))
        for row in rows
        for key in visibility_keys
    ]

    feature_row = {
        "processed_frames": processed_frames,
        "pose_frames": len(rows),
        "pose_coverage": len(rows) / max(processed_frames, 1),
        "mean_visibility": (
            float(np.mean(visibility_values))
            if visibility_values
            else 0.0
        ),
    }

    for name, values in series.items():
        feature_row.update(summarize(values, name))

    return feature_row


def score_feature_row(feature_row, target):
    package = load_candidate(target)
    columns = package["feature_columns"]

    model_input = pd.DataFrame(
        [
            {
                column: feature_row.get(column)
                for column in columns
            }
        ],
        columns=columns,
    )

    probability = float(
        package["model"].predict_proba(model_input)[0, 1]
    )
    threshold = float(package["threshold"])

    return {
        "target": target,
        "model_version": package["model_version"],
        "probability": probability,
        "threshold": threshold,
        "detected": bool(probability >= threshold),
        "feature_count": len(columns),
    }


def score_biomechanics_window(biomechanics_window):
    results = {}

    for target in MODEL_FILES:
        feature_row = build_live_feature_row(
            biomechanics_window,
            target,
        )
        results[target] = score_feature_row(
            feature_row,
            target,
        )

    return results


def fixed_pose_window(
    biomechanics,
    center_frame,
    analysis_fps,
    window_seconds=1.5,
    target_fps=10.0,
):
    """
    Build a fixed-duration pose window matching training extraction:
    approximately 1.5 seconds sampled at 10 FPS.
    """
    if not biomechanics:
        return []

    analysis_fps = max(safe_float(analysis_fps, 30.0), 1.0)
    target_fps = max(safe_float(target_fps, 10.0), 1.0)

    half_span = 0.5 * window_seconds * analysis_fps
    start_frame = float(center_frame) - half_span
    end_frame = float(center_frame) + half_span

    sample_count = max(
        4,
        int(round(window_seconds * target_fps)) + 1,
    )

    target_frames = np.linspace(
        start_frame,
        end_frame,
        sample_count,
    )

    ordered = sorted(
        biomechanics,
        key=lambda row: int(row.get("frame_number", 0)),
    )

    frame_numbers = np.asarray(
        [
            int(row.get("frame_number", index))
            for index, row in enumerate(ordered)
        ],
        dtype=float,
    )

    selected = []
    used_indices = set()

    for target_frame in target_frames:
        nearest_index = int(
            np.argmin(np.abs(frame_numbers - target_frame))
        )

        # Avoid duplicate pose rows when tracking is sparse.
        if nearest_index in used_indices:
            continue

        used_indices.add(nearest_index)
        selected.append(ordered[nearest_index])

    return selected


def score_push_press_rep(
    biomechanics,
    rep,
    analysis_fps,
):
    """
    Shadow-only scoring using target-specific 1.5-second windows.

    Knee window is centered on the dip.
    Elbow window is centered between drive and lockout.
    """
    start_frame = int(rep.get("start_frame", 0))
    end_frame = int(rep.get("end_frame", start_frame))

    dip_frame = int(
        rep.get(
            "dip_frame",
            start_frame + (end_frame - start_frame) * 0.35,
        )
    )
    drive_frame = int(
        rep.get(
            "drive_frame",
            start_frame + (end_frame - start_frame) * 0.55,
        )
    )
    lockout_frame = int(
        rep.get(
            "lockout_frame",
            end_frame,
        )
    )

    elbow_center = int(
        round((drive_frame + lockout_frame) / 2.0)
    )

    target_windows = {
        "elbow_error": fixed_pose_window(
            biomechanics,
            center_frame=elbow_center,
            analysis_fps=analysis_fps,
        ),
    }

    results = {}

    for target, window in target_windows.items():
        if len(window) < 4:
            results[target] = {
                "target": target,
                "available": False,
                "reason": "insufficient_pose_frames",
                "pose_frames": len(window),
            }
            continue

        feature_row = build_live_feature_row(
            window,
            target,
        )
        result = score_feature_row(
            feature_row,
            target,
        )

        result["available"] = True
        result["pose_frames"] = len(window)
        result["window_start_frame"] = int(
            window[0].get("frame_number", 0)
        )
        result["window_end_frame"] = int(
            window[-1].get("frame_number", 0)
        )

        results[target] = result

    return results
