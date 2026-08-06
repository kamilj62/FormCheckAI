from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ml.analysis_quality.fitness_aqa_squat.build_knee_v5_features import (
    geometry,
)
from ml.analysis_quality.fitness_aqa_squat.build_knee_v9_rep_features import (
    build_feature_names,
    build_vector,
    phase_windows,
)


EXPECTED_FEATURE_COUNT = 709
MIN_REP_ROWS = 9
MIN_ROWS_EACH_SIDE = 4


def build_frame_rows(
    sequence: Sequence[Sequence[float]],
    biomechanics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Pair the aligned runtime 68-feature vectors and biomechanics rows.

    The caller must pass the original, untrimmed sequence returned by the
    video extraction function.
    """
    if len(sequence) != len(biomechanics):
        raise ValueError(
            "Forward-lean runtime alignment mismatch: "
            f"{len(sequence)} feature rows != "
            f"{len(biomechanics)} biomechanics rows"
        )

    rows: list[dict[str, Any]] = []

    for features, bio in zip(sequence, biomechanics):
        frame_number = bio.get("frame_number")

        if frame_number is None:
            raise ValueError(
                "Biomechanics row is missing frame_number"
            )

        feature_array = np.asarray(
            features,
            dtype=np.float32,
        )

        if feature_array.shape != (68,):
            raise ValueError(
                "Expected a 68-feature pose row, got "
                f"shape {feature_array.shape}"
            )

        rows.append({
            "frame_number": int(frame_number),
            "features": feature_array,
            "biomechanics": dict(bio),
        })

    rows.sort(
        key=lambda row: int(row["frame_number"])
    )

    return rows


def build_rep_feature_vector(
    frame_rows: Iterable[Mapping[str, Any]],
    start_frame: int,
    bottom_frame: int,
    end_frame: int,
) -> np.ndarray:
    """
    Build the exact 709-feature V9 rep vector used during training.
    """
    start_frame = int(start_frame)
    bottom_frame = int(bottom_frame)
    end_frame = int(end_frame)

    if not (
        start_frame <= bottom_frame <= end_frame
    ):
        raise ValueError(
            "Invalid rep frame order: "
            f"{start_frame} -> {bottom_frame} -> {end_frame}"
        )

    rep_rows = [
        dict(row)
        for row in frame_rows
        if (
            start_frame
            <= int(row["frame_number"])
            <= end_frame
        )
    ]

    rep_rows.sort(
        key=lambda row: int(row["frame_number"])
    )

    if len(rep_rows) < MIN_REP_ROWS:
        raise ValueError(
            "Insufficient pose rows for forward-lean inference: "
            f"{len(rep_rows)} < {MIN_REP_ROWS}"
        )

    bottom_position = min(
        range(len(rep_rows)),
        key=lambda index: abs(
            int(rep_rows[index]["frame_number"])
            - bottom_frame
        ),
    )

    rows_before_bottom = bottom_position
    rows_after_bottom = (
        len(rep_rows) - bottom_position - 1
    )

    if (
        rows_before_bottom < MIN_ROWS_EACH_SIDE
        or rows_after_bottom < MIN_ROWS_EACH_SIDE
    ):
        raise ValueError(
            "Insufficient rows around squat bottom: "
            f"before={rows_before_bottom}, "
            f"after={rows_after_bottom}"
        )

    geometry_rows = [
        geometry(row)
        for row in rep_rows
    ]

    windows = phase_windows(
        rep_rows,
        bottom_position,
    )

    vector = build_vector(
        rep_rows,
        windows,
        geometry_rows,
        bottom_position,
    )

    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    if vector.shape != (EXPECTED_FEATURE_COUNT,):
        raise ValueError(
            "Forward-lean feature count mismatch: "
            f"{vector.shape} != "
            f"({EXPECTED_FEATURE_COUNT},)"
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            "Forward-lean feature vector contains "
            "non-finite values"
        )

    expected_names = build_feature_names()

    if len(expected_names) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Forward-lean feature-name count mismatch: "
            f"{len(expected_names)} != "
            f"{EXPECTED_FEATURE_COUNT}"
        )

    return vector

_FORWARD_LEAN_MODEL = None
_FORWARD_LEAN_MODEL_ERROR = None
_FORWARD_LEAN_MODEL_ATTEMPTED = False

MODEL_FILENAME = (
    "forward_lean_manual_v9_batch09_extratrees.joblib"
)
MODEL_THRESHOLD = 0.50
MAX_SHOULDER_WIDTH = 0.60
MAX_HIP_WIDTH = 0.40


def load_forward_lean_model():
    """
    Lazily load the Batch 09 model.

    Model-loading failures remain diagnostic only and never interrupt
    the production analysis request.
    """
    global _FORWARD_LEAN_MODEL
    global _FORWARD_LEAN_MODEL_ERROR
    global _FORWARD_LEAN_MODEL_ATTEMPTED

    if _FORWARD_LEAN_MODEL_ATTEMPTED:
        return (
            _FORWARD_LEAN_MODEL,
            _FORWARD_LEAN_MODEL_ERROR,
        )

    _FORWARD_LEAN_MODEL_ATTEMPTED = True

    try:
        import joblib
        from pathlib import Path

        model_path = Path(__file__).resolve().parent / MODEL_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(
                f"Forward-lean model not found: {model_path}"
            )

        model = joblib.load(model_path)

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                "Forward-lean model does not provide predict_proba"
            )

        feature_count = getattr(
            model,
            "n_features_in_",
            None,
        )

        if int(feature_count or 0) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                "Forward-lean model feature count mismatch: "
                f"{feature_count} != {EXPECTED_FEATURE_COUNT}"
            )

        classes = [
            int(value)
            for value in np.asarray(
                getattr(model, "classes_", [])
            ).tolist()
        ]

        if 1 not in classes:
            raise ValueError(
                "Forward-lean model does not contain positive class 1"
            )

        _FORWARD_LEAN_MODEL = model

    except Exception as exc:
        _FORWARD_LEAN_MODEL = None
        _FORWARD_LEAN_MODEL_ERROR = (
            f"{type(exc).__name__}: {exc}"
        )

    return (
        _FORWARD_LEAN_MODEL,
        _FORWARD_LEAN_MODEL_ERROR,
    )


def evaluate_forward_lean_shadow(
    *,
    sequence,
    biomechanics,
    reps,
    exercise_label,
):
    """
    Evaluate excessive forward lean without changing production results.

    This function returns diagnostics only. It never modifies the supplied
    rep dictionaries and catches all model/feature errors internally.
    """
    result = {
        "mode": "shadow_only",
        "model": "batch09_extratrees",
        "threshold": MODEL_THRESHOLD,
        "exercise_label": exercise_label,
        "status": "not_applicable",
        "rep_count": len(reps or []),
        "eligible_rep_count": 0,
        "predicted_excessive_count": 0,
        "reps": [],
        "error": None,
    }

    if exercise_label != "squat_back":
        return result

    result["status"] = "running"

    try:
        frame_rows = build_frame_rows(
            sequence,
            biomechanics,
        )
    except Exception as exc:
        result["status"] = "feature_rows_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    model, model_error = load_forward_lean_model()

    if model is None:
        result["status"] = "model_unavailable"
        result["error"] = model_error
        return result

    classes = [
        int(value)
        for value in np.asarray(model.classes_).tolist()
    ]
    positive_index = classes.index(1)

    for position, rep in enumerate(reps or [], start=1):
        rep_number = int(rep.get("rep", position))

        rep_result = {
            "rep": rep_number,
            "status": "pending",
            "eligible": False,
            "prediction": None,
            "excessive_probability": None,
            "median_shoulder_width": None,
            "median_hip_width": None,
            "start_frame": rep.get("start_frame"),
            "bottom_frame": rep.get("bottom_frame"),
            "end_frame": rep.get("end_frame"),
            "error": None,
        }

        try:
            start_frame = int(rep["start_frame"])
            bottom_frame = int(rep["bottom_frame"])
            end_frame = int(rep["end_frame"])

            rep_biomechanics = [
                row
                for row in biomechanics
                if (
                    start_frame
                    <= int(row.get("frame_number", -1))
                    <= end_frame
                )
            ]

            if not rep_biomechanics:
                raise ValueError(
                    "No biomechanics rows found inside rep window"
                )

            shoulder_widths = np.asarray(
                [
                    float(row.get("shoulder_width_x", np.nan))
                    for row in rep_biomechanics
                ],
                dtype=np.float32,
            )
            hip_widths = np.asarray(
                [
                    float(row.get("hip_width_x", np.nan))
                    for row in rep_biomechanics
                ],
                dtype=np.float32,
            )

            shoulder_widths = shoulder_widths[
                np.isfinite(shoulder_widths)
            ]
            hip_widths = hip_widths[
                np.isfinite(hip_widths)
            ]

            if (
                shoulder_widths.size == 0
                or hip_widths.size == 0
            ):
                raise ValueError(
                    "Missing camera-view width measurements"
                )

            median_shoulder = float(
                np.median(shoulder_widths)
            )
            median_hip = float(
                np.median(hip_widths)
            )

            rep_result["median_shoulder_width"] = round(
                median_shoulder,
                6,
            )
            rep_result["median_hip_width"] = round(
                median_hip,
                6,
            )

            camera_eligible = (
                median_shoulder < MAX_SHOULDER_WIDTH
                and median_hip < MAX_HIP_WIDTH
            )

            rep_result["eligible"] = bool(camera_eligible)

            if not camera_eligible:
                rep_result["status"] = "ineligible_camera_view"
                result["reps"].append(rep_result)
                continue

            vector = build_rep_feature_vector(
                frame_rows=frame_rows,
                start_frame=start_frame,
                bottom_frame=bottom_frame,
                end_frame=end_frame,
            )

            probabilities = model.predict_proba(
                vector.reshape(1, -1)
            )[0]

            excessive_probability = float(
                probabilities[positive_index]
            )
            prediction = (
                "excessive"
                if excessive_probability >= MODEL_THRESHOLD
                else "clear"
            )

            rep_result["status"] = "predicted"
            rep_result["prediction"] = prediction
            rep_result["excessive_probability"] = round(
                excessive_probability,
                6,
            )

            result["eligible_rep_count"] += 1

            if prediction == "excessive":
                result["predicted_excessive_count"] += 1

        except Exception as exc:
            rep_result["status"] = "rep_error"
            rep_result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        result["reps"].append(rep_result)

    result["status"] = "completed"
    return result

