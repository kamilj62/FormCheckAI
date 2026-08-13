from __future__ import annotations

from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return int(default)


def _debug_float(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    return _float((data or {}).get(key, default), default)


def _wrist_to_shoulder_range_ratio(
    bodyweight_debug: dict[str, Any],
) -> float:
    return _debug_float(bodyweight_debug, "wrist_y_range") / max(
        _debug_float(bodyweight_debug, "shoulder_y_range"),
        0.001,
    )


def should_recover_controlled_push_press(
    *,
    forced_exercise_label: str | None,
    final_label: str | None,
    bio_label: str | None,
    bio_conf: float,
    squat_label: str | None,
    squat_conf: float,
    olympic_pred: str | None,
    olympic_conf: float,
    looks_cj: bool,
    looks_split: bool,
    explosive_score: float,
    bodyweight_debug: dict[str, Any],
) -> bool:
    return (
        not forced_exercise_label
        and final_label == "clean_and_jerk"
        and bio_label == "push_press"
        and _float(bio_conf) >= 0.75
        and squat_label == "squat_back"
        and _float(squat_conf) >= 0.90
        and olympic_pred == "clean_and_jerk"
        and _float(olympic_conf) >= 0.85
        and not bool(looks_cj)
        and not bool(looks_split)
        and _float(explosive_score) < 15.0
        and _debug_float(bodyweight_debug, "shoulder_y_range", 999.0) <= 0.10
        and _debug_float(bodyweight_debug, "hip_y_range", 999.0) <= 0.09
        and _debug_float(bodyweight_debug, "wrist_above_shoulder_ratio") >= 0.85
        and _int((bodyweight_debug or {}).get("total_frames")) >= 250
    )


def should_recover_strict_press(
    *,
    forced_exercise_label: str | None,
    final_label: str | None,
    raw_label: str | None,
    bio_label: str | None,
    looks_strict: bool,
    looks_split: bool,
    looks_thruster: bool,
    explosive_score: float,
    squat_knee_range: float,
    squat_hip_range: float,
    bodyweight_debug: dict[str, Any] | None = None,
) -> bool:
    bodyweight_debug = bodyweight_debug or {}

    # Original strict-press recovery for clips already recognized as
    # push_press by the broad classifiers.
    legacy_push_press_shape = (
        final_label == "push_press"
        and raw_label == "push_press"
        and bio_label == "push_press"
        and bool(looks_strict)
        and _float(squat_knee_range) < 20.0
        and _float(squat_hip_range) < 12.0
    )

    # Some strict presses are projected as squat_front while the
    # biomechanics classifier still sees a press. Use actual body
    # translation to distinguish them from genuine squats:
    # wrists travel substantially while the hips remain nearly fixed.
    controlled_standing_press_shape = (
        final_label in {"squat_front", "push_press"}
        and raw_label in {"squat_front", "push_press"}
        and bio_label == "push_press"
        and _debug_float(bodyweight_debug, "hip_y_range", 999.0) <= 0.06
        and _debug_float(bodyweight_debug, "wrist_y_range", 0.0) >= 0.20
    )

    return (
        not forced_exercise_label
        and (legacy_push_press_shape or controlled_standing_press_shape)
        and not bool(looks_split)
        and not bool(looks_thruster)
        and _float(explosive_score) < 15.0
    )


def should_recover_push_press_over_weak_cj_split(
    *,
    forced_exercise_label: str | None,
    final_label: str | None,
    raw_label: str | None,
    base_conf: float,
    bio_label: str | None,
    bio_conf: float,
    router_v6_label: str | None,
    router_v6_conf: float,
    olympic_pred: str | None,
    olympic_conf: float,
    looks_cj: bool,
    explosive_score: float,
    bodyweight_debug: dict[str, Any],
) -> bool:
    return (
        not forced_exercise_label
        and final_label == "split_jerk"
        and raw_label == "push_press"
        and _float(base_conf) >= 0.70
        and bio_label == "push_press"
        and _float(bio_conf) >= 0.75
        and router_v6_label == "push_press"
        and _float(router_v6_conf) >= 0.70
        and olympic_pred == "clean_and_jerk"
        and _float(olympic_conf) < 0.65
        and not bool(looks_cj)
        and _float(explosive_score) >= 70.0
        and _wrist_to_shoulder_range_ratio(bodyweight_debug) >= 3.0
    )


def should_recover_push_press_over_back_squat(
    *,
    forced_exercise_label: str | None,
    final_label: str | None,
    raw_label: str | None,
    base_conf: float,
    bio_label: str | None,
    bio_conf: float,
    squat_label: str | None,
    squat_conf: float,
    olympic_pred: str | None,
    olympic_conf: float,
    looks_cj: bool,
    explosive_score: float,
    bodyweight_debug: dict[str, Any],
) -> str | None:
    low_motion = (
        not forced_exercise_label
        and final_label == "squat_back"
        and raw_label == "squat"
        and _float(base_conf) <= 0.75
        and bio_label == "squat"
        and _float(bio_conf) <= 0.75
        and squat_label == "squat_back"
        and _float(squat_conf) >= 0.90
        and olympic_pred == "split_jerk"
        and _float(olympic_conf) >= 0.80
        and not bool(looks_cj)
        and _float(explosive_score) < 15.0
        and _debug_float(bodyweight_debug, "shoulder_y_range", 1.0) <= 0.15
        and _debug_float(bodyweight_debug, "hip_y_range", 1.0) <= 0.15
        and _wrist_to_shoulder_range_ratio(bodyweight_debug) >= 2.20
    )

    if low_motion:
        return "low_motion_push_press_over_back_squat"

    explosive = (
        not forced_exercise_label
        and final_label == "squat_back"
        and raw_label == "squat"
        and _float(base_conf) <= 0.70
        and bio_label == "push_press"
        and _float(bio_conf) >= 0.75
        and squat_label == "squat_back"
        and _float(squat_conf) >= 0.95
        and _float(olympic_conf) < 0.50
        and not bool(looks_cj)
        and _float(explosive_score) >= 100.0
        and _wrist_to_shoulder_range_ratio(bodyweight_debug) >= 2.20
    )

    if explosive:
        return "explosive_push_press_over_back_squat"

    return None


def should_recover_explosive_push_press_authority(
    *,
    forced_exercise_label: str | None,
    final_label: str | None,
    raw_label: str | None,
    base_conf: float,
    bio_label: str | None,
    bio_conf: float,
    squat_label: str | None,
    squat_conf: float,
    olympic_conf: float,
    explosive_score: float,
    bar_debug: dict[str, Any],
) -> bool:
    return (
        not forced_exercise_label
        and final_label == "overhead_squat"
        and raw_label == "push_press"
        and _float(base_conf) >= 0.99
        and bio_label == "push_press"
        and _float(bio_conf) >= 0.99
        and squat_label == "overhead_squat"
        and 0.78 <= _float(squat_conf) <= 0.85
        and _float(explosive_score) >= 100.0
        and _float(olympic_conf) < 0.75
        and _debug_float(bar_debug, "overhead_ratio") >= 0.95
    )
