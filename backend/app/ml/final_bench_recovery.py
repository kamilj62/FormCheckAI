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


def should_recover_short_bench_over_pushup(
    *,
    forced_exercise_label: str | None,
    final_label: str | None,
    raw_label: str | None,
    base_conf: float,
    bio_label: str | None,
    bio_conf: float,
    squat_label: str | None,
    squat_conf: float,
    router_v6_label: str | None,
    router_v6_conf: float,
    olympic_pred: str | None,
    olympic_conf: float,
    bodyweight_debug: dict[str, Any],
) -> bool:
    """Recover short bench clips that look like floor push-ups to pose geometry."""
    return (
        not forced_exercise_label
        and final_label == "push_up"
        and raw_label == "squat_front"
        and _float(base_conf) >= 0.65
        and bio_label == "deadlift"
        and 0.82 <= _float(bio_conf) <= 1.0
        and squat_label == "squat_front"
        and _float(squat_conf) >= 0.96
        and router_v6_label == "squat_front"
        and _float(router_v6_conf) >= 0.93
        and olympic_pred == "clean_and_jerk"
        and 0.66 <= _float(olympic_conf) <= 0.72
        and 65.0 <= _debug_float(bodyweight_debug, "avg_torso_angle") <= 80.0
        and 0.15 <= _debug_float(bodyweight_debug, "wrist_y_range") <= 0.23
        and _debug_float(bodyweight_debug, "shoulder_y_range", 1.0) <= 0.07
        and _debug_float(bodyweight_debug, "hip_y_range", 1.0) <= 0.07
        and _debug_float(bodyweight_debug, "elbow_range") >= 160.0
        and _debug_float(bodyweight_debug, "avg_elbow") >= 130.0
        and _debug_float(bodyweight_debug, "min_elbow", 999.0) <= 10.0
        and 0.08
        <= _debug_float(bodyweight_debug, "wrist_above_shoulder_ratio")
        <= 0.22
        and 30 <= _int((bodyweight_debug or {}).get("total_frames")) <= 60
    )
