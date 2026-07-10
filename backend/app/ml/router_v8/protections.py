from dataclasses import dataclass
from typing import Any


@dataclass
class ProtectionResult:
    label: str | None = None
    confidence: float = 0.0
    reason: str | None = None


def bodyweight_protections(
    *,
    raw_label: str | None,
    squat_label: str | None,
    bodyweight_debug: dict[str, Any],
    looks_push_up: bool,
    looks_pull_up: bool,
    looks_handstand_push_up: bool,
    looks_muscle_up: bool,
    looks_burpee: bool,
) -> ProtectionResult:
    """
    Preserve the original V7 priority order for the first five
    bodyweight protection rules.
    """

    if looks_muscle_up:
        return ProtectionResult(
            label="muscle_up",
            confidence=0.86,
            reason="muscle_up_bodyweight_pattern",
        )

    if looks_burpee:
        return ProtectionResult(
            label="burpee",
            confidence=0.84,
            reason="burpee_bodyweight_pattern",
        )

    if looks_handstand_push_up:
        return ProtectionResult(
            label="handstand_push_up",
            confidence=0.86,
            reason="handstand_push_up_bodyweight_pattern",
        )

    push_press_pull_up_signature = (
        raw_label == "push_press"
        and not (
            squat_label == "overhead_squat"
            and float(bodyweight_debug.get("wrist_y_range", 1.0)) > 0.20
        )
        and float(bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)) >= 0.75
        and float(bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)) <= -0.08
        and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
        and float(bodyweight_debug.get("min_elbow", 180.0)) <= 45.0
        and float(bodyweight_debug.get("avg_torso_angle", 180.0)) <= 20.0
        and float(bodyweight_debug.get("avg_wrist_forward", 1.0)) <= 0.02
        and float(bodyweight_debug.get("shoulder_y_range", 1.0)) <= 0.36
        and float(bodyweight_debug.get("hip_y_range", 1.0)) <= 0.38
    )

    if looks_pull_up or push_press_pull_up_signature:
        return ProtectionResult(
            label="pull_up",
            confidence=0.86,
            reason="pull_up_bodyweight_pattern",
        )

    if looks_push_up:
        return ProtectionResult(
            label="push_up",
            confidence=0.86,
            reason="push_up_bodyweight_pattern",
        )

    return ProtectionResult()
