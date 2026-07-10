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


def early_strength_protections(
    *,
    raw_label: str | None,
    base_conf: float,
    bio_conf: float,
    squat_label: str | None,
    bodyweight_debug: dict[str, Any],
    short_overhead_bench_setup: bool,
    pull_up_router_guard: bool,
    deadlift_low_speed_setup: bool,
    deadlift_upright_setup: bool,
    deadlift_raw_pull_setup: bool,
    looks_push_up: bool,
    looks_pull_up: bool,
    looks_handstand_push_up: bool,
) -> ProtectionResult:
    """
    Preserve the original V7 priority order for the first strength
    protection rules immediately following bodyweight protections.
    """

    if short_overhead_bench_setup and not pull_up_router_guard:
        return ProtectionResult(
            label="bench_press",
            confidence=max(
                float(base_conf or 0.0),
                float(bio_conf or 0.0),
                0.80,
            ),
            reason="bench_press_short_overhead_rescue",
        )

    if (
        deadlift_low_speed_setup
        or deadlift_upright_setup
        or deadlift_raw_pull_setup
    ):
        return ProtectionResult(
            label="deadlift",
            confidence=max(
                float(bio_conf or 0.0),
                float(base_conf or 0.0),
                0.82,
            ),
            reason="deadlift_setup_rescue",
        )

    trusted_base_bench = (
        raw_label == "bench_press"
        and not (
            looks_push_up
            or looks_pull_up
            or looks_handstand_push_up
        )
        and not (
            float(
                bodyweight_debug.get(
                    "wrist_above_shoulder_ratio",
                    1.0,
                )
            ) < 0.10
            and float(
                bodyweight_debug.get(
                    "mean_wrist_minus_shoulder_y",
                    0.0,
                )
            ) > 0.15
            and float(
                bodyweight_debug.get(
                    "median_head_drop",
                    0.0,
                )
            ) > 0.04
            and float(
                bodyweight_debug.get(
                    "wrist_y_range",
                    1.0,
                )
            ) < 0.12
        )
    )

    if trusted_base_bench:
        return ProtectionResult(
            label="bench_press",
            confidence=max(float(base_conf or 0.0), 0.80),
            reason="trusted_base_bench_press",
        )

    return ProtectionResult()


def apply_protections(
    *,
    bodyweight_inputs: dict[str, Any],
    early_strength_inputs: dict[str, Any],
) -> ProtectionResult:
    """
    Run extracted protection groups in the original V7 priority order.

    This function intentionally stops at the first matched protection.
    Remaining strength and Olympic protections still live in main.py.
    """

    result = bodyweight_protections(**bodyweight_inputs)
    if result.label:
        return result

    result = early_strength_protections(**early_strength_inputs)
    if result.label:
        return result

    return ProtectionResult()
