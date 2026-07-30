from dataclasses import dataclass
from typing import Any


@dataclass
class ProtectionResult:
    label: str | None = None
    confidence: float = 0.0
    reason: str | None = None


# ------------------------------------------------------------------
# Bodyweight protections
# ------------------------------------------------------------------

def bodyweight_protections(
    *,
    raw_label: str | None,
    squat_label: str | None,
    bodyweight_debug: dict[str, Any],
    bodyweight_router_label: str | None = None,
    bodyweight_router_conf: float = 0.0,
    strong_bench_evidence: bool = False,
    looks_push_up: bool = False,
    looks_pull_up: bool,
    looks_handstand_push_up: bool,
    looks_muscle_up: bool,
    looks_burpee: bool,
    credible_split_jerk: bool = False,
) -> ProtectionResult:
    """
    Preserve the original V7 priority order for bodyweight protections.
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
        and float(
            bodyweight_debug.get("wrist_above_shoulder_ratio", 0.0)
        ) >= 0.75
        and float(
            bodyweight_debug.get("mean_wrist_minus_shoulder_y", 1.0)
        ) <= -0.08
        and float(bodyweight_debug.get("elbow_range", 0.0)) >= 120.0
        and float(bodyweight_debug.get("min_elbow", 180.0)) <= 45.0
        and float(
            bodyweight_debug.get("avg_torso_angle", 180.0)
        ) <= 20.0
        and float(
            bodyweight_debug.get("avg_wrist_forward", 1.0)
        ) <= 0.02
        and float(
            bodyweight_debug.get("shoulder_y_range", 1.0)
        ) <= 0.36
        and float(bodyweight_debug.get("hip_y_range", 1.0)) <= 0.38
    )

    if (
        (looks_pull_up or push_press_pull_up_signature)
        and not strong_bench_evidence
        and not credible_split_jerk
    ):
        return ProtectionResult(
            label="pull_up",
            confidence=0.86,
            reason="pull_up_bodyweight_pattern",
        )

    specialized_hspu_router_guard = (
        bodyweight_router_label == "handstand_push_up"
        and float(bodyweight_router_conf or 0.0) >= 0.50
        and float(
            bodyweight_debug.get("avg_torso_angle", 0.0)
        ) >= 120.0
        and float(
            bodyweight_debug.get("min_elbow", 0.0)
        ) >= 90.0
        and float(
            bodyweight_debug.get("wrist_y_range", 1.0)
        ) <= 0.05
    )

    if looks_push_up and not specialized_hspu_router_guard:
        return ProtectionResult(
            label="push_up",
            confidence=0.86,
            reason="push_up_bodyweight_pattern",
        )

    if specialized_hspu_router_guard:
        return ProtectionResult(
            label="handstand_push_up",
            confidence=max(
                float(bodyweight_router_conf or 0.0),
                0.86,
            ),
            reason="handstand_push_up_router_geometry_guard",
        )

    return ProtectionResult()


# ------------------------------------------------------------------
# Early strength protections
# ------------------------------------------------------------------

def early_strength_protections(
    *,
    raw_label: str | None,
    base_conf: float,
    bio_conf: float,
    squat_label: str | None,
    olympic_conf: float,
    explosive_score: float,
    bar_debug: dict[str, Any],
    bodyweight_debug: dict[str, Any],
    short_overhead_bench_setup: bool,
    pull_up_router_guard: bool,
    deadlift_low_speed_setup: bool,
    deadlift_upright_setup: bool,
    deadlift_raw_pull_setup: bool,
    looks_push_up: bool,
    looks_pull_up: bool,
    looks_handstand_push_up: bool,
    looks_thruster: bool,
    looks_split: bool,
) -> ProtectionResult:
    """
    Preserve the original V7 priority order immediately following
    bodyweight protections.
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

    # This rule existed in main.py between the short-bench rescue
    # and deadlift rescue. Keep that exact priority.
    early_thruster = (
        looks_thruster
        and raw_label in {"bench_press", "push_press"}
        and float(explosive_score or 0.0) > 20.0
        and float(
            bar_debug.get("front_rack_elbow_p25", 180.0)
        ) <= 65.0
        and float(olympic_conf or 0.0) < 0.65
        and not looks_split
    )

    if early_thruster:
        return ProtectionResult(
            label="thruster",
            confidence=max(float(base_conf or 0.0), 0.76),
            reason="thruster_pattern_detected",
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


# ------------------------------------------------------------------
# Press / thruster protections
# ------------------------------------------------------------------

def strength_protections(
    *,
    raw_label: str | None,
    base_conf: float,
    bio_label: str | None,
    bio_conf: float,
    squat_label: str | None,
    squat_conf: float,
    explosive_score: float,
    looks_strict: bool,
    looks_thruster: bool,
    looks_clean_only: bool,
    looks_cj: bool,
    looks_split: bool,
) -> ProtectionResult:
    """
    Preserve the original V7 priority for strict press, push press,
    and the later thruster protection.
    """

    strict_press_pattern = (
        looks_strict
        and raw_label == "push_press"
        and not (
            bio_label == "push_press"
            and float(bio_conf or 0.0) >= 0.95
        )
        and not looks_split
    )

    if strict_press_pattern:
        return ProtectionResult(
            label="strict_press",
            confidence=max(float(base_conf or 0.0), 0.78),
            reason="strict_press_pattern_detected",
        )

    push_press_pattern = (
        raw_label == "push_press"
        and bio_label == "push_press"
        and not looks_clean_only
        and not looks_cj
        and not looks_split

        # Strong overhead-squat evidence plus low explosiveness is more
        # consistent with a controlled overhead squat than a push press.
        and not (
            squat_label == "overhead_squat"
            and float(squat_conf or 0.0) >= 0.80
            and float(explosive_score or 0.0) < 30.0
        )
    )

    if push_press_pattern:
        return ProtectionResult(
            label="push_press",
            confidence=max(
                float(base_conf or 0.0),
                float(bio_conf or 0.0),
                0.78,
            ),
            reason="push_press_pattern_detected",
        )

    later_thruster_pattern = (
        looks_thruster
        and raw_label == "push_press"
        and bio_label == "squat"
        and float(squat_conf or 0.0) >= 0.60
        and not looks_split
    )

    if later_thruster_pattern:
        return ProtectionResult(
            label="thruster",
            confidence=max(float(base_conf or 0.0), 0.76),
            reason="thruster_pattern_detected",
        )

    return ProtectionResult()


# ------------------------------------------------------------------
# Public orchestrator
# ------------------------------------------------------------------

def apply_protections(
    *,
    bodyweight_inputs: dict[str, Any],
    early_strength_inputs: dict[str, Any],
    strength_inputs: dict[str, Any],
) -> ProtectionResult:
    """
    Run extracted protection groups in the original V7 priority order.

    Remaining bench rescues and Olympic interactions still live in main.py.
    """

    result = bodyweight_protections(**bodyweight_inputs)
    if result.label:
        return result

    result = early_strength_protections(**early_strength_inputs)
    if result.label:
        return result

    result = strength_protections(**strength_inputs)
    if result.label:
        return result

    return ProtectionResult()
