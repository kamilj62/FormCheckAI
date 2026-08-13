from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.ml.final_bench_recovery import should_recover_short_bench_over_pushup
from app.ml.final_press_recovery import (
    should_recover_controlled_push_press,
    should_recover_explosive_push_press_authority,
    should_recover_push_press_over_back_squat,
    should_recover_push_press_over_weak_cj_split,
    should_recover_strict_press,
)


OLYMPIC_LABELS = {
    "snatch",
    "clean",
    "clean_and_jerk",
    "split_jerk",
}


@dataclass(frozen=True)
class FinalDecisionState:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None


@dataclass(frozen=True)
class FinalPreProbeArbitrationContext:
    state: FinalDecisionState
    forced_exercise_label: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    bodyweight_router_label: str | None
    bodyweight_router_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    run_oly_router: bool
    strong_oly_lock: bool
    strong_bench_evidence: bool
    credible_split_jerk: bool
    looks_cj: bool
    looks_split: bool
    looks_thruster: bool
    bodyweight_debug: dict[str, Any]
    router_v5_label: str | None
    router_v5_confidence: float
    router_v5_debug: dict[str, Any] | None


@dataclass(frozen=True)
class FinalPreProbeArbitrationDecision:
    state: FinalDecisionState
    pull_up_long_squat_barbell_collision: bool
    pull_up_long_overhead_barbell_collision: bool


@dataclass(frozen=True)
class FinalMidArbitrationContext:
    state: FinalDecisionState
    forced_exercise_label: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    bodyweight_router_label: str | None
    bodyweight_router_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    wrist_overhead_ratio: float
    looks_clean_only: bool
    looks_cj: bool
    looks_split: bool
    looks_strict: bool
    looks_thruster: bool
    looks_burpee: bool
    strong_front_squat_consensus: bool
    bench_model_consensus: bool
    pull_up_long_squat_barbell_collision: bool
    pull_up_long_overhead_barbell_collision: bool
    squat_knee_range: float
    squat_hip_range: float
    bodyweight_debug: dict[str, Any]
    router_v5_label: str | None
    router_v5_debug: dict[str, Any] | None


@dataclass(frozen=True)
class FinalMidArbitrationDecision:
    state: FinalDecisionState


@dataclass(frozen=True)
class FinalTailArbitrationContext:
    state: FinalDecisionState
    forced_exercise_label: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    wrist_overhead_ratio: float
    looks_cj: bool
    looks_split: bool
    bodyweight_debug: dict[str, Any]
    bar_debug: dict[str, Any]
    router_v5_debug: dict[str, Any] | None
    family_router_shadow: dict[str, Any] | None
    learned_family_shadow_label: str | None
    learned_family_shadow_confidence: float
    learned_family_shadow_trusted: bool
    deadlift_probe: Callable[[FinalDecisionState], FinalDecisionState]


@dataclass(frozen=True)
class FinalTailArbitrationDecision:
    state: FinalDecisionState


@dataclass(frozen=True)
class FinalArbitrationContext:
    state: FinalDecisionState
    forced_exercise_label: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    bodyweight_router_label: str | None
    bodyweight_router_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    wrist_overhead_ratio: float
    run_oly_router: bool
    strong_oly_lock: bool
    strong_bench_evidence: bool
    credible_split_jerk: bool
    looks_clean_only: bool
    looks_cj: bool
    looks_split: bool
    looks_strict: bool
    looks_thruster: bool
    looks_burpee: bool
    strong_front_squat_consensus: bool
    bench_model_consensus: bool
    squat_knee_range: float
    squat_hip_range: float
    bodyweight_debug: dict[str, Any]
    bar_debug: dict[str, Any]
    router_v5_label: str | None
    router_v5_confidence: float
    router_v5_debug: dict[str, Any] | None
    family_router_shadow: dict[str, Any] | None
    learned_family_shadow_label: str | None
    learned_family_shadow_confidence: float
    learned_family_shadow_trusted: bool
    push_press_probe: Callable[[FinalDecisionState], FinalDecisionState]
    yolo_deadlift_recovery: Callable[[FinalDecisionState], FinalDecisionState]
    deadlift_probe: Callable[[FinalDecisionState], FinalDecisionState]


@dataclass(frozen=True)
class FinalArbitrationDecision:
    state: FinalDecisionState
    pull_up_long_squat_barbell_collision: bool
    pull_up_long_overhead_barbell_collision: bool


@dataclass(frozen=True)
class ProtectedEvidenceContext:
    raw_label: str | None
    base_conf: float
    bio_label: str | None
    bio_conf: float
    bio_override: bool
    bio_reason: str | None
    squat_label: str | None
    squat_conf: float
    olympic_pred: str | None
    olympic_conf: float
    run_oly_router: bool
    explosive_score: float
    wrist_overhead_ratio: float
    router_v6_conf: float
    strong_bench_evidence: bool
    protection: Any
    looks_clean_only: bool
    looks_cj: bool
    looks_split: bool
    looks_strict: bool
    looks_thruster: bool
    looks_push_up: bool
    looks_pull_up: bool
    looks_handstand_push_up: bool
    truly_explosive: bool
    squat_confident: bool
    deadlift_setup_geometry: bool
    short_low_camera_bench_setup: bool
    bodyweight_debug: dict[str, Any]
    bar_debug: dict[str, Any]


@dataclass(frozen=True)
class ProtectedEvidenceDecision:
    label: str | None = None
    confidence: float = 0.0
    reason: str | None = None
    bench_model_consensus: bool = False


def final_state_from_decision(decision: Any) -> FinalDecisionState:
    """Convert any final-router decision shape into a common state object."""
    return FinalDecisionState(
        final_label=decision.final_label,
        final_confidence=_float(decision.final_confidence),
        analysis_mode=decision.analysis_mode,
        protected_label=decision.protected_label,
        protected_confidence=decision.protected_confidence,
        protected_reason=decision.protected_reason,
    )


@dataclass(frozen=True)
class EarlyFinalContext:
    protected_label: str | None
    protected_conf: float
    protected_reason: str | None
    strong_oly_lock: bool
    bodyweight_router_label: str | None
    bodyweight_router_conf: float
    raw_label: str | None
    base_conf: float
    bio_label: str | None
    bio_conf: float
    squat_label: str | None
    squat_conf: float
    olympic_pred: str | None
    olympic_conf: float
    run_oly_router: bool
    explosive_score: float
    wrist_overhead_ratio: float
    router_v6_label: str | None
    router_v6_conf: float
    pull_up_router_guard: bool
    looks_cj: bool
    looks_split: bool
    truly_explosive: bool
    strong_overhead: bool
    bodyweight_debug: dict[str, Any]


@dataclass(frozen=True)
class EarlyFinalDecision:
    label: str
    confidence: float
    mode: str
    protected_label: str | None = None
    protected_conf: float = 0.0
    protected_reason: str | None = None


@dataclass(frozen=True)
class FallbackFinalContext:
    raw_label: str | None
    base_conf: float
    bio_label: str | None
    bio_conf: float
    bio_override: bool
    squat_label: str | None
    squat_conf: float
    bar_conf: float
    olympic_pred: str | None
    olympic_conf: float
    run_oly_router: bool
    explosive_score: float
    wrist_overhead_ratio: float
    router_v6_label: str | None
    router_v6_conf: float
    squat_confident: bool
    truly_explosive: bool
    strong_overhead: bool
    bar_says_overhead_squat: bool
    has_real_squat_motion: bool
    push_press_should_hold: bool
    looks_clean_only: bool
    looks_cj: bool
    looks_split: bool
    looks_thruster: bool
    bodyweight_debug: dict[str, Any]


@dataclass(frozen=True)
class FallbackFinalDecision:
    label: str
    confidence: float
    mode: str


@dataclass(frozen=True)
class RouterV5AdjustmentContext:
    router_v5_label: str | None
    router_v5_conf: float
    router_v5_debug: dict[str, Any] | None
    raw_label: str | None
    base_conf: float
    bio_label: str | None
    bio_conf: float
    squat_label: str | None
    squat_conf: float
    olympic_pred: str | None
    olympic_conf: float
    explosive_score: float
    wrist_overhead_ratio: float
    looks_clean_only: bool
    looks_cj: bool
    looks_split: bool
    looks_thruster: bool
    truly_explosive: bool
    bodyweight_debug: dict[str, Any]


@dataclass(frozen=True)
class RouterV5Adjustment:
    label: str | None
    confidence: float
    debug: dict[str, Any] | None
    decision: str
    clean_rescue_active: bool
    upright_curl_signature: bool


@dataclass(frozen=True)
class RouterV5OverrideContext:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    router_v5_label: str | None
    router_v5_confidence: float
    router_v5_debug: dict[str, Any] | None
    router_v6_label: str | None
    router_v6_confidence: float
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    clean_rescue_active: bool
    upright_curl_signature: bool
    router_v8_cj_lock: bool
    clear_squat_should_hold: bool
    looks_clean_only: bool
    looks_cj: bool
    looks_split: bool
    looks_thruster: bool
    truly_explosive: bool
    bodyweight_debug: dict[str, Any]


@dataclass(frozen=True)
class RouterV5OverrideDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    router_v5_debug: dict[str, Any] | None
    snatch_rescue_from_overhead_squat: bool


@dataclass(frozen=True)
class BodyweightFinalArbitrationContext:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    router_v6_label: str | None
    router_v6_confidence: float
    bodyweight_router_label: str | None
    bodyweight_router_confidence: float
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    strong_oly_lock: bool
    strong_bench_evidence: bool
    credible_split_jerk: bool
    bodyweight_debug: dict[str, Any]


@dataclass(frozen=True)
class BodyweightFinalArbitrationDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    allowed: bool
    pull_up_long_squat_barbell_collision: bool
    pull_up_long_overhead_barbell_collision: bool


@dataclass(frozen=True)
class FinalShapeAuthorityContext:
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    run_oly_router: bool
    credible_split_jerk: bool
    looks_cj: bool
    looks_split: bool
    looks_thruster: bool
    bodyweight_debug: dict[str, Any]
    router_v5_label: str | None
    router_v5_confidence: float
    router_v5_debug: dict[str, Any] | None


@dataclass(frozen=True)
class FinalShapeAuthorityDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None


@dataclass(frozen=True)
class FinalPostProbeAuthorityContext:
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    wrist_overhead_ratio: float
    bodyweight_router_confidence: float
    looks_cj: bool
    looks_split: bool
    looks_thruster: bool


@dataclass(frozen=True)
class FinalPostProbeAuthorityDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None


@dataclass(frozen=True)
class FinalCleanBenchPushupContext:
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    bodyweight_router_label: str | None
    bodyweight_router_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    looks_clean_only: bool
    looks_cj: bool
    looks_split: bool
    looks_burpee: bool
    strong_front_squat_consensus: bool
    router_v5_label: str | None
    router_v5_debug: dict[str, Any] | None
    bodyweight_debug: dict[str, Any]


@dataclass(frozen=True)
class FinalCleanBenchPushupDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None


@dataclass(frozen=True)
class FinalPullupPushupAuthorityContext:
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    bio_label: str | None
    base_confidence: float
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    bodyweight_router_label: str | None
    bodyweight_router_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    bench_model_consensus: bool
    pull_up_long_squat_barbell_collision: bool
    pull_up_long_overhead_barbell_collision: bool
    looks_cj: bool
    looks_split: bool
    looks_strict: bool
    looks_thruster: bool
    bodyweight_debug: dict[str, Any]


@dataclass(frozen=True)
class FinalPullupPushupAuthorityDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None


@dataclass(frozen=True)
class FinalCollisionRecoveryContext:
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    bodyweight_router_label: str | None
    bodyweight_router_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    bodyweight_debug: dict[str, Any]


@dataclass(frozen=True)
class FinalCollisionRecoveryDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None


@dataclass(frozen=True)
class FinalPressSquatPreProbeContext:
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    looks_cj: bool
    looks_split: bool
    looks_strict: bool
    looks_thruster: bool
    squat_knee_range: float
    squat_hip_range: float
    bodyweight_debug: dict[str, Any]


@dataclass(frozen=True)
class FinalPressSquatPostProbeContext:
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    wrist_overhead_ratio: float
    looks_cj: bool
    bodyweight_debug: dict[str, Any]
    bar_debug: dict[str, Any]


@dataclass(frozen=True)
class FinalPressSquatAuthorityDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None


@dataclass(frozen=True)
class FinalPushPressProbeContext:
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    bodyweight_router_label: str | None
    bodyweight_router_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    bar_debug: dict[str, Any]


@dataclass(frozen=True)
class FinalPushPressProbePlan:
    should_probe: bool
    minimum_rep_count: int
    recovery_reason: str | None


@dataclass(frozen=True)
class FinalPushPressProbeDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None


@dataclass(frozen=True)
class FinalCleanOlympicAuthorityContext:
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    olympic_pred: str | None
    olympic_confidence: float
    explosive_score: float
    wrist_overhead_ratio: float
    looks_cj: bool
    looks_split: bool
    bodyweight_debug: dict[str, Any]
    router_v5_debug: dict[str, Any] | None
    family_router_shadow: dict[str, Any] | None
    learned_family_shadow_label: str | None
    learned_family_shadow_confidence: float
    learned_family_shadow_trusted: bool
    apply_segment_rules: bool = True
    apply_olympic_authority: bool = True


@dataclass(frozen=True)
class FinalCleanOlympicAuthorityDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None


@dataclass(frozen=True)
class FinalDeadliftProbeContext:
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    base_confidence: float
    bio_label: str | None
    bio_confidence: float
    squat_label: str | None
    squat_confidence: float
    router_v6_label: str | None
    router_v6_confidence: float
    wrist_overhead_ratio: float
    explosive_score: float
    deadlift_knee_range: float
    deadlift_hip_range: float
    deadlift_torso_range: float
    bodyweight_debug: dict[str, Any]
    bar_debug: dict[str, Any]


@dataclass(frozen=True)
class FinalDeadliftProbePlan:
    should_probe: bool
    recovery_reason: str | None


@dataclass(frozen=True)
class FinalDeadliftProbeDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None


@dataclass(frozen=True)
class YoloDeadliftRecoveryContext:
    use_yolo_tracking: bool
    forced_exercise_label: str | None
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    raw_label: str | None
    bio_label: str | None
    olympic_confidence: float


@dataclass(frozen=True)
class YoloDeadliftRecoveryPlan:
    should_probe: bool
    squat_probe_label: str | None


@dataclass(frozen=True)
class YoloDeadliftRecoveryDecision:
    final_label: str | None
    final_confidence: float
    analysis_mode: str
    protected_label: str | None
    protected_confidence: float | None
    protected_reason: str | None
    recovered: bool


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


def _push_press_pull_up_signature(debug: dict[str, Any]) -> bool:
    return (
        _debug_float(debug, "wrist_above_shoulder_ratio") >= 0.85
        and _debug_float(debug, "mean_wrist_minus_shoulder_y", 1.0)
        <= -0.10
        and _debug_float(debug, "elbow_range") >= 120.0
        and _debug_float(debug, "min_elbow", 180.0) <= 35.0
        and _debug_float(debug, "avg_torso_angle", 180.0) <= 8.0
        and _debug_float(debug, "avg_wrist_forward", 1.0) <= 0.02
    )


def _push_press_pull_up_signature_with_low_wrist_range(
    debug: dict[str, Any],
) -> bool:
    return (
        _push_press_pull_up_signature(debug)
        and _debug_float(debug, "wrist_y_range", 1.0) <= 0.15
    )


def _wrist_to_shoulder_range_ratio(debug: dict[str, Any]) -> float:
    return _debug_float(debug, "wrist_y_range") / max(
        _debug_float(debug, "shoulder_y_range"),
        0.001,
    )


def _false_pull_up_barbell_path_collision(
    *,
    pull_up_label: str | None,
    raw_label: str | None,
    bio_label: str | None,
    squat_label: str | None,
    olympic_pred: str | None,
    debug: dict[str, Any],
) -> bool:
    if pull_up_label != "pull_up":
        return False

    barbell_context = (
        raw_label in {"push_press", "squat", "squat_front"}
        or bio_label in {"push_press", "squat", "squat_front"}
        or squat_label in {"squat_back", "squat_front", "overhead_squat"}
        or olympic_pred in {"clean_and_jerk", "split_jerk", "snatch"}
    )
    if not barbell_context:
        return False

    wrist_y_range = _debug_float(debug, "wrist_y_range")
    shoulder_y_range = _debug_float(debug, "shoulder_y_range")
    if wrist_y_range <= 0.0 or shoulder_y_range <= 0.0:
        return False

    return (
        _int((debug or {}).get("total_frames")) >= 90
        and _debug_float(debug, "wrist_above_shoulder_ratio") >= 0.65
        and (
            _debug_float(debug, "mean_wrist_minus_shoulder_y", 1.0)
            > -0.12
            or _wrist_to_shoulder_range_ratio(debug) >= 0.75
        )
    )


def _set_debug_decision(
    debug: dict[str, Any] | None,
    decision: str,
) -> dict[str, Any] | None:
    if isinstance(debug, dict):
        debug = dict(debug)
        debug["decision"] = decision
    return debug


def adjust_router_v5_prediction(
    ctx: RouterV5AdjustmentContext,
) -> RouterV5Adjustment:
    """
    Apply existing Router V5 label/confidence/debug rescue adjustments.

    This preserves the original `main.py` order before production decides
    whether Router V5 can override the current final label.
    """
    label = ctx.router_v5_label
    confidence = _float(ctx.router_v5_conf)
    debug = dict(ctx.router_v5_debug) if isinstance(ctx.router_v5_debug, dict) else ctx.router_v5_debug

    split_features = (
        debug.get("split_features", {})
        if isinstance(debug, dict)
        else {}
    )
    likely_standalone_split = (
        ctx.raw_label == "push_press"
        and (
            _float(split_features.get("lockout_duration")) >= 200.0
            or _float(split_features.get("catch_to_finish")) >= 300.0
        )
    )

    if (
        likely_standalone_split
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_conf) >= 0.80
        and _float(ctx.olympic_conf) < 0.97
        and ctx.raw_label == "push_press"
        and ctx.bio_label == "push_press"
        and ctx.looks_split
    ):
        label = "split_jerk"
        confidence = max(confidence, 0.80)
        debug = _set_debug_decision(debug, "standalone_split_from_cj")

    if (
        label == "split_jerk"
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_conf) >= 0.90
        and not likely_standalone_split
    ):
        label = "clean_and_jerk"
        confidence = max(_float(ctx.olympic_conf), confidence)
        debug = _set_debug_decision(debug, "clean_and_jerk_high_conf_rescue")

    cj_events = (
        debug.get("events", {})
        if isinstance(debug, dict)
        else {}
    )
    cj_features = (
        debug.get("features", {})
        if isinstance(debug, dict)
        else {}
    )

    try:
        full_cj_event_sequence = (
            int(cj_events.get("clean_extension", -1))
            < int(cj_events.get("clean_catch", -1))
            <= int(cj_events.get("clean_recovery", -1))
            < int(cj_events.get("jerk_dip", -1))
            <= int(cj_events.get("jerk_drive", -1))
            <= int(cj_events.get("jerk_catch", -1))
            < int(cj_events.get("lockout", -1))
        )
    except (TypeError, ValueError):
        full_cj_event_sequence = False

    if (
        label == "clean"
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_conf) >= 0.50
        and full_cj_event_sequence
        and _float(cj_features.get("has_overhead")) >= 0.90
        and _float(cj_features.get("catch_overhead")) >= 0.90
        and _float(cj_features.get("lockout_duration")) >= 20.0
        and _float(ctx.explosive_score) >= 25.0
    ):
        label = "clean_and_jerk"
        confidence = max(confidence, 0.80)
        debug = _set_debug_decision(
            debug,
            "clean_and_jerk_full_sequence_rescue",
        )

    short_split_press = (
        _float(split_features.get("lockout_duration")) <= 90.0
        and _float(split_features.get("catch_to_finish")) <= 120.0
    )

    upright_curl_signature = (
        _debug_float(ctx.bodyweight_debug, "avg_torso_angle", 180.0) <= 15.0
        and _debug_float(ctx.bodyweight_debug, "elbow_range") >= 80.0
        and _debug_float(ctx.bodyweight_debug, "avg_wrist_forward") >= 0.08
        and _debug_float(ctx.bodyweight_debug, "mean_hip_minus_shoulder_y")
        >= 0.20
        and _debug_float(
            ctx.bodyweight_debug,
            "wrist_above_shoulder_ratio",
            1.0,
        )
        < 0.10
    )

    if (
        label == "split_jerk"
        and ctx.raw_label in {"squat", "deadlift"}
        and ctx.bio_label in {"push_press", "squat", "deadlift"}
        and short_split_press
        and _float(ctx.olympic_conf) < 0.70
        and not (
            ctx.raw_label == "squat"
            and ctx.bio_label == "squat"
            and _float(ctx.base_conf) >= 0.95
            and _float(ctx.bio_conf) >= 0.95
            and ctx.squat_label == "squat_back"
            and _float(ctx.squat_conf) >= 0.90
            and _float(ctx.explosive_score) >= 70.0
            and _float(ctx.wrist_overhead_ratio) < 0.20
            and _int(ctx.bodyweight_debug.get("total_frames", 999), 999)
            <= 60
        )
        and not ctx.looks_clean_only
        and not ctx.looks_cj
        and not upright_curl_signature
    ):
        label = "bench_press"
        confidence = max(_float(ctx.base_conf), _float(ctx.bio_conf), 0.80)
        debug = _set_debug_decision(debug, "bench_press_short_split_rescue")

    cj_features = (
        debug.get("features", {})
        if isinstance(debug, dict)
        else {}
    )
    short_cj_press = (
        _float(cj_features.get("lockout_duration")) <= 65.0
        and _float(cj_features.get("catch_to_finish")) <= 65.0
    )

    if (
        label == "clean_and_jerk"
        and ctx.raw_label in {"squat", "squat_front", "squat_back", "deadlift"}
        and ctx.bio_label in {"push_press", "squat", "deadlift"}
        and short_cj_press
        and _float(ctx.olympic_conf) < 0.90
        and not (
            ctx.raw_label == "squat"
            and ctx.bio_label == "push_press"
            and ctx.squat_label == "squat_front"
            and ctx.olympic_pred == "clean_and_jerk"
            and _float(ctx.olympic_conf) >= 0.73
            and _float(ctx.explosive_score) >= 45.0
        )
        and not ctx.looks_clean_only
        and not ctx.looks_cj
        and not upright_curl_signature
    ):
        label = "bench_press"
        confidence = max(_float(ctx.base_conf), _float(ctx.bio_conf), 0.80)
        debug = _set_debug_decision(debug, "bench_press_short_cj_rescue")

    if (
        label == "snatch"
        and ctx.looks_cj
        and not ctx.looks_clean_only
        and ctx.raw_label in {"squat", "squat_front", "squat_back"}
        and _float(ctx.olympic_conf) < 0.90
        and _float(ctx.wrist_overhead_ratio) < 0.45
        and _float(ctx.explosive_score) > 80.0
    ):
        label = "clean_and_jerk"
        confidence = max(confidence, 0.76)
        debug = _set_debug_decision(debug, "clean_and_jerk_shape_rescue")

    decision = (
        str(debug.get("decision", ""))
        if isinstance(debug, dict)
        else ""
    )

    clean_rescue_active = (
        label == "clean"
        and decision == "clean_rescue_from_weak_snatch"
        and ctx.truly_explosive
        and confidence >= 0.70

        # A verified thruster already explains the explosive squat-to-press
        # pattern. Do not reinterpret it as a clean merely because the
        # Olympic router sees a weak-snatch/clean signature.
        and not ctx.looks_thruster
    )

    return RouterV5Adjustment(
        label=label,
        confidence=confidence,
        debug=debug,
        decision=decision,
        clean_rescue_active=clean_rescue_active,
        upright_curl_signature=upright_curl_signature,
    )


def select_router_v5_override(
    ctx: RouterV5OverrideContext,
) -> RouterV5OverrideDecision:
    """Apply the post-Router V5 Olympic override without changing its order."""
    final_label = ctx.final_label
    final_confidence = _float(ctx.final_confidence)
    analysis_mode = ctx.analysis_mode
    protected_label = ctx.protected_label
    protected_confidence = ctx.protected_confidence
    protected_reason = ctx.protected_reason
    router_v5_debug = (
        dict(ctx.router_v5_debug)
        if isinstance(ctx.router_v5_debug, dict)
        else ctx.router_v5_debug
    )
    snatch_rescue_from_overhead_squat = False

    protected_non_olympic = protected_label in {
        "bench_press",
        "burpee",
        "deadlift",
        "handstand_push_up",
        "muscle_up",
        "push_up",
        "pull_up",
        "push_press",
        "thruster",
        "strict_press",
    }
    squat_should_hold = (
        final_label in {"squat_back", "squat_front", "overhead_squat"}
        and (
            _float(ctx.olympic_confidence) < 0.65
            or not ctx.truly_explosive
        )
        and not ctx.clean_rescue_active
    )

    push_press_should_hold = (
        (
            protected_label == "push_press"
            and ctx.raw_label == "push_press"
            and ctx.bio_label == "push_press"
            and ctx.router_v5_label == "clean_and_jerk"
        )
        or (
            ctx.raw_label == "push_press"
            and _float(ctx.base_confidence) >= 0.85
            and ctx.bio_label == "push_press"
            and _float(ctx.bio_confidence) >= 0.85
            and ctx.router_v6_label == "push_press"
            and _float(ctx.router_v6_confidence) >= 0.85
            and ctx.router_v5_label == "clean_and_jerk"
            and _float(ctx.olympic_confidence) < 0.60
            and not ctx.looks_cj
        )
    )

    vertical_pull_up_collision = (
        ctx.raw_label == "push_press"
        and ctx.router_v5_label in OLYMPIC_LABELS
        and _debug_float(ctx.bodyweight_debug, "wrist_above_shoulder_ratio")
        >= 0.85
        and _debug_float(
            ctx.bodyweight_debug,
            "mean_wrist_minus_shoulder_y",
            1.0,
        )
        <= -0.10
        and _debug_float(ctx.bodyweight_debug, "elbow_range") >= 120.0
        and _debug_float(ctx.bodyweight_debug, "min_elbow", 180.0) <= 35.0
        and _debug_float(ctx.bodyweight_debug, "avg_torso_angle", 180.0)
        <= 8.0
        and _debug_float(ctx.bodyweight_debug, "avg_wrist_forward", 1.0)
        <= 0.02
        and not ctx.truly_explosive
    )

    front_squat_push_press_guard = (
        ctx.squat_label == "squat_front"
        and _float(ctx.squat_confidence) >= 0.80
        and ctx.raw_label == "push_press"
        and _float(ctx.olympic_confidence) < 0.75
    )
    front_squat_weak_cj_guard = (
        ctx.squat_label == "squat_front"
        and _float(ctx.squat_confidence) >= 0.80
        and ctx.router_v5_label == "clean_and_jerk"
        and _float(ctx.router_v5_confidence) < 0.75
    )

    # If production routing has already resolved the movement as a
    # thruster and the full-body thruster geometry agrees, do not let a
    # generic Router V5 clean prediction steal it.
    #
    # Requiring final_label == "thruster" is intentional: looks_thruster
    # can also be true on very dynamic squat clips, so geometry alone is
    # not sufficient to veto Olympic routing.
    thruster_should_hold = (
        final_label == "thruster"
        and ctx.looks_thruster
        and ctx.router_v5_label == "clean"
    )

    should_apply_router_v5 = (
        not push_press_should_hold
        and not front_squat_push_press_guard
        and not front_squat_weak_cj_guard
        and not thruster_should_hold
        and not vertical_pull_up_collision
        and (
            final_label in OLYMPIC_LABELS
            or (
                ctx.router_v5_label in OLYMPIC_LABELS
                and not protected_non_olympic
                and not squat_should_hold
            )
        )
    )

    if should_apply_router_v5:
        strong_explosive_snatch = (
            (
                ctx.olympic_pred == "snatch"
                and _float(ctx.olympic_confidence) >= 0.80
                and ctx.truly_explosive
                and _float(ctx.explosive_score) >= 100.0
            )
            or (
                ctx.olympic_pred == "snatch"
                and _float(ctx.olympic_confidence) >= 0.60
                and ctx.raw_label == "squat"
                and ctx.bio_label == "push_press"
                and ctx.squat_label == "squat_back"
                and _float(ctx.squat_confidence) >= 0.90
                and _float(ctx.explosive_score) >= 50.0
                and _float(ctx.router_v6_confidence) < 0.75
            )
        )

        clear_squat_router_guard = (
            ctx.raw_label in {
                "squat",
                "squat_back",
                "squat_front",
                "overhead_squat",
            }
            and ctx.squat_label in {
                "squat_back",
                "squat_front",
                "overhead_squat",
            }
            and _float(ctx.squat_confidence) >= 0.90
            and ctx.router_v5_label in OLYMPIC_LABELS
            and _float(ctx.router_v5_confidence) < 0.85
            and not ctx.clean_rescue_active
            and not strong_explosive_snatch
        )

        if clear_squat_router_guard:
            final_label = ctx.squat_label
            final_confidence = max(
                _float(ctx.squat_confidence),
                (
                    _float(ctx.base_confidence)
                    if ctx.raw_label == ctx.squat_label
                    else 0.0
                ),
            )
            analysis_mode = "squat_router_protected"
        elif ctx.router_v8_cj_lock:
            final_label = "clean_and_jerk"
            final_confidence = max(_float(ctx.router_v5_confidence), 0.80)
            analysis_mode = "router_v8_context_lock"
        else:
            final_label = ctx.router_v5_label
            final_confidence = _float(ctx.router_v5_confidence)
            analysis_mode = "router_v5"

        if (
            ctx.upright_curl_signature
            and final_label in OLYMPIC_LABELS
            and not ctx.looks_clean_only
            and not ctx.looks_cj
            and not ctx.looks_split
        ):
            final_label = "unknown"
            final_confidence = 0.50
            analysis_mode = "insufficient_signal"
            protected_label = None
            protected_confidence = None
            protected_reason = None
            router_v5_debug = _set_debug_decision(
                router_v5_debug,
                "rejected_upright_curl_signature",
            )

        if ctx.clean_rescue_active:
            final_label = "clean"
            final_confidence = _float(ctx.router_v5_confidence, 0.75)
            analysis_mode = "router_v5"

        snatch_rescue_from_overhead_squat = (
            ctx.olympic_pred == "snatch"
            and str((router_v5_debug or {}).get("decision", ""))
            == "snatch_rescue_from_squat"
            and _float(ctx.olympic_confidence) >= 0.74
            and ctx.squat_label == "overhead_squat"
            and _float(ctx.explosive_score) >= 40.0
            and _debug_float(ctx.bodyweight_debug, "wrist_y_range") >= 0.45
        )

        if (
            ctx.clear_squat_should_hold
            and not ctx.clean_rescue_active
            and not snatch_rescue_from_overhead_squat
        ):
            final_label = ctx.squat_label
            final_confidence = max(
                _float(ctx.squat_confidence),
                (
                    _float(ctx.base_confidence)
                    if ctx.raw_label == ctx.squat_label
                    else 0.0
                ),
            )
            analysis_mode = "squat_router_protected"

    return RouterV5OverrideDecision(
        final_label=final_label,
        final_confidence=final_confidence,
        analysis_mode=analysis_mode,
        protected_label=protected_label,
        protected_confidence=protected_confidence,
        protected_reason=protected_reason,
        router_v5_debug=router_v5_debug,
        snatch_rescue_from_overhead_squat=snatch_rescue_from_overhead_squat,
    )


def select_bodyweight_final_arbitration(
    ctx: BodyweightFinalArbitrationContext,
) -> BodyweightFinalArbitrationDecision:
    """Apply final Router V6/bodyweight authority before rep analysis."""
    pull_up_long_squat_barbell_collision = (
        ctx.router_v6_label == "pull_up"
        and ctx.squat_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and _float(ctx.squat_confidence) >= 0.75
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_confidence) >= 0.70
        and _int(ctx.bodyweight_debug.get("total_frames")) >= 250
        and _debug_float(ctx.bodyweight_debug, "wrist_y_range") >= 0.27
        and _debug_float(ctx.bodyweight_debug, "hip_y_range") >= 0.18
        and _debug_float(ctx.bodyweight_debug, "wrist_above_shoulder_ratio")
        >= 0.85
    )

    pull_up_long_overhead_barbell_collision = (
        ctx.router_v6_label == "pull_up"
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_confidence) >= 0.87
        and _int(ctx.bodyweight_debug.get("total_frames")) >= 300
        and _debug_float(ctx.bodyweight_debug, "shoulder_y_range", 999.0)
        <= 0.11
        and _debug_float(ctx.bodyweight_debug, "hip_y_range", 999.0) <= 0.10
        and _debug_float(ctx.bodyweight_debug, "wrist_y_range") >= 0.20
        and (
            ctx.raw_label == "push_press"
            or ctx.bio_label == "push_press"
        )
    )

    pull_up_barbell_path_collision = _false_pull_up_barbell_path_collision(
        pull_up_label=ctx.router_v6_label,
        raw_label=ctx.raw_label,
        bio_label=ctx.bio_label,
        squat_label=ctx.squat_label,
        olympic_pred=ctx.olympic_pred,
        debug=ctx.bodyweight_debug,
    )

    allowed = (
        not (
            ctx.protected_reason == "strict_press_pattern_detected"
            and ctx.protected_label == "strict_press"
            and _float(ctx.protected_confidence) >= 0.90
        )
        and ctx.router_v6_label in {
            "push_up",
            "pull_up",
            "handstand_push_up",
        }
        and ctx.bodyweight_router_label == ctx.router_v6_label
        and _float(ctx.bodyweight_router_confidence) >= 0.95
        and _float(ctx.router_v6_confidence) >= 0.72
        and not ctx.strong_oly_lock
        and not ctx.strong_bench_evidence
        and not pull_up_long_squat_barbell_collision
        and not pull_up_long_overhead_barbell_collision
        and not pull_up_barbell_path_collision
        and not (
            ctx.router_v6_label == "pull_up"
            and ctx.raw_label == "bench_press"
            and ctx.bio_label == "bench_press"
            and _float(ctx.base_confidence) >= 0.60
            and _float(ctx.bio_confidence) >= 0.60
        )
        and not (
            ctx.router_v6_label == "pull_up"
            and ctx.credible_split_jerk
        )
        and not (
            ctx.router_v6_label == "pull_up"
            and ctx.final_label == "overhead_squat"
            and ctx.squat_label == "overhead_squat"
            and _float(ctx.squat_confidence) >= 0.75
        )
        and not (
            ctx.protected_reason == "push_press_pattern_detected"
            and _float(ctx.olympic_confidence) >= 0.90
        )
        and not (
            ctx.router_v6_label == "pull_up"
            and ctx.raw_label == "push_press"
            and ctx.bio_label == "push_press"
            and _float(ctx.base_confidence) >= 0.65
            and _float(ctx.bio_confidence) >= 0.75
            and _float(ctx.explosive_score) < 30.0
        )
    )

    if not allowed:
        return BodyweightFinalArbitrationDecision(
            final_label=ctx.final_label,
            final_confidence=_float(ctx.final_confidence),
            analysis_mode=ctx.analysis_mode,
            protected_label=ctx.protected_label,
            protected_confidence=ctx.protected_confidence,
            protected_reason=ctx.protected_reason,
            allowed=False,
            pull_up_long_squat_barbell_collision=(
                pull_up_long_squat_barbell_collision
                or pull_up_barbell_path_collision
            ),
            pull_up_long_overhead_barbell_collision=(
                pull_up_long_overhead_barbell_collision
                or pull_up_barbell_path_collision
            ),
        )

    final_confidence = max(
        _float(ctx.bodyweight_router_confidence),
        _float(ctx.router_v6_confidence),
        0.90,
    )
    return BodyweightFinalArbitrationDecision(
        final_label=ctx.router_v6_label,
        final_confidence=final_confidence,
        analysis_mode="router_v6_bodyweight",
        protected_label=ctx.router_v6_label,
        protected_confidence=final_confidence,
        protected_reason="router_v6_bodyweight_winner",
        allowed=True,
        pull_up_long_squat_barbell_collision=(
            pull_up_long_squat_barbell_collision
            or pull_up_barbell_path_collision
        ),
        pull_up_long_overhead_barbell_collision=(
            pull_up_long_overhead_barbell_collision
            or pull_up_barbell_path_collision
        ),
    )


def select_final_shape_authority(
    ctx: FinalShapeAuthorityContext,
) -> FinalShapeAuthorityDecision:
    """Apply deterministic final Olympic/split shape authority decisions."""
    final_label = ctx.final_label
    final_confidence = _float(ctx.final_confidence)
    analysis_mode = ctx.analysis_mode
    protected_label = ctx.protected_label
    protected_confidence = ctx.protected_confidence
    protected_reason = ctx.protected_reason

    if (
        final_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
            "squat",
        }
        and ctx.looks_cj
        and ctx.run_oly_router
        and _float(ctx.olympic_confidence) >= 0.80
        and _float(ctx.explosive_score) >= 100.0
    ):
        final_label = "clean_and_jerk"
        final_confidence = max(0.86, _float(ctx.olympic_confidence))
        analysis_mode = "router_v5"
        protected_label = final_label
        protected_confidence = final_confidence
        protected_reason = "clean_and_jerk_shape_final_recovery"

    split_shape_final_recovery = (
        not ctx.forced_exercise_label
        and ctx.credible_split_jerk
        and not ctx.looks_cj
        and ctx.olympic_pred in {"clean_and_jerk", "split_jerk"}
        and _float(ctx.olympic_confidence) >= 0.80
        and not (
            ctx.raw_label == "bench_press"
            and ctx.bio_label == "bench_press"
            and _float(ctx.base_confidence) >= 0.90
            and _float(ctx.bio_confidence) >= 0.90
            and ctx.router_v6_label == "bench_press"
            and _float(ctx.router_v6_confidence) >= 0.90
        )
        and ctx.raw_label != "push_press"
        and _float(ctx.explosive_score) < 55.0
    )

    if split_shape_final_recovery:
        final_label = "split_jerk"
        final_confidence = max(_float(ctx.olympic_confidence), 0.80)
        analysis_mode = "router_v5"
        protected_label = "split_jerk"
        protected_confidence = final_confidence
        protected_reason = "standalone_split_shape_recovery"

    final_low_explosive_push_press_over_split = (
        not ctx.forced_exercise_label
        and final_label == "split_jerk"
        and protected_reason == "standalone_split_shape_recovery"
        and ctx.raw_label == "squat_front"
        and _float(ctx.base_confidence) >= 0.99
        and ctx.bio_label == "push_press"
        and _float(ctx.bio_confidence) >= 0.99
        and ctx.looks_split
        and not ctx.looks_cj
        and _float(ctx.explosive_score) < 15.0
    )

    if final_low_explosive_push_press_over_split:
        final_label = "push_press"
        final_confidence = max(
            _float(ctx.base_confidence),
            _float(ctx.bio_confidence),
            0.95,
        )
        analysis_mode = "biomechanics_override"
        protected_label = "push_press"
        protected_confidence = final_confidence
        protected_reason = "low_explosive_push_press_over_split"

    final_split_rescue = (
        not ctx.forced_exercise_label
        and isinstance(ctx.router_v5_debug, dict)
        and ctx.router_v5_debug.get("decision") == "standalone_split_from_cj"
    )

    if final_split_rescue:
        final_label = "split_jerk"
        final_confidence = max(_float(ctx.router_v5_confidence), 0.80)
        analysis_mode = "router_v5"
        protected_label = "split_jerk"
        protected_confidence = final_confidence
        protected_reason = "standalone_split_from_cj"

    return FinalShapeAuthorityDecision(
        final_label=final_label,
        final_confidence=final_confidence,
        analysis_mode=analysis_mode,
        protected_label=protected_label,
        protected_confidence=protected_confidence,
        protected_reason=protected_reason,
    )


def select_final_pre_probe_arbitration(
    ctx: FinalPreProbeArbitrationContext,
) -> FinalPreProbeArbitrationDecision:
    """Run pure final arbitration before optional analyzer probes."""
    state = ctx.state

    bodyweight_arbitration = select_bodyweight_final_arbitration(
        BodyweightFinalArbitrationContext(
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            router_v6_label=ctx.router_v6_label,
            router_v6_confidence=ctx.router_v6_confidence,
            bodyweight_router_label=ctx.bodyweight_router_label,
            bodyweight_router_confidence=ctx.bodyweight_router_confidence,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            strong_oly_lock=ctx.strong_oly_lock,
            strong_bench_evidence=ctx.strong_bench_evidence,
            credible_split_jerk=ctx.credible_split_jerk,
            bodyweight_debug=ctx.bodyweight_debug,
        )
    )
    pull_up_long_squat_barbell_collision = (
        bodyweight_arbitration.pull_up_long_squat_barbell_collision
    )
    pull_up_long_overhead_barbell_collision = (
        bodyweight_arbitration.pull_up_long_overhead_barbell_collision
    )

    if bodyweight_arbitration.allowed:
        state = final_state_from_decision(bodyweight_arbitration)

    shape_authority = select_final_shape_authority(
        FinalShapeAuthorityContext(
            forced_exercise_label=ctx.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            router_v6_label=ctx.router_v6_label,
            router_v6_confidence=ctx.router_v6_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            run_oly_router=ctx.run_oly_router,
            credible_split_jerk=ctx.credible_split_jerk,
            looks_cj=ctx.looks_cj,
            looks_split=ctx.looks_split,
            looks_thruster=ctx.looks_thruster,
            bodyweight_debug=ctx.bodyweight_debug,
            router_v5_label=ctx.router_v5_label,
            router_v5_confidence=ctx.router_v5_confidence,
            router_v5_debug=ctx.router_v5_debug,
        )
    )

    return FinalPreProbeArbitrationDecision(
        state=final_state_from_decision(shape_authority),
        pull_up_long_squat_barbell_collision=(
            pull_up_long_squat_barbell_collision
        ),
        pull_up_long_overhead_barbell_collision=(
            pull_up_long_overhead_barbell_collision
        ),
    )


def select_final_post_probe_authority(
    ctx: FinalPostProbeAuthorityContext,
) -> FinalPostProbeAuthorityDecision:
    """Apply final pure authority decisions after optional probe analyzers."""
    final_label = ctx.final_label
    final_confidence = _float(ctx.final_confidence)
    analysis_mode = ctx.analysis_mode
    protected_label = ctx.protected_label
    protected_confidence = ctx.protected_confidence
    protected_reason = ctx.protected_reason

    if (
        not ctx.forced_exercise_label
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_confidence) >= 0.97
        and ctx.raw_label == "push_press"
        and ctx.bio_label == "push_press"
    ):
        final_label = "clean_and_jerk"
        final_confidence = max(_float(ctx.olympic_confidence), 0.97)
        analysis_mode = "router_v5"
        protected_label = "clean_and_jerk"
        protected_confidence = final_confidence
        protected_reason = "clean_and_jerk_high_conf_final_authority"

    if (
        not ctx.forced_exercise_label
        and ctx.raw_label == "push_press"
        and _float(ctx.base_confidence) >= 0.40
        and ctx.bio_label == "squat"
        and _float(ctx.squat_confidence) < 0.60
        and ctx.looks_thruster
        and not ctx.looks_cj
        and not ctx.looks_split
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_confidence) < 0.85
    ):
        final_label = "push_press"
        final_confidence = max(_float(ctx.base_confidence), 0.76)
        analysis_mode = "detailed_rep_analysis"
        protected_label = "push_press"
        protected_confidence = final_confidence
        protected_reason = "push_press_from_false_cj_agreement"

    ring_muscle_up_rescue = (
        not ctx.forced_exercise_label
        and final_label == "pull_up"
        and ctx.router_v6_label == "pull_up"
        and ctx.raw_label == "squat"
        and ctx.bio_label == "push_press"
        and _float(ctx.bio_confidence) >= 0.75
        and _float(ctx.bodyweight_router_confidence) >= 0.97
        and _float(ctx.explosive_score) >= 90.0
        and 0.50 <= _float(ctx.wrist_overhead_ratio) <= 0.85
        and _float(ctx.olympic_confidence) < 0.70
    )

    bar_muscle_up_rescue = (
        not ctx.forced_exercise_label
        and final_label == "pull_up"
        and ctx.router_v6_label == "pull_up"
        and ctx.raw_label == "squat_front"
        and ctx.bio_label == "squat_front"
        and _float(ctx.base_confidence) >= 0.95
        and _float(ctx.bio_confidence) >= 0.95
        and ctx.squat_label == "overhead_squat"
        and _float(ctx.squat_confidence) >= 0.90
        and _float(ctx.bodyweight_router_confidence) >= 0.94
        and _float(ctx.explosive_score) < 30.0
        and 0.35 <= _float(ctx.wrist_overhead_ratio) <= 0.70
    )

    if ring_muscle_up_rescue or bar_muscle_up_rescue:
        final_label = "muscle_up"
        final_confidence = max(
            _float(ctx.bodyweight_router_confidence),
            0.86,
        )
        analysis_mode = "biomechanics_override"
        protected_label = final_label
        protected_confidence = final_confidence
        protected_reason = (
            "ring_muscle_up_final_recovery"
            if ring_muscle_up_rescue
            else "bar_muscle_up_final_recovery"
        )

    return FinalPostProbeAuthorityDecision(
        final_label=final_label,
        final_confidence=final_confidence,
        analysis_mode=analysis_mode,
        protected_label=protected_label,
        protected_confidence=protected_confidence,
        protected_reason=protected_reason,
    )


def select_final_mid_arbitration(
    ctx: FinalMidArbitrationContext,
) -> FinalMidArbitrationDecision:
    """Run pure final arbitration after YOLO and before press probes."""
    state = ctx.state

    clean_bench_pushup_authority = (
        select_final_clean_bench_pushup_authority(
            FinalCleanBenchPushupContext(
                forced_exercise_label=ctx.forced_exercise_label,
                final_label=state.final_label,
                final_confidence=state.final_confidence,
                analysis_mode=state.analysis_mode,
                protected_label=state.protected_label,
                protected_confidence=state.protected_confidence,
                protected_reason=state.protected_reason,
                raw_label=ctx.raw_label,
                base_confidence=ctx.base_confidence,
                bio_label=ctx.bio_label,
                bio_confidence=ctx.bio_confidence,
                router_v6_label=ctx.router_v6_label,
                router_v6_confidence=ctx.router_v6_confidence,
                bodyweight_router_label=ctx.bodyweight_router_label,
                bodyweight_router_confidence=(
                    ctx.bodyweight_router_confidence
                ),
                olympic_pred=ctx.olympic_pred,
                olympic_confidence=ctx.olympic_confidence,
                looks_clean_only=ctx.looks_clean_only,
                looks_cj=ctx.looks_cj,
                looks_split=ctx.looks_split,
                looks_burpee=ctx.looks_burpee,
                strong_front_squat_consensus=(
                    ctx.strong_front_squat_consensus
                ),
                router_v5_label=ctx.router_v5_label,
                router_v5_debug=ctx.router_v5_debug,
                bodyweight_debug=ctx.bodyweight_debug,
            )
        )
    )
    state = final_state_from_decision(clean_bench_pushup_authority)

    collision_recovery = select_final_collision_recovery(
        FinalCollisionRecoveryContext(
            forced_exercise_label=ctx.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            router_v6_label=ctx.router_v6_label,
            router_v6_confidence=ctx.router_v6_confidence,
            bodyweight_router_label=ctx.bodyweight_router_label,
            bodyweight_router_confidence=ctx.bodyweight_router_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            bodyweight_debug=ctx.bodyweight_debug,
        )
    )
    state = final_state_from_decision(collision_recovery)

    pullup_pushup_authority = select_final_pullup_pushup_authority(
        FinalPullupPushupAuthorityContext(
            forced_exercise_label=ctx.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=ctx.raw_label,
            bio_label=ctx.bio_label,
            base_confidence=ctx.base_confidence,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            router_v6_label=ctx.router_v6_label,
            router_v6_confidence=ctx.router_v6_confidence,
            bodyweight_router_label=ctx.bodyweight_router_label,
            bodyweight_router_confidence=ctx.bodyweight_router_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            bench_model_consensus=ctx.bench_model_consensus,
            pull_up_long_squat_barbell_collision=(
                ctx.pull_up_long_squat_barbell_collision
            ),
            pull_up_long_overhead_barbell_collision=(
                ctx.pull_up_long_overhead_barbell_collision
            ),
            looks_cj=ctx.looks_cj,
            looks_split=ctx.looks_split,
            looks_strict=ctx.looks_strict,
            looks_thruster=ctx.looks_thruster,
            bodyweight_debug=ctx.bodyweight_debug,
        )
    )
    state = final_state_from_decision(pullup_pushup_authority)

    press_squat_pre_probe = select_final_press_squat_pre_probe_authority(
        FinalPressSquatPreProbeContext(
            forced_exercise_label=ctx.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            looks_cj=ctx.looks_cj,
            looks_split=ctx.looks_split,
            looks_strict=ctx.looks_strict,
            looks_thruster=ctx.looks_thruster,
            squat_knee_range=ctx.squat_knee_range,
            squat_hip_range=ctx.squat_hip_range,
            bodyweight_debug=ctx.bodyweight_debug,
        )
    )

    return FinalMidArbitrationDecision(
        state=final_state_from_decision(press_squat_pre_probe),
    )


def select_final_tail_arbitration(
    ctx: FinalTailArbitrationContext,
) -> FinalTailArbitrationDecision:
    """Run post-press-probe final arbitration and final Olympic authority."""
    state = ctx.state

    press_squat_post_probe = select_final_press_squat_post_probe_authority(
        FinalPressSquatPostProbeContext(
            forced_exercise_label=ctx.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            router_v6_label=ctx.router_v6_label,
            router_v6_confidence=ctx.router_v6_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            wrist_overhead_ratio=ctx.wrist_overhead_ratio,
            looks_cj=ctx.looks_cj,
            bodyweight_debug=ctx.bodyweight_debug,
            bar_debug=ctx.bar_debug,
        )
    )
    state = final_state_from_decision(press_squat_post_probe)

    clean_segment_authority = select_final_clean_olympic_authority(
        FinalCleanOlympicAuthorityContext(
            forced_exercise_label=ctx.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            wrist_overhead_ratio=ctx.wrist_overhead_ratio,
            looks_cj=ctx.looks_cj,
            looks_split=ctx.looks_split,
            bodyweight_debug=ctx.bodyweight_debug,
            router_v5_debug=ctx.router_v5_debug,
            family_router_shadow=ctx.family_router_shadow,
            learned_family_shadow_label=ctx.learned_family_shadow_label,
            learned_family_shadow_confidence=(
                ctx.learned_family_shadow_confidence
            ),
            learned_family_shadow_trusted=ctx.learned_family_shadow_trusted,
            apply_segment_rules=True,
            apply_olympic_authority=False,
        )
    )
    state = final_state_from_decision(clean_segment_authority)

    state = ctx.deadlift_probe(state)

    olympic_authority = select_final_clean_olympic_authority(
        FinalCleanOlympicAuthorityContext(
            forced_exercise_label=ctx.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            wrist_overhead_ratio=ctx.wrist_overhead_ratio,
            looks_cj=ctx.looks_cj,
            looks_split=ctx.looks_split,
            bodyweight_debug=ctx.bodyweight_debug,
            router_v5_debug=ctx.router_v5_debug,
            family_router_shadow=ctx.family_router_shadow,
            learned_family_shadow_label=ctx.learned_family_shadow_label,
            learned_family_shadow_confidence=(
                ctx.learned_family_shadow_confidence
            ),
            learned_family_shadow_trusted=ctx.learned_family_shadow_trusted,
            apply_segment_rules=False,
            apply_olympic_authority=True,
        )
    )

    return FinalTailArbitrationDecision(
        state=final_state_from_decision(olympic_authority),
    )


def run_final_arbitration(
    ctx: FinalArbitrationContext,
) -> FinalArbitrationDecision:
    """Run the final production arbitration sequence."""
    pre_probe_arbitration = select_final_pre_probe_arbitration(
        FinalPreProbeArbitrationContext(
            state=ctx.state,
            forced_exercise_label=ctx.forced_exercise_label,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            router_v6_label=ctx.router_v6_label,
            router_v6_confidence=ctx.router_v6_confidence,
            bodyweight_router_label=ctx.bodyweight_router_label,
            bodyweight_router_confidence=ctx.bodyweight_router_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            run_oly_router=ctx.run_oly_router,
            strong_oly_lock=ctx.strong_oly_lock,
            strong_bench_evidence=ctx.strong_bench_evidence,
            credible_split_jerk=ctx.credible_split_jerk,
            looks_cj=ctx.looks_cj,
            looks_split=ctx.looks_split,
            looks_thruster=ctx.looks_thruster,
            bodyweight_debug=ctx.bodyweight_debug,
            router_v5_label=ctx.router_v5_label,
            router_v5_confidence=ctx.router_v5_confidence,
            router_v5_debug=ctx.router_v5_debug,
        )
    )
    state = pre_probe_arbitration.state
    pull_up_long_squat_barbell_collision = (
        pre_probe_arbitration.pull_up_long_squat_barbell_collision
    )
    pull_up_long_overhead_barbell_collision = (
        pre_probe_arbitration.pull_up_long_overhead_barbell_collision
    )

    state = ctx.push_press_probe(state)

    post_probe_authority = select_final_post_probe_authority(
        FinalPostProbeAuthorityContext(
            forced_exercise_label=ctx.forced_exercise_label,
            final_label=state.final_label,
            final_confidence=state.final_confidence,
            analysis_mode=state.analysis_mode,
            protected_label=state.protected_label,
            protected_confidence=state.protected_confidence,
            protected_reason=state.protected_reason,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            router_v6_label=ctx.router_v6_label,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            wrist_overhead_ratio=ctx.wrist_overhead_ratio,
            bodyweight_router_confidence=ctx.bodyweight_router_confidence,
            looks_cj=ctx.looks_cj,
            looks_split=ctx.looks_split,
            looks_thruster=ctx.looks_thruster,
        )
    )
    state = final_state_from_decision(post_probe_authority)

    state = ctx.yolo_deadlift_recovery(state)

    mid_arbitration = select_final_mid_arbitration(
        FinalMidArbitrationContext(
            state=state,
            forced_exercise_label=ctx.forced_exercise_label,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            router_v6_label=ctx.router_v6_label,
            router_v6_confidence=ctx.router_v6_confidence,
            bodyweight_router_label=ctx.bodyweight_router_label,
            bodyweight_router_confidence=ctx.bodyweight_router_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            wrist_overhead_ratio=ctx.wrist_overhead_ratio,
            looks_clean_only=ctx.looks_clean_only,
            looks_cj=ctx.looks_cj,
            looks_split=ctx.looks_split,
            looks_strict=ctx.looks_strict,
            looks_thruster=ctx.looks_thruster,
            looks_burpee=ctx.looks_burpee,
            strong_front_squat_consensus=ctx.strong_front_squat_consensus,
            bench_model_consensus=ctx.bench_model_consensus,
            pull_up_long_squat_barbell_collision=(
                pull_up_long_squat_barbell_collision
            ),
            pull_up_long_overhead_barbell_collision=(
                pull_up_long_overhead_barbell_collision
            ),
            squat_knee_range=ctx.squat_knee_range,
            squat_hip_range=ctx.squat_hip_range,
            bodyweight_debug=ctx.bodyweight_debug,
            router_v5_label=ctx.router_v5_label,
            router_v5_debug=ctx.router_v5_debug,
        )
    )
    state = mid_arbitration.state

    state = ctx.push_press_probe(state)

    tail_arbitration = select_final_tail_arbitration(
        FinalTailArbitrationContext(
            state=state,
            forced_exercise_label=ctx.forced_exercise_label,
            raw_label=ctx.raw_label,
            base_confidence=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_confidence=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_confidence=ctx.squat_confidence,
            router_v6_label=ctx.router_v6_label,
            router_v6_confidence=ctx.router_v6_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_confidence=ctx.olympic_confidence,
            explosive_score=ctx.explosive_score,
            wrist_overhead_ratio=ctx.wrist_overhead_ratio,
            looks_cj=ctx.looks_cj,
            looks_split=ctx.looks_split,
            bodyweight_debug=ctx.bodyweight_debug,
            bar_debug=ctx.bar_debug,
            router_v5_debug=ctx.router_v5_debug,
            family_router_shadow=ctx.family_router_shadow,
            learned_family_shadow_label=ctx.learned_family_shadow_label,
            learned_family_shadow_confidence=ctx.learned_family_shadow_confidence,
            learned_family_shadow_trusted=ctx.learned_family_shadow_trusted,
            deadlift_probe=ctx.deadlift_probe,
        )
    )

    return FinalArbitrationDecision(
        state=tail_arbitration.state,
        pull_up_long_squat_barbell_collision=(
            pull_up_long_squat_barbell_collision
        ),
        pull_up_long_overhead_barbell_collision=(
            pull_up_long_overhead_barbell_collision
        ),
    )


def select_final_clean_bench_pushup_authority(
    ctx: FinalCleanBenchPushupContext,
) -> FinalCleanBenchPushupDecision:
    """Apply clean-only, short-bench, and push-up final authority rules."""
    final_label = ctx.final_label
    final_confidence = _float(ctx.final_confidence)
    analysis_mode = ctx.analysis_mode
    protected_label = ctx.protected_label
    protected_confidence = ctx.protected_confidence
    protected_reason = ctx.protected_reason

    final_clean_shape_authority = (
        not ctx.forced_exercise_label
        and ctx.looks_clean_only
        and not ctx.looks_cj
        and not ctx.looks_split
        # A verified burpee full-body cycle must outrank clean-shape fallback.
        and not ctx.looks_burpee
        and not ctx.strong_front_squat_consensus
        and not (
            ctx.raw_label == "bench_press"
            and ctx.bio_label == "bench_press"
            and _float(ctx.base_confidence) >= 0.80
            and _float(ctx.bio_confidence) >= 0.80
        )
        and (
            (
                ctx.olympic_pred == "clean_and_jerk"
                and _float(ctx.olympic_confidence) >= 0.62
            )
            or (
                ctx.router_v5_label == "clean"
                and str((ctx.router_v5_debug or {}).get("decision", ""))
                == "clean_rescue_from_weak_snatch"
            )
        )
    )

    if final_clean_shape_authority:
        final_label = "clean"
        final_confidence = 0.75
        analysis_mode = "shape_override"
        protected_label = "clean"
        protected_confidence = final_confidence
        protected_reason = "clean_only_shape_final_authority"

    final_short_horizontal_bench_recovery = (
        not ctx.forced_exercise_label
        and final_label == "push_press"
        and ctx.raw_label == "push_press"
        and ctx.bio_label == "push_press"
        and ctx.router_v6_label == "push_press"
        and protected_reason == "push_press_pattern_detected"
        and _debug_float(
            ctx.bodyweight_debug,
            "wrist_above_shoulder_ratio",
            1.0,
        )
        < 0.05
        and _debug_float(ctx.bodyweight_debug, "avg_torso_angle", 180.0)
        < 15.0
        and _debug_float(ctx.bodyweight_debug, "elbow_range") >= 150.0
        and _debug_float(ctx.bodyweight_debug, "wrist_y_range") >= 0.15
        and _debug_float(ctx.bodyweight_debug, "shoulder_y_range") >= 0.15
        and _debug_float(ctx.bodyweight_debug, "hip_y_range") >= 0.18
        and 20 <= _int(ctx.bodyweight_debug.get("total_frames")) <= 60
    )

    if final_short_horizontal_bench_recovery:
        final_label = "bench_press"
        final_confidence = max(
            final_confidence,
            _float(ctx.base_confidence),
            _float(ctx.bio_confidence),
            0.86,
        )
        analysis_mode = "biomechanics_override"
        protected_label = "bench_press"
        protected_confidence = final_confidence
        protected_reason = "short_horizontal_bench_final_recovery"

    final_push_up_authority = (
        not ctx.forced_exercise_label
        and final_label in {"clean", "clean_and_jerk"}
        and ctx.bodyweight_router_label == "push_up"
        and ctx.router_v6_label == "push_up"
        and _float(ctx.bodyweight_router_confidence) >= 0.85
        and _float(ctx.router_v6_confidence) >= 0.64
        and _debug_float(
            ctx.bodyweight_debug,
            "wrist_above_shoulder_ratio",
            1.0,
        )
        < 0.10
        and not ctx.looks_cj
        and not ctx.looks_split
    )

    if final_push_up_authority:
        final_label = "push_up"
        final_confidence = max(
            _float(ctx.bodyweight_router_confidence),
            _float(ctx.router_v6_confidence),
            0.86,
        )
        analysis_mode = "router_v6_bodyweight"
        protected_label = "push_up"
        protected_confidence = final_confidence
        protected_reason = "push_up_final_authority"

    return FinalCleanBenchPushupDecision(
        final_label=final_label,
        final_confidence=final_confidence,
        analysis_mode=analysis_mode,
        protected_label=protected_label,
        protected_confidence=protected_confidence,
        protected_reason=protected_reason,
    )


def select_final_pullup_pushup_authority(
    ctx: FinalPullupPushupAuthorityContext,
) -> FinalPullupPushupAuthorityDecision:
    """Apply late horizontal push-up and pull-up authority decisions."""
    final_label = ctx.final_label
    final_confidence = _float(ctx.final_confidence)
    analysis_mode = ctx.analysis_mode
    protected_label = ctx.protected_label
    protected_confidence = ctx.protected_confidence
    protected_reason = ctx.protected_reason

    final_horizontal_push_up_recovery = (
        not ctx.forced_exercise_label
        and final_label == "clean_and_jerk"
        and ctx.raw_label == "deadlift"
        and ctx.bio_label == "squat"
        and ctx.bodyweight_router_label == "push_up"
        and _float(ctx.bodyweight_router_confidence) >= 0.50
        and _debug_float(ctx.bodyweight_debug, "avg_torso_angle") >= 160.0
        and _debug_float(ctx.bodyweight_debug, "shoulder_y_range") >= 0.30
        and _debug_float(ctx.bodyweight_debug, "hip_y_range") >= 0.20
        and _debug_float(ctx.bodyweight_debug, "elbow_range") >= 150.0
        and _debug_float(
            ctx.bodyweight_debug,
            "wrist_above_shoulder_ratio",
            1.0,
        )
        <= 0.30
    )

    if final_horizontal_push_up_recovery:
        final_label = "push_up"
        final_confidence = max(
            _float(ctx.bodyweight_router_confidence),
            0.86,
        )
        analysis_mode = "router_v6_bodyweight"
        protected_label = "push_up"
        protected_confidence = final_confidence
        protected_reason = "horizontal_push_up_final_recovery"

    final_low_motion_pull_up_recovery = (
        not ctx.forced_exercise_label
        and final_label == "push_press"
        and ctx.raw_label == "push_press"
        and ctx.bio_label == "push_press"
        and ctx.router_v6_label == "push_press"
        and protected_reason == "push_press_pattern_detected"
        and ctx.bodyweight_router_label == "pull_up"
        and _float(ctx.bodyweight_router_confidence) >= 0.85
        and _debug_float(
            ctx.bodyweight_debug,
            "wrist_above_shoulder_ratio",
            1.0,
        )
        < 0.05
        and _debug_float(
            ctx.bodyweight_debug,
            "mean_wrist_minus_shoulder_y",
        )
        >= 0.20
        and _debug_float(ctx.bodyweight_debug, "elbow_range", 999.0)
        <= 25.0
        and _debug_float(ctx.bodyweight_debug, "min_elbow") >= 145.0
        and _debug_float(ctx.bodyweight_debug, "shoulder_y_range", 1.0)
        <= 0.03
        and _debug_float(ctx.bodyweight_debug, "hip_y_range", 1.0) <= 0.03
        and _debug_float(ctx.bodyweight_debug, "avg_wrist_forward") >= 0.05
        and 50 <= _int(ctx.bodyweight_debug.get("total_frames")) <= 110
    )

    if final_low_motion_pull_up_recovery:
        final_label = "pull_up"
        final_confidence = max(
            _float(ctx.bodyweight_router_confidence),
            0.86,
        )
        analysis_mode = "router_v6_bodyweight"
        protected_label = "pull_up"
        protected_confidence = final_confidence
        protected_reason = "low_motion_pull_up_final_recovery"

    final_pull_up_authority = (
        not ctx.forced_exercise_label
        and final_label != "muscle_up"
        and ctx.bodyweight_router_label == "pull_up"
        and ctx.router_v6_label == "pull_up"
        and _float(ctx.bodyweight_router_confidence) >= 0.90
        and not ctx.bench_model_consensus
        and not ctx.pull_up_long_squat_barbell_collision
        and not ctx.pull_up_long_overhead_barbell_collision
        and not _false_pull_up_barbell_path_collision(
            pull_up_label=ctx.bodyweight_router_label,
            raw_label=ctx.raw_label,
            bio_label=ctx.bio_label,
            squat_label=ctx.squat_label,
            olympic_pred=ctx.olympic_pred,
            debug=ctx.bodyweight_debug,
        )
        and not ctx.looks_cj
        and not (
            ctx.looks_split
            and not (
                ctx.squat_label == "overhead_squat"
                and _float(ctx.bodyweight_router_confidence) >= 0.99
                and _float(ctx.router_v6_confidence) >= 0.74
                and _float(ctx.explosive_score) < 20.0
            )
        )
        and not ctx.looks_strict
        and not (
            ctx.looks_thruster
            and _float(ctx.olympic_confidence) < 0.70
        )
        and not (
            ctx.squat_label == "overhead_squat"
            and _float(ctx.squat_confidence) >= 0.80
            and ctx.looks_thruster
            and _float(ctx.explosive_score) < 25.0
        )
        and not (
            ctx.olympic_pred == "snatch"
            and _float(ctx.olympic_confidence) >= 0.60
            and ctx.raw_label == "squat"
            and ctx.bio_label == "push_press"
            and _float(ctx.explosive_score) >= 50.0
            and _float(ctx.router_v6_confidence) < 0.75
        )
        and protected_reason != "push_press_pattern_detected"
    )

    if final_pull_up_authority:
        final_label = "pull_up"
        final_confidence = max(
            _float(ctx.bodyweight_router_confidence),
            _float(ctx.router_v6_confidence),
            0.90,
        )
        analysis_mode = "router_v6_bodyweight"
        protected_label = "pull_up"
        protected_confidence = final_confidence
        protected_reason = "pull_up_final_authority"

    return FinalPullupPushupAuthorityDecision(
        final_label=final_label,
        final_confidence=final_confidence,
        analysis_mode=analysis_mode,
        protected_label=protected_label,
        protected_confidence=protected_confidence,
        protected_reason=protected_reason,
    )


def select_final_collision_recovery(
    ctx: FinalCollisionRecoveryContext,
) -> FinalCollisionRecoveryDecision:
    """Apply pure late squat/deadlift/snatch collision recoveries."""
    final_label = ctx.final_label
    final_confidence = _float(ctx.final_confidence)
    analysis_mode = ctx.analysis_mode
    protected_label = ctx.protected_label
    protected_confidence = ctx.protected_confidence
    protected_reason = ctx.protected_reason

    final_long_squat_over_cj_recovery = (
        not ctx.forced_exercise_label
        and final_label == "clean_and_jerk"
        and ctx.raw_label == "squat"
        and _float(ctx.base_confidence) >= 0.99
        and ctx.bio_label == "squat"
        and _float(ctx.bio_confidence) >= 0.99
        and ctx.squat_label == "squat_front"
        and _float(ctx.squat_confidence) >= 0.85
        and ctx.router_v6_label == "squat"
        and _float(ctx.router_v6_confidence) >= 0.98
        and ctx.olympic_pred == "clean_and_jerk"
        and 0.70 <= _float(ctx.olympic_confidence) <= 0.74
        and _debug_float(ctx.bodyweight_debug, "avg_torso_angle", 999.0)
        <= 3.0
        and _debug_float(
            ctx.bodyweight_debug,
            "wrist_above_shoulder_ratio",
            1.0,
        )
        <= 0.20
        and _debug_float(ctx.bodyweight_debug, "shoulder_y_range") >= 0.55
        and _debug_float(ctx.bodyweight_debug, "hip_y_range") >= 0.40
        and _debug_float(ctx.bodyweight_debug, "elbow_range") >= 145.0
        and _debug_float(ctx.bodyweight_debug, "avg_elbow", 999.0) <= 60.0
        and _int(ctx.bodyweight_debug.get("total_frames")) >= 500
    )

    if final_long_squat_over_cj_recovery:
        final_label = "squat_back"
        final_confidence = max(_float(ctx.squat_confidence), 0.86)
        analysis_mode = "squat_router_protected"
        protected_label = "squat_back"
        protected_confidence = final_confidence
        protected_reason = "long_squat_over_cj_final_recovery"

    final_medium_squat_over_cj_recovery = (
        not ctx.forced_exercise_label
        and final_label == "clean_and_jerk"
        and ctx.raw_label == "squat"
        and _float(ctx.base_confidence) >= 0.99
        and ctx.bio_label == "squat"
        and _float(ctx.bio_confidence) >= 0.99
        and ctx.squat_label == "squat_front"
        and _float(ctx.squat_confidence) >= 0.96
        and ctx.router_v6_label == "squat"
        and _float(ctx.router_v6_confidence) >= 0.98
        and ctx.olympic_pred == "clean_and_jerk"
        and 0.70 <= _float(ctx.olympic_confidence) <= 0.75
        and 3.5
        <= _debug_float(ctx.bodyweight_debug, "avg_torso_angle")
        <= 7.0
        and _debug_float(
            ctx.bodyweight_debug,
            "wrist_above_shoulder_ratio",
            1.0,
        )
        <= 0.10
        and _debug_float(ctx.bodyweight_debug, "shoulder_y_range") >= 0.45
        and _debug_float(ctx.bodyweight_debug, "hip_y_range") >= 0.60
        and _debug_float(ctx.bodyweight_debug, "elbow_range") >= 165.0
        and _debug_float(ctx.bodyweight_debug, "avg_elbow", 999.0) <= 45.0
        and 300 <= _int(ctx.bodyweight_debug.get("total_frames")) <= 450
    )

    if final_medium_squat_over_cj_recovery:
        final_label = "squat_back"
        final_confidence = max(_float(ctx.squat_confidence), 0.97)
        analysis_mode = "squat_router_protected"
        protected_label = "squat_back"
        protected_confidence = final_confidence
        protected_reason = "medium_squat_over_cj_final_recovery"

    final_long_deadlift_over_squat_recovery = (
        not ctx.forced_exercise_label
        and final_label == "squat_back"
        and ctx.raw_label == "push_press"
        and 0.84 <= _float(ctx.base_confidence) <= 0.90
        and ctx.bio_label == "squat"
        and 0.84 <= _float(ctx.bio_confidence) <= 0.90
        and ctx.squat_label == "squat_back"
        and _float(ctx.squat_confidence) >= 0.96
        and ctx.router_v6_label == "squat_back"
        and 0.58 <= _float(ctx.router_v6_confidence) <= 0.64
        and ctx.olympic_pred == "snatch"
        and 0.58 <= _float(ctx.olympic_confidence) <= 0.64
        and 22.0
        <= _debug_float(ctx.bodyweight_debug, "avg_torso_angle")
        <= 30.0
        and 0.30
        <= _debug_float(ctx.bodyweight_debug, "wrist_y_range")
        <= 0.40
        and 0.30
        <= _debug_float(ctx.bodyweight_debug, "shoulder_y_range")
        <= 0.42
        and 0.12 <= _debug_float(ctx.bodyweight_debug, "hip_y_range") <= 0.20
        and 35.0 <= _debug_float(ctx.bodyweight_debug, "elbow_range") <= 55.0
        and _debug_float(ctx.bodyweight_debug, "avg_elbow") >= 170.0
        and _debug_float(ctx.bodyweight_debug, "min_elbow") >= 125.0
        and _debug_float(
            ctx.bodyweight_debug,
            "wrist_above_shoulder_ratio",
            1.0,
        )
        <= 0.05
        and _int(ctx.bodyweight_debug.get("total_frames")) >= 800
    )

    if final_long_deadlift_over_squat_recovery:
        final_label = "deadlift"
        final_confidence = 0.82
        analysis_mode = "biomechanics_override"
        protected_label = "deadlift"
        protected_confidence = final_confidence
        protected_reason = "long_deadlift_over_squat_final_recovery"

    final_short_deadlift_over_snatch_recovery = (
        not ctx.forced_exercise_label
        and final_label == "snatch"
        and ctx.raw_label == "deadlift"
        and ctx.bio_label == "squat"
        and ctx.bodyweight_router_label == "push_up"
        and _float(ctx.bodyweight_router_confidence) >= 0.85
        and ctx.router_v6_label == "push_up"
        and _debug_float(ctx.bodyweight_debug, "avg_torso_angle") >= 60.0
        and _debug_float(
            ctx.bodyweight_debug,
            "wrist_above_shoulder_ratio",
            1.0,
        )
        < 0.05
        and _debug_float(ctx.bodyweight_debug, "elbow_range", 999.0) <= 30.0
        and _debug_float(ctx.bodyweight_debug, "min_elbow") >= 150.0
        and _debug_float(ctx.bodyweight_debug, "wrist_y_range", 1.0) <= 0.30
        and _debug_float(ctx.bodyweight_debug, "hip_y_range", 1.0) <= 0.20
        and _int(ctx.bodyweight_debug.get("total_frames")) <= 80
        and _float(ctx.explosive_score) >= 80.0
    )

    if final_short_deadlift_over_snatch_recovery:
        final_label = "deadlift"
        final_confidence = max(_float(ctx.base_confidence), 0.82)
        analysis_mode = "biomechanics_override"
        protected_label = "deadlift"
        protected_confidence = final_confidence
        protected_reason = "short_deadlift_over_snatch_final_recovery"

    final_long_snatch_over_squat_recovery = (
        not ctx.forced_exercise_label
        and final_label == "squat_back"
        and ctx.raw_label == "squat"
        and 0.45 <= _float(ctx.base_confidence) <= 0.55
        and ctx.bio_label == "squat"
        and 0.45 <= _float(ctx.bio_confidence) <= 0.55
        and ctx.squat_label == "squat_back"
        and 0.92 <= _float(ctx.squat_confidence) <= 0.96
        and ctx.router_v6_label == "squat_back"
        and 0.56 <= _float(ctx.router_v6_confidence) <= 0.62
        and ctx.olympic_pred == "clean_and_jerk"
        and 0.54 <= _float(ctx.olympic_confidence) <= 0.60
        and _debug_float(ctx.bodyweight_debug, "wrist_y_range") >= 0.65
        and _debug_float(ctx.bodyweight_debug, "shoulder_y_range") >= 0.28
        and _debug_float(ctx.bodyweight_debug, "hip_y_range") >= 0.30
        and _debug_float(ctx.bodyweight_debug, "elbow_range") >= 165.0
        and _debug_float(ctx.bodyweight_debug, "avg_elbow") >= 155.0
        and _debug_float(ctx.bodyweight_debug, "min_elbow", 999.0) <= 10.0
        and _debug_float(ctx.bodyweight_debug, "wrist_above_shoulder_ratio")
        >= 0.45
        and _int(ctx.bodyweight_debug.get("total_frames")) >= 1200
        and _float(ctx.explosive_score) >= 130.0
    )

    if final_long_snatch_over_squat_recovery:
        final_label = "snatch"
        final_confidence = max(_float(ctx.olympic_confidence), 0.70)
        analysis_mode = "router_v5"
        protected_label = "snatch"
        protected_confidence = final_confidence
        protected_reason = "long_snatch_over_squat_final_recovery"

    if should_recover_short_bench_over_pushup(
        forced_exercise_label=ctx.forced_exercise_label,
        final_label=final_label,
        raw_label=ctx.raw_label,
        base_conf=ctx.base_confidence,
        bio_label=ctx.bio_label,
        bio_conf=ctx.bio_confidence,
        squat_label=ctx.squat_label,
        squat_conf=ctx.squat_confidence,
        router_v6_label=ctx.router_v6_label,
        router_v6_conf=ctx.router_v6_confidence,
        olympic_pred=ctx.olympic_pred,
        olympic_conf=ctx.olympic_confidence,
        bodyweight_debug=ctx.bodyweight_debug,
    ):
        final_label = "bench_press"
        final_confidence = 0.86
        analysis_mode = "biomechanics_override"
        protected_label = "bench_press"
        protected_confidence = final_confidence
        protected_reason = "short_bench_over_pushup_final_recovery"

    return FinalCollisionRecoveryDecision(
        final_label=final_label,
        final_confidence=final_confidence,
        analysis_mode=analysis_mode,
        protected_label=protected_label,
        protected_confidence=protected_confidence,
        protected_reason=protected_reason,
    )


def select_final_press_squat_pre_probe_authority(
    ctx: FinalPressSquatPreProbeContext,
) -> FinalPressSquatAuthorityDecision:
    """Apply final press/squat authority rules before analyzer probes."""
    final_label = ctx.final_label
    final_confidence = _float(ctx.final_confidence)
    analysis_mode = ctx.analysis_mode
    protected_label = ctx.protected_label
    protected_confidence = ctx.protected_confidence
    protected_reason = ctx.protected_reason

    if should_recover_controlled_push_press(
        forced_exercise_label=ctx.forced_exercise_label,
        final_label=final_label,
        bio_label=ctx.bio_label,
        bio_conf=ctx.bio_confidence,
        squat_label=ctx.squat_label,
        squat_conf=ctx.squat_confidence,
        olympic_pred=ctx.olympic_pred,
        olympic_conf=ctx.olympic_confidence,
        looks_cj=ctx.looks_cj,
        looks_split=ctx.looks_split,
        explosive_score=ctx.explosive_score,
        bodyweight_debug=ctx.bodyweight_debug,
    ):
        final_label = "push_press"
        final_confidence = max(_float(ctx.bio_confidence), 0.78)
        analysis_mode = "biomechanics_override"
        protected_label = "push_press"
        protected_confidence = final_confidence
        protected_reason = "controlled_push_press_final_recovery"

    controlled_overhead_squat_recovery = (
        not ctx.forced_exercise_label
        and final_label == "clean_and_jerk"
        and ctx.squat_label == "overhead_squat"
        and _float(ctx.squat_confidence) >= 0.80
        and ctx.raw_label == "push_press"
        and ctx.bio_label == "push_press"
        and _float(ctx.base_confidence) >= 0.95
        and _float(ctx.bio_confidence) >= 0.95
        and ctx.olympic_pred == "clean_and_jerk"
        and 0.75 <= _float(ctx.olympic_confidence) <= 0.85
        and not ctx.looks_cj
        and not ctx.looks_split
        and _float(ctx.explosive_score) < 30.0
        and _debug_float(ctx.bodyweight_debug, "wrist_above_shoulder_ratio")
        >= 0.85
        and _debug_float(ctx.bodyweight_debug, "avg_elbow") >= 150.0
        and _debug_float(ctx.bodyweight_debug, "min_elbow") >= 30.0
        and _int(ctx.bodyweight_debug.get("total_frames")) >= 200
    )

    if controlled_overhead_squat_recovery:
        final_label = "overhead_squat"
        final_confidence = max(_float(ctx.squat_confidence), 0.81)
        analysis_mode = "detailed_rep_analysis"
        protected_label = "overhead_squat"
        protected_confidence = final_confidence
        protected_reason = "controlled_overhead_squat_final_recovery"

    if should_recover_strict_press(
        forced_exercise_label=ctx.forced_exercise_label,
        final_label=final_label,
        raw_label=ctx.raw_label,
        bio_label=ctx.bio_label,
        looks_strict=ctx.looks_strict,
        looks_split=ctx.looks_split,
        looks_thruster=ctx.looks_thruster,
        explosive_score=ctx.explosive_score,
        squat_knee_range=ctx.squat_knee_range,
        squat_hip_range=ctx.squat_hip_range,
        bodyweight_debug=ctx.bodyweight_debug,
    ):
        final_label = "strict_press"
        final_confidence = max(0.86, final_confidence)
        analysis_mode = "shape_override"
        protected_label = "strict_press"
        protected_confidence = final_confidence
        protected_reason = "strict_press_low_leg_drive_final_authority"

    return FinalPressSquatAuthorityDecision(
        final_label=final_label,
        final_confidence=final_confidence,
        analysis_mode=analysis_mode,
        protected_label=protected_label,
        protected_confidence=protected_confidence,
        protected_reason=protected_reason,
    )


def select_final_press_squat_post_probe_authority(
    ctx: FinalPressSquatPostProbeContext,
) -> FinalPressSquatAuthorityDecision:
    """Apply final press/squat authority rules after analyzer probes."""
    final_label = ctx.final_label
    final_confidence = _float(ctx.final_confidence)
    analysis_mode = ctx.analysis_mode
    protected_label = ctx.protected_label
    protected_confidence = ctx.protected_confidence
    protected_reason = ctx.protected_reason

    if should_recover_push_press_over_weak_cj_split(
        forced_exercise_label=ctx.forced_exercise_label,
        final_label=final_label,
        raw_label=ctx.raw_label,
        base_conf=ctx.base_confidence,
        bio_label=ctx.bio_label,
        bio_conf=ctx.bio_confidence,
        router_v6_label=ctx.router_v6_label,
        router_v6_conf=ctx.router_v6_confidence,
        olympic_pred=ctx.olympic_pred,
        olympic_conf=ctx.olympic_confidence,
        looks_cj=ctx.looks_cj,
        explosive_score=ctx.explosive_score,
        bodyweight_debug=ctx.bodyweight_debug,
    ):
        final_label = "push_press"
        final_confidence = max(
            _float(ctx.base_confidence),
            _float(ctx.bio_confidence),
            _float(ctx.router_v6_confidence),
            0.86,
        )
        analysis_mode = "biomechanics_override"
        protected_label = "push_press"
        protected_confidence = final_confidence
        protected_reason = "push_press_over_weak_cj_split"

    push_press_back_squat_recovery_reason = (
        should_recover_push_press_over_back_squat(
            forced_exercise_label=ctx.forced_exercise_label,
            final_label=final_label,
            raw_label=ctx.raw_label,
            base_conf=ctx.base_confidence,
            bio_label=ctx.bio_label,
            bio_conf=ctx.bio_confidence,
            squat_label=ctx.squat_label,
            squat_conf=ctx.squat_confidence,
            olympic_pred=ctx.olympic_pred,
            olympic_conf=ctx.olympic_confidence,
            looks_cj=ctx.looks_cj,
            explosive_score=ctx.explosive_score,
            bodyweight_debug=ctx.bodyweight_debug,
        )
    )

    if push_press_back_squat_recovery_reason:
        final_label = "push_press"
        final_confidence = max(
            _float(ctx.base_confidence),
            _float(ctx.bio_confidence),
            0.86,
        )
        analysis_mode = "biomechanics_override"
        protected_label = "push_press"
        protected_confidence = final_confidence
        protected_reason = push_press_back_squat_recovery_reason

    if should_recover_explosive_push_press_authority(
        forced_exercise_label=ctx.forced_exercise_label,
        final_label=final_label,
        raw_label=ctx.raw_label,
        base_conf=ctx.base_confidence,
        bio_label=ctx.bio_label,
        bio_conf=ctx.bio_confidence,
        squat_label=ctx.squat_label,
        squat_conf=ctx.squat_confidence,
        olympic_conf=ctx.olympic_confidence,
        explosive_score=ctx.explosive_score,
        bar_debug=ctx.bar_debug,
    ):
        final_label = "push_press"
        final_confidence = max(
            _float(ctx.base_confidence),
            _float(ctx.bio_confidence),
            0.90,
        )
        analysis_mode = "biomechanics_override"
        protected_label = "push_press"
        protected_confidence = final_confidence
        protected_reason = "explosive_push_press_consensus_final_authority"

    sustained_overhead_squat = (
        not ctx.forced_exercise_label
        and final_label == "clean_and_jerk"
        and ctx.squat_label == "overhead_squat"
        and _float(ctx.squat_confidence) >= 0.80
        and _float(ctx.wrist_overhead_ratio) >= 0.90
        and _int(ctx.bodyweight_debug.get("total_frames")) >= 300
        and not ctx.looks_cj
        and _float(ctx.olympic_confidence) <= 0.82
    )

    if sustained_overhead_squat:
        final_label = "overhead_squat"
        final_confidence = _float(ctx.squat_confidence, 0.81)
        analysis_mode = "detailed_rep_analysis"
        protected_label = "overhead_squat"
        protected_confidence = final_confidence
        protected_reason = "sustained_overhead_squat_final_authority"

    return FinalPressSquatAuthorityDecision(
        final_label=final_label,
        final_confidence=final_confidence,
        analysis_mode=analysis_mode,
        protected_label=protected_label,
        protected_confidence=protected_confidence,
        protected_reason=protected_reason,
    )


def plan_final_push_press_probe(
    ctx: FinalPushPressProbeContext,
) -> FinalPushPressProbePlan:
    """Decide whether push-press analyzer confirmation should run."""
    if ctx.forced_exercise_label:
        return FinalPushPressProbePlan(False, 0, None)

    split_to_push_press = (
        ctx.final_label == "split_jerk"
        and ctx.protected_reason == "standalone_split_from_cj"
        and ctx.raw_label == "push_press"
        and _float(ctx.base_confidence) >= 0.99
        and ctx.bio_label == "push_press"
        and _float(ctx.bio_confidence) >= 0.99
        and ctx.squat_label == "overhead_squat"
        and 0.78 <= _float(ctx.squat_confidence) <= 0.85
        and ctx.olympic_pred == "clean_and_jerk"
        and 0.80 <= _float(ctx.olympic_confidence) <= 0.85
        and _float(ctx.explosive_score) < 20.0
        and 0.80 <= _debug_float(ctx.bar_debug, "overhead_ratio") < 0.90
        and _int(ctx.bar_debug.get("total_frames")) >= 250
    )

    if split_to_push_press:
        return FinalPushPressProbePlan(
            True,
            3,
            "push_press_analyzer_over_split_authority",
        )

    pull_up_to_push_press = (
        ctx.final_label == "pull_up"
        and ctx.raw_label == "push_press"
        and _float(ctx.base_confidence) >= 0.99
        and ctx.bio_label == "push_press"
        and _float(ctx.bio_confidence) >= 0.99
        and ctx.squat_label == "overhead_squat"
        and _float(ctx.squat_confidence) >= 0.86
        and ctx.bodyweight_router_label == "pull_up"
        and _float(ctx.bodyweight_router_confidence) >= 0.95
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_confidence) < 0.70
        and _float(ctx.explosive_score) < 30.0
    )

    if pull_up_to_push_press:
        return FinalPushPressProbePlan(
            True,
            2,
            "push_press_analyzer_over_pull_up_authority",
        )

    return FinalPushPressProbePlan(False, 0, None)


def apply_final_push_press_probe_result(
    ctx: FinalPushPressProbeContext,
    *,
    probe_reps: list[Any],
    minimum_rep_count: int,
    recovery_reason: str | None,
) -> FinalPushPressProbeDecision:
    """Apply a confirmed final push-press probe result."""
    if len(probe_reps or []) >= int(minimum_rep_count or 0) > 0:
        final_confidence = max(
            _float(ctx.base_confidence),
            _float(ctx.bio_confidence),
            0.90,
        )
        return FinalPushPressProbeDecision(
            final_label="push_press",
            final_confidence=final_confidence,
            analysis_mode="biomechanics_override",
            protected_label="push_press",
            protected_confidence=final_confidence,
            protected_reason=recovery_reason,
        )

    return FinalPushPressProbeDecision(
        final_label=ctx.final_label,
        final_confidence=_float(ctx.final_confidence),
        analysis_mode=ctx.analysis_mode,
        protected_label=ctx.protected_label,
        protected_confidence=ctx.protected_confidence,
        protected_reason=ctx.protected_reason,
    )


def select_final_clean_olympic_authority(
    ctx: FinalCleanOlympicAuthorityContext,
) -> FinalCleanOlympicAuthorityDecision:
    """Apply final clean-segment and specialized Olympic authority rules."""
    final_label = ctx.final_label
    final_confidence = _float(ctx.final_confidence)
    analysis_mode = ctx.analysis_mode
    protected_label = ctx.protected_label
    protected_confidence = ctx.protected_confidence
    protected_reason = ctx.protected_reason

    router_v5_features = (
        ctx.router_v5_debug.get("features", {})
        if isinstance(ctx.router_v5_debug, dict)
        else {}
    )
    family_shadow_family = (
        ctx.family_router_shadow.get("family")
        if isinstance(ctx.family_router_shadow, dict)
        else None
    )
    learned_family_olympic = (
        ctx.learned_family_shadow_trusted
        and ctx.learned_family_shadow_label == "olympic"
        and _float(ctx.learned_family_shadow_confidence) >= 0.70
    )

    final_short_clean_segment = (
        ctx.apply_segment_rules
        and not ctx.forced_exercise_label
        and final_label == "squat_back"
        and ctx.raw_label == "squat"
        and ctx.bio_label == "squat"
        and _float(ctx.base_confidence) >= 0.95
        and _float(ctx.bio_confidence) >= 0.95
        and ctx.squat_label == "squat_back"
        and _float(ctx.squat_confidence) >= 0.90
        and ctx.olympic_pred == "clean_and_jerk"
        and 0.50 <= _float(ctx.olympic_confidence) < 0.65
        and ctx.looks_split
        and not ctx.looks_cj
        and _float(ctx.explosive_score) >= 80.0
        and _float(ctx.wrist_overhead_ratio) < 0.20
        and _debug_float(ctx.bodyweight_debug, "wrist_y_range") >= 0.35
        and _int(ctx.bodyweight_debug.get("total_frames"), 999) <= 60
    )

    if final_short_clean_segment:
        final_label = "clean"
        final_confidence = 0.75
        analysis_mode = "shape_override"
        protected_label = "clean"
        protected_confidence = final_confidence
        protected_reason = "short_explosive_clean_segment"

    final_squat_protected_clean_segment = (
        ctx.apply_segment_rules
        and not ctx.forced_exercise_label
        and final_label == "squat_back"
        and analysis_mode == "squat_router_protected"
        and ctx.raw_label == "squat"
        and ctx.bio_label == "squat"
        and ctx.squat_label == "squat_back"
        and _float(ctx.squat_confidence) >= 0.90
        and ctx.olympic_pred == "clean"

        # A protected, near-unanimous back squat should not be stolen by
        # weak/moderate clean evidence. Require genuinely strong Olympic
        # evidence before crossing family boundaries.
        and _float(ctx.olympic_confidence) >= 0.75
        and (
            _float(ctx.olympic_confidence) >= 0.85
            or learned_family_olympic
        )
        and (
            family_shadow_family == "olympic"
            or learned_family_olympic
        )
        and _float(ctx.wrist_overhead_ratio) < 0.25
        and not ctx.looks_cj
    )

    if final_squat_protected_clean_segment:
        final_label = "clean"
        final_confidence = max(
            _float(ctx.olympic_confidence),
            _float(ctx.learned_family_shadow_confidence),
            0.75,
        )
        analysis_mode = "shape_override"
        protected_label = "clean"
        protected_confidence = final_confidence
        protected_reason = "squat_protected_clean_segment_recovery"

    final_segmented_clean_from_weak_cj = (
        ctx.apply_segment_rules
        and not ctx.forced_exercise_label
        and final_label == "clean_and_jerk"
        and ctx.raw_label == "squat"
        and ctx.bio_label == "squat"
        and _float(ctx.base_confidence) >= 0.95
        and _float(ctx.bio_confidence) >= 0.95
        and ctx.squat_label == "squat_front"
        and _float(ctx.squat_confidence) < 0.60
        and ctx.olympic_pred == "clean_and_jerk"
        and 0.50 <= _float(ctx.olympic_confidence) < 0.60
        and ctx.looks_split
        and ctx.looks_cj
        and _float(ctx.explosive_score) < 60.0
        and _float(ctx.wrist_overhead_ratio) < 0.40
        and 80 <= _int(ctx.bodyweight_debug.get("total_frames")) <= 130
        and _float(router_v5_features.get("catch_overhead", 1.0)) == 0.0
        and _float(router_v5_features.get("extension_to_catch", 999.0))
        <= 8.0
        and _float(router_v5_features.get("catch_to_finish", 999.0)) <= 70.0
        and _float(router_v5_features.get("lockout_duration", 999.0)) <= 40.0
    )

    if final_segmented_clean_from_weak_cj:
        final_label = "clean"
        final_confidence = 0.75
        analysis_mode = "shape_override"
        protected_label = "clean"
        protected_confidence = final_confidence
        protected_reason = "segmented_clean_from_weak_cj"

    final_compact_clean_and_jerk = (
        ctx.apply_segment_rules
        and not ctx.forced_exercise_label
        and final_label == "squat_back"
        and ctx.raw_label == "squat"
        and ctx.bio_label == "push_press"
        and _float(ctx.base_confidence) >= 0.70
        and _float(ctx.bio_confidence) >= 0.75
        and ctx.squat_label == "squat_back"
        and _float(ctx.squat_confidence) >= 0.90
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_confidence) >= 0.75
        and str((ctx.router_v5_debug or {}).get("decision", ""))
        == "agreement"
        and _float(ctx.explosive_score) >= 80.0
        and 0.55 <= _float(ctx.wrist_overhead_ratio) <= 0.85
        and 80 <= _int(ctx.bodyweight_debug.get("total_frames")) <= 220
        and _float(router_v5_features.get("catch_to_finish")) >= 50.0
        and _float(router_v5_features.get("lockout_duration")) >= 40.0
    )

    if final_compact_clean_and_jerk:
        final_label = "clean_and_jerk"
        final_confidence = _float(ctx.olympic_confidence, 0.75)
        analysis_mode = "router_v5"
        protected_label = "clean_and_jerk"
        protected_confidence = final_confidence
        protected_reason = "compact_clean_and_jerk_final_authority"

    final_olympic_authority = None

    if (
        ctx.apply_olympic_authority
        and not ctx.forced_exercise_label
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_confidence) >= 0.65
        and ctx.looks_cj
    ):
        final_olympic_authority = "clean_and_jerk"
    elif (
        ctx.apply_olympic_authority
        and not ctx.forced_exercise_label
        and ctx.olympic_pred == "snatch"
        and _float(ctx.olympic_confidence) >= 0.70
        and _float(ctx.explosive_score) >= 80.0
    ):
        final_olympic_authority = "snatch"
    elif (
        ctx.apply_olympic_authority
        and not ctx.forced_exercise_label
        and ctx.olympic_pred == "snatch"
        and _float(ctx.olympic_confidence) >= 0.50
        and ctx.raw_label == "squat"
        and ctx.bio_label == "push_press"
        and ctx.squat_label == "squat_back"
        and ctx.looks_split
        and not ctx.looks_cj
    ):
        final_olympic_authority = "snatch"
    elif (
        ctx.apply_olympic_authority
        and not ctx.forced_exercise_label
        and ctx.olympic_pred == "split_jerk"
        and _float(ctx.olympic_confidence) >= 0.70
        and ctx.looks_split
        and not ctx.looks_cj
    ):
        final_olympic_authority = "split_jerk"
    elif (
        ctx.apply_olympic_authority
        and not ctx.forced_exercise_label
        and final_label == "overhead_squat"
        and ctx.squat_label == "overhead_squat"
        and learned_family_olympic
        and ctx.raw_label == "deadlift"
        and ctx.bio_label == "push_press"
        and ctx.olympic_pred == "clean_and_jerk"
        and 0.25 <= _float(ctx.olympic_confidence) < 0.60
        and _float(ctx.wrist_overhead_ratio) >= 0.55
        and ctx.looks_split
    ):
        final_olympic_authority = "snatch"

    if final_olympic_authority:
        final_label = final_olympic_authority
        final_confidence = max(_float(ctx.olympic_confidence), 0.75)
        analysis_mode = "olympic_final_authority"
        protected_label = final_label
        protected_confidence = final_confidence
        protected_reason = "specialized_olympic_final_authority"

    return FinalCleanOlympicAuthorityDecision(
        final_label=final_label,
        final_confidence=final_confidence,
        analysis_mode=analysis_mode,
        protected_label=protected_label,
        protected_confidence=protected_confidence,
        protected_reason=protected_reason,
    )


def plan_final_deadlift_probe(
    ctx: FinalDeadliftProbeContext,
) -> FinalDeadliftProbePlan:
    """Decide whether the final deadlift analyzer confirmation should run."""
    squat_to_deadlift_geometry = (
        ctx.final_label == "squat_back"
        and ctx.raw_label == "squat"
        and ctx.bio_label == "squat"
        and _float(ctx.base_confidence) >= 0.95
        and _float(ctx.bio_confidence) >= 0.95
        and ctx.squat_label == "squat_back"
        and _float(ctx.squat_confidence) >= 0.90
        and _float(ctx.wrist_overhead_ratio) < 0.02
        and _debug_float(ctx.bodyweight_debug, "avg_elbow") >= 165.0
        and _debug_float(ctx.bodyweight_debug, "elbow_range", 999.0) <= 30.0
        and _debug_float(ctx.bodyweight_debug, "avg_torso_angle") >= 20.0
        and _debug_float(ctx.bodyweight_debug, "wrist_y_range") >= 0.25
        and _debug_float(ctx.bar_debug, "front_rack_elbow_p25") >= 160.0
        and _debug_float(ctx.bar_debug, "wrist_height_above_shoulder")
        <= -0.10
        and 25.0 <= _float(ctx.explosive_score) <= 55.0
        and 100 <= _int(ctx.bodyweight_debug.get("total_frames")) <= 220
    )

    bench_to_deadlift_geometry = (
        ctx.final_label == "bench_press"
        and ctx.raw_label == "bench_press"
        and ctx.bio_label == "bench_press"
        and ctx.router_v6_label == "bench_press"
        and _float(ctx.base_confidence) >= 0.95
        and _float(ctx.bio_confidence) >= 0.95
        and _float(ctx.router_v6_confidence) >= 0.95
        and _debug_float(ctx.bodyweight_debug, "elbow_range", 999.0) <= 20.0
        and _debug_float(ctx.bodyweight_debug, "min_elbow") >= 155.0
        and _debug_float(ctx.bodyweight_debug, "avg_elbow") >= 165.0
        and _debug_float(ctx.bodyweight_debug, "avg_torso_angle", 999.0)
        <= 30.0
        and _float(ctx.deadlift_knee_range) >= 80.0
        and _float(ctx.deadlift_hip_range) >= 80.0
        and _float(ctx.deadlift_torso_range) >= 25.0
    )

    if ctx.forced_exercise_label:
        return FinalDeadliftProbePlan(False, None)

    if bench_to_deadlift_geometry:
        return FinalDeadliftProbePlan(
            True,
            "straight_arm_hinge_deadlift_recovery",
        )

    if squat_to_deadlift_geometry:
        return FinalDeadliftProbePlan(
            True,
            "deadlift_analyzer_disagreement_rescue",
        )

    return FinalDeadliftProbePlan(False, None)


def apply_final_deadlift_probe_result(
    ctx: FinalDeadliftProbeContext,
    *,
    probe_reps: list[Any],
    recovery_reason: str | None,
) -> FinalDeadliftProbeDecision:
    """Apply a confirmed final deadlift probe result."""
    if probe_reps:
        final_confidence = max(
            _float(ctx.base_confidence),
            _float(ctx.bio_confidence),
            _float(ctx.squat_confidence),
            0.90,
        )
        return FinalDeadliftProbeDecision(
            final_label="deadlift",
            final_confidence=final_confidence,
            analysis_mode="biomechanics_override",
            protected_label="deadlift",
            protected_confidence=final_confidence,
            protected_reason=recovery_reason,
        )

    return FinalDeadliftProbeDecision(
        final_label=ctx.final_label,
        final_confidence=_float(ctx.final_confidence),
        analysis_mode=ctx.analysis_mode,
        protected_label=ctx.protected_label,
        protected_confidence=ctx.protected_confidence,
        protected_reason=ctx.protected_reason,
    )


def plan_yolo_deadlift_recovery(
    ctx: YoloDeadliftRecoveryContext,
) -> YoloDeadliftRecoveryPlan:
    """Decide whether YOLO deadlift/squat analyzer comparison should run."""
    should_probe = (
        ctx.use_yolo_tracking
        and not ctx.forced_exercise_label
        and ctx.final_label
        in {"squat", "squat_back", "squat_front", "overhead_squat"}
        and ctx.raw_label in {"squat", "squat_back", "squat_front"}
        and ctx.bio_label == "squat"
        and _float(ctx.olympic_confidence) < 0.80
    )

    if not should_probe:
        return YoloDeadliftRecoveryPlan(False, None)

    return YoloDeadliftRecoveryPlan(
        True,
        ctx.final_label if ctx.final_label != "squat" else "squat_back",
    )


def apply_yolo_deadlift_recovery_result(
    ctx: YoloDeadliftRecoveryContext,
    *,
    deadlift_probe_reps: list[Any],
    squat_probe_reps: list[Any],
) -> YoloDeadliftRecoveryDecision:
    """Apply a confirmed YOLO deadlift-vs-squat rep-count disagreement."""
    deadlift_count = len(deadlift_probe_reps or [])
    squat_count = len(squat_probe_reps or [])
    recovered = 1 <= deadlift_count <= 4 and squat_count >= deadlift_count * 2 + 3

    if recovered:
        final_confidence = max(0.86, _float(ctx.final_confidence))
        return YoloDeadliftRecoveryDecision(
            final_label="deadlift",
            final_confidence=final_confidence,
            analysis_mode="yolo_deadlift_recovery",
            protected_label="deadlift",
            protected_confidence=final_confidence,
            protected_reason="yolo_rep_count_deadlift_recovery",
            recovered=True,
        )

    return YoloDeadliftRecoveryDecision(
        final_label=ctx.final_label,
        final_confidence=_float(ctx.final_confidence),
        analysis_mode=ctx.analysis_mode,
        protected_label=ctx.protected_label,
        protected_confidence=ctx.protected_confidence,
        protected_reason=ctx.protected_reason,
        recovered=False,
    )


def select_fallback_final_decision(
    ctx: FallbackFinalContext,
) -> FallbackFinalDecision:
    """
    Select the fallback production label after early decisions do not fire.

    This is a behavior-preserving extraction from the remaining squat/Olympic
    fallback chain in `main.py`.
    """
    if (
        ctx.squat_confident
        and ctx.bio_label not in {"push_up", "pull_up", "handstand_push_up"}
        and ctx.raw_label == "squat"
        and ctx.squat_label in {"squat_back", "squat_front"}
        and not ctx.truly_explosive
        and not ctx.strong_overhead
        and _float(ctx.olympic_conf) < 0.85
    ):
        return FallbackFinalDecision(
            label=str(ctx.squat_label),
            confidence=_float(ctx.squat_conf),
            mode="detailed_rep_analysis",
        )

    if (
        ctx.squat_label == "overhead_squat"
        and _float(ctx.squat_conf) >= 0.75
        and _float(ctx.olympic_conf) < 0.85
        and not _push_press_pull_up_signature_with_low_wrist_range(
            ctx.bodyweight_debug
        )
    ):
        return FallbackFinalDecision(
            label="overhead_squat",
            confidence=max(_float(ctx.bar_conf), _float(ctx.squat_conf)),
            mode="detailed_rep_analysis",
        )

    clean_thruster_collision = (
        ctx.olympic_pred == "clean"
        and ctx.looks_thruster
    )

    if (
        ctx.run_oly_router
        and ctx.olympic_pred in OLYMPIC_LABELS
        and _float(ctx.olympic_conf) >= 0.65
        and not ctx.push_press_should_hold
        and not clean_thruster_collision
        and not (
            ctx.squat_label == "squat_front"
            and _float(ctx.squat_conf) >= 0.80
            and _float(ctx.olympic_conf) < 0.75
        )
        and not (
            _push_press_pull_up_signature(ctx.bodyweight_debug)
            and not ctx.truly_explosive
            and ctx.raw_label == "push_press"
        )
    ):
        return FallbackFinalDecision(
            label=str(ctx.olympic_pred),
            confidence=_float(ctx.olympic_conf),
            mode="olympic_locked",
        )

    if (
        ctx.run_oly_router
        and ctx.olympic_pred == "snatch"
        and _float(ctx.olympic_conf) >= 0.45
        and _float(ctx.wrist_overhead_ratio) > 0.35
        and ctx.looks_split
        and ctx.raw_label in {"squat", "squat_front", "squat_back"}
    ):
        return FallbackFinalDecision(
            label="snatch",
            confidence=max(0.60, _float(ctx.olympic_conf)),
            mode="olympic_rescue",
        )

    if (
        (
            ctx.bar_says_overhead_squat
            and not ctx.truly_explosive
        )
        or (
            ctx.squat_label == "overhead_squat"
            and _float(ctx.squat_conf) >= 0.90
            and _float(ctx.olympic_conf) < 0.85
            and not _push_press_pull_up_signature_with_low_wrist_range(
                ctx.bodyweight_debug
            )
        )
    ):
        return FallbackFinalDecision(
            label="overhead_squat",
            confidence=max(_float(ctx.bar_conf), _float(ctx.squat_conf)),
            mode="detailed_rep_analysis",
        )

    if (
        ctx.run_oly_router
        and ctx.olympic_pred in OLYMPIC_LABELS
        and _float(ctx.olympic_conf) >= 0.80
        and ctx.truly_explosive
        and not ctx.push_press_should_hold
        and not clean_thruster_collision
    ):
        return FallbackFinalDecision(
            label=str(ctx.olympic_pred),
            confidence=_float(ctx.olympic_conf),
            mode="olympic_locked",
        )

    if (
        ctx.run_oly_router
        and ctx.olympic_pred in OLYMPIC_LABELS
        and _float(ctx.olympic_conf) >= 0.65
        and ctx.truly_explosive
        and not ctx.push_press_should_hold
        and not clean_thruster_collision
    ):
        return FallbackFinalDecision(
            label=str(ctx.olympic_pred),
            confidence=_float(ctx.olympic_conf),
            mode="olympic_locked",
        )

    strong_push_press_consensus = (
        ctx.raw_label == "push_press"
        and _float(ctx.base_conf) >= 0.85
        and ctx.bio_label == "push_press"
        and _float(ctx.bio_conf) >= 0.85
        and ctx.router_v6_label == "push_press"
        and _float(ctx.router_v6_conf) >= 0.85
    )

    if (
        ctx.looks_split
        and not strong_push_press_consensus
        and not (
            ctx.squat_label == "overhead_squat"
            and _float(ctx.squat_conf) >= 0.75
        )
        and _float(ctx.explosive_score) > 20.0
        and ctx.olympic_pred != "snatch"
        and not (
            ctx.raw_label == "squat"
            and ctx.bio_label == "squat"
            and _float(ctx.base_conf) >= 0.95
            and _float(ctx.bio_conf) >= 0.95
            and ctx.olympic_pred == "clean_and_jerk"
            and _float(ctx.olympic_conf) < 0.60
            and ctx.looks_cj
            and _float(ctx.wrist_overhead_ratio) < 0.40
        )
    ):
        return FallbackFinalDecision(
            label="split_jerk",
            confidence=0.80,
            mode="shape_override",
        )

    if (
        ctx.looks_clean_only
        and ctx.truly_explosive
        and not ctx.looks_cj
        and not ctx.looks_split
    ):
        return FallbackFinalDecision(
            label="clean",
            confidence=0.75,
            mode="shape_override",
        )

    if (
        ctx.squat_confident
        and not ctx.truly_explosive
        and ctx.bio_label not in {"push_up", "pull_up", "handstand_push_up"}
    ):
        return FallbackFinalDecision(
            label=str(ctx.squat_label),
            confidence=_float(ctx.squat_conf),
            mode="detailed_rep_analysis",
        )

    if ctx.looks_cj and ctx.strong_overhead:
        return FallbackFinalDecision(
            label="clean_and_jerk",
            confidence=0.68,
            mode="shape_override",
        )

    if ctx.looks_clean_only:
        return FallbackFinalDecision(
            label="clean",
            confidence=0.68,
            mode="shape_override",
        )

    if (
        ctx.squat_label
        and _float(ctx.squat_conf) >= 0.55
        and ctx.has_real_squat_motion
        and ctx.bio_label not in {"push_up", "pull_up", "handstand_push_up"}
        and not _push_press_pull_up_signature_with_low_wrist_range(
            ctx.bodyweight_debug
        )
    ):
        return FallbackFinalDecision(
            label=str(ctx.squat_label),
            confidence=_float(ctx.squat_conf),
            mode="detailed_rep_analysis",
        )

    if (
        ctx.run_oly_router
        and ctx.olympic_pred in OLYMPIC_LABELS
        and _float(ctx.olympic_conf) >= 0.50
        and not ctx.push_press_should_hold
        and not clean_thruster_collision
        and not (
            _push_press_pull_up_signature(ctx.bodyweight_debug)
            and not ctx.truly_explosive
            and ctx.raw_label == "push_press"
        )
    ):
        return FallbackFinalDecision(
            label=str(ctx.olympic_pred),
            confidence=_float(ctx.olympic_conf),
            mode="olympic_locked",
        )

    if ctx.bio_override and ctx.bio_label == "push_press":
        return FallbackFinalDecision(
            label="push_press",
            confidence=_float(ctx.bio_conf),
            mode="biomechanics_override",
        )

    if (
        ctx.raw_label == "push_press"
        and _float(ctx.base_conf) >= 0.65
        and not ctx.looks_split
    ):
        return FallbackFinalDecision(
            label="push_press",
            confidence=_float(ctx.base_conf),
            mode="base_model_locked",
        )

    return FallbackFinalDecision(
        label="unknown",
        confidence=0.5,
        mode="insufficient_signal",
    )


def select_early_final_decision(
    ctx: EarlyFinalContext,
) -> EarlyFinalDecision | None:
    """
    Select the first early final label after protected evidence is known.

    This keeps the existing `main.py` priority order while moving the
    production decision into the final router module.
    """
    if ctx.protected_label and not ctx.strong_oly_lock:
        return EarlyFinalDecision(
            label=ctx.protected_label,
            confidence=_float(ctx.protected_conf),
            mode="biomechanics_override",
            protected_label=ctx.protected_label,
            protected_conf=_float(ctx.protected_conf),
            protected_reason=ctx.protected_reason,
        )

    if (
        ctx.bodyweight_router_label == "pull_up"
        and _float(ctx.bodyweight_router_conf) >= 0.95
        and not (
            ctx.squat_label == "overhead_squat"
            and _float(ctx.squat_conf) < 0.86
        )
        and ctx.raw_label == "push_press"
        and ctx.bio_label == "push_press"
        and _debug_float(ctx.bodyweight_debug, "wrist_above_shoulder_ratio")
        >= 0.80
    ):
        final_conf = max(_float(ctx.bodyweight_router_conf), 0.90)
        return EarlyFinalDecision(
            label="pull_up",
            confidence=final_conf,
            mode="bodyweight_router",
            protected_label="pull_up",
            protected_conf=final_conf,
            protected_reason="bodyweight_router_pull_up_high_conf",
        )

    if (
        ctx.squat_label == "squat_front"
        and _float(ctx.squat_conf) >= 0.80
        and ctx.raw_label == "push_press"
        and _float(ctx.olympic_conf) < 0.75
    ):
        return EarlyFinalDecision(
            label="squat_front",
            confidence=_float(ctx.squat_conf),
            mode="detailed_rep_analysis",
            protected_label=ctx.protected_label,
            protected_conf=_float(ctx.protected_conf),
            protected_reason=ctx.protected_reason,
        )

    if (
        ctx.run_oly_router
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_conf) >= 0.75
        and _float(ctx.explosive_score) > 20.0
        and not (
            ctx.pull_up_router_guard
            and not (
                ctx.squat_label == "overhead_squat"
                and _float(ctx.squat_conf) >= 0.70
            )
            and _float(ctx.olympic_conf) < 0.95
        )
        and not (
            ctx.squat_label == "squat_front"
            and _float(ctx.squat_conf) >= 0.80
            and ctx.raw_label == "squat"
            and _float(ctx.base_conf) >= 0.95
            and ctx.bio_label == "squat"
            and _float(ctx.bio_conf) >= 0.95
        )
        and (
            ctx.raw_label in {
                "squat",
                "squat_back",
                "squat_front",
                "bench_press",
            }
            or (
                ctx.raw_label == "push_press"
                and not ctx.looks_split
            )
        )
    ):
        return EarlyFinalDecision(
            label="clean_and_jerk",
            confidence=_float(ctx.olympic_conf),
            mode="olympic_locked",
            protected_label=ctx.protected_label,
            protected_conf=_float(ctx.protected_conf),
            protected_reason=ctx.protected_reason,
        )

    strong_push_press_consensus = (
        ctx.raw_label == "push_press"
        and _float(ctx.base_conf) >= 0.85
        and ctx.bio_label == "push_press"
        and _float(ctx.bio_conf) >= 0.85
        and ctx.router_v6_label == "push_press"
        and _float(ctx.router_v6_conf) >= 0.85
    )

    if (
        ctx.looks_split
        and not strong_push_press_consensus
        and not (
            ctx.squat_label == "overhead_squat"
            and _float(ctx.squat_conf) >= 0.75
        )
        and not ctx.looks_cj
        and _float(ctx.explosive_score) > 20.0
        and ctx.olympic_pred != "snatch"
        and (
            ctx.raw_label == "push_press"
            or ctx.olympic_pred != "clean_and_jerk"
            or _float(ctx.olympic_conf) < 0.75
        )
    ):
        return EarlyFinalDecision(
            label="split_jerk",
            confidence=0.80,
            mode="shape_override",
            protected_label=ctx.protected_label,
            protected_conf=_float(ctx.protected_conf),
            protected_reason=ctx.protected_reason,
        )

    if (
        strong_push_press_consensus
        and not ctx.looks_cj
        and not (
            ctx.squat_label == "overhead_squat"
            and _float(ctx.squat_conf) >= 0.75
            and _wrist_to_shoulder_range_ratio(ctx.bodyweight_debug) <= 1.50
        )
    ):
        return EarlyFinalDecision(
            label="push_press",
            confidence=max(
                _float(ctx.base_conf),
                _float(ctx.bio_conf),
                _float(ctx.router_v6_conf),
            ),
            mode="biomechanics_override",
            protected_label=ctx.protected_label,
            protected_conf=_float(ctx.protected_conf),
            protected_reason=ctx.protected_reason,
        )

    if (
        ctx.raw_label == "squat_front"
        and _float(ctx.base_conf) >= 0.90
        and ctx.bio_label == "squat"
        and _float(ctx.bio_conf) >= 0.85
        and not ctx.truly_explosive
        and not ctx.strong_overhead
        and _float(ctx.olympic_conf) < 0.75
    ):
        return EarlyFinalDecision(
            label="squat_front",
            confidence=max(_float(ctx.base_conf), _float(ctx.bio_conf)),
            mode="squat_raw_consensus",
            protected_label=ctx.protected_label,
            protected_conf=_float(ctx.protected_conf),
            protected_reason=ctx.protected_reason,
        )

    return None


def select_protected_evidence(
    ctx: ProtectedEvidenceContext,
) -> ProtectedEvidenceDecision:
    """
    Select the first protected movement label before global arbitration.

    This is a behavior-preserving extraction from `main.py`; it intentionally
    keeps the existing priority order.
    """
    protected_label = None
    protected_conf = 0.0
    protected_reason = None

    bench_model_consensus = (
        ctx.raw_label == "bench_press"
        and ctx.bio_label == "bench_press"
        and _float(ctx.base_conf) >= 0.80
        and _float(ctx.bio_conf) >= 0.80
        # A verified deep squat-to-press cycle is incompatible with bench.
        # Do not let broad bench-model agreement suppress thruster geometry.
        and not bool(ctx.looks_thruster)
        and not (
            ctx.squat_label in {"squat_back", "squat_front"}
            and _float(ctx.squat_conf) >= 0.95
        )
    )

    protection_label = getattr(ctx.protection, "label", None)
    protection_conf = getattr(ctx.protection, "confidence", 0.0)
    protection_reason = getattr(ctx.protection, "reason", None)

    if ctx.strong_bench_evidence or bench_model_consensus:
        protected_label = "bench_press"
        protected_conf = max(
            _float(ctx.base_conf),
            _float(ctx.bio_conf),
            0.95,
        )
        protected_reason = (
            "strong_bench_model_agreement"
            if ctx.strong_bench_evidence
            else "bench_model_consensus"
        )

    elif (
        protection_label
        and not (
            protection_label == "pull_up"
            and ctx.olympic_pred == "snatch"
            and _float(ctx.olympic_conf) >= 0.60
            and ctx.raw_label == "squat"
            and ctx.bio_label == "push_press"
            and _float(ctx.explosive_score) >= 50.0
            and _float(ctx.router_v6_conf) < 0.75
        )
    ):
        protected_label = protection_label
        protected_conf = _float(protection_conf)
        protected_reason = protection_reason

    elif (
        ctx.bio_label == "bench_press"
        and not ctx.looks_strict
        and not ctx.looks_thruster
        and not (
            ctx.looks_push_up
            or ctx.looks_pull_up
            or ctx.looks_handstand_push_up
        )
    ):
        bench_blocked_by_oly = (
            ctx.olympic_pred in OLYMPIC_LABELS
            and _float(ctx.olympic_conf) >= 0.90
            and ctx.raw_label in {"squat", "deadlift", "push_press"}
        )

        if not bench_blocked_by_oly:
            protected_label = "bench_press"
            protected_conf = max(_float(ctx.bio_conf), 0.80)
            protected_reason = ctx.bio_reason
        else:
            protected_reason = "bench_blocked_by_olympic_router"

    elif (
        ctx.raw_label == "squat"
        and ctx.squat_label == "squat_front"
        and not ctx.looks_clean_only
        and not ctx.looks_cj
        and not ctx.looks_split
        and not ctx.looks_strict
        and not ctx.looks_thruster
        and not (
            ctx.olympic_pred in OLYMPIC_LABELS
            and _float(ctx.olympic_conf) >= 0.50
        )
    ):
        protected_label = "bench_press"
        protected_conf = max(_float(ctx.base_conf), 0.80)
        protected_reason = "bench_press_squat_front_false_positive"

    elif (
        ctx.raw_label in {"squat", "deadlift"}
        and ctx.squat_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and not (
            ctx.squat_label == "squat_front"
            and _float(ctx.squat_conf) >= 0.90
        )
        and ctx.bio_label in {"squat", "deadlift"}
        and _float(ctx.explosive_score) >= 60.0
        and _debug_float(ctx.bodyweight_debug, "shoulder_y_range", 1.0)
        <= 0.20
        and _debug_float(ctx.bodyweight_debug, "hip_y_range", 1.0)
        <= 0.20
        and _float(ctx.olympic_conf) < 0.70
        and not (
            ctx.raw_label in {"squat", "squat_front", "squat_back"}
            and _float(ctx.base_conf) >= 0.90
            and ctx.bio_label == "squat"
            and _float(ctx.bio_conf) >= 0.90
        )
        and not ctx.looks_clean_only
        and not ctx.looks_cj
        and not ctx.looks_split
        and not ctx.looks_strict
    ):
        protected_label = "bench_press"
        protected_conf = max(
            _float(ctx.base_conf),
            _float(ctx.bio_conf),
            0.80,
        )
        protected_reason = "bench_press_fast_press_rescue"

    elif (
        ctx.raw_label in {
            "squat",
            "squat_front",
            "squat_back",
            "push_press",
        }
        and ctx.squat_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and not (
            ctx.squat_label == "overhead_squat"
            and _float(ctx.squat_conf) >= 0.80
        )
        and ctx.bio_label in {"squat", "push_press", "deadlift"}
        and not (
            ctx.squat_label == "squat_front"
            and _float(ctx.squat_conf) >= 0.90
        )
        and not (
            ctx.raw_label == "squat"
            and ctx.bio_label == "push_press"
            and ctx.squat_label == "squat_front"
            and ctx.olympic_pred == "clean_and_jerk"
            and _float(ctx.olympic_conf) >= 0.73
            and _float(ctx.explosive_score) >= 45.0
        )
        and not (
            ctx.raw_label in {"squat", "squat_front", "squat_back"}
            and _float(ctx.base_conf) >= 0.85
            and ctx.bio_label == "squat"
            and _float(ctx.bio_conf) >= 0.85
        )
        and not ctx.looks_clean_only
        and not ctx.looks_cj
        and not ctx.looks_split
        and not ctx.looks_thruster
        and (
            not ctx.deadlift_setup_geometry
            or ctx.short_low_camera_bench_setup
        )
        and (
            _int(ctx.bar_debug.get("squat_frames_used", 999), 999) <= 35
            or (
                ctx.run_oly_router
                and
                _float(ctx.olympic_conf) < 0.80
                and _float(ctx.wrist_overhead_ratio) > 0.25
                and _float(ctx.explosive_score) > 20.0
            )
        )
    ):
        protected_label = "bench_press"
        protected_conf = max(
            _float(ctx.base_conf),
            _float(ctx.bio_conf),
            0.80,
        )
        protected_reason = "bench_press_short_squat_rescue"

    elif (
        ctx.bio_label == "deadlift"
        and (
            (
                ctx.bio_override
                and not (
                    ctx.raw_label in {"squat", "squat_front"}
                    and ctx.squat_confident
                )
            )
            or (
                _float(ctx.wrist_overhead_ratio) < 0.08
                and _float(ctx.explosive_score) <= 30.0
                and _debug_float(ctx.bar_debug, "front_rack_elbow_p25")
                >= 130.0
                and not ctx.looks_clean_only
                and not ctx.looks_cj
                and not ctx.looks_split
                and not ctx.looks_strict
                and not ctx.looks_thruster
            )
        )
    ):
        protected_label = "deadlift"
        protected_conf = _float(ctx.bio_conf)
        protected_reason = ctx.bio_reason

    vertical_pullup_signature = (
        ctx.raw_label == "push_press"
        and _debug_float(ctx.bodyweight_debug, "wrist_above_shoulder_ratio")
        >= 0.80
        and _debug_float(
            ctx.bodyweight_debug,
            "mean_wrist_minus_shoulder_y",
            1.0,
        )
        <= -0.08
        and _debug_float(ctx.bodyweight_debug, "elbow_range") >= 110.0
        and _debug_float(ctx.bodyweight_debug, "min_elbow", 180.0) <= 45.0
        and _debug_float(ctx.bodyweight_debug, "avg_torso_angle", 180.0)
        <= 10.0
        and _debug_float(ctx.bodyweight_debug, "avg_wrist_forward", 1.0)
        <= 0.03
        and _debug_float(ctx.bodyweight_debug, "wrist_y_range", 1.0) <= 0.18
        and not ctx.truly_explosive
    )

    if vertical_pullup_signature and protected_label is None:
        protected_label = "pull_up"
        protected_conf = 0.86
        protected_reason = "vertical_pullup_signature"

    front_squat_weak_router_recovery = (
        protected_label is None
        and ctx.raw_label == "squat"
        and _float(ctx.base_conf) >= 0.95
        and ctx.bio_label == "squat"
        and _float(ctx.bio_conf) >= 0.95
        and ctx.squat_label == "squat_back"
        and _float(ctx.squat_conf) <= 0.65
        and 0.25 <= _float(ctx.wrist_overhead_ratio) <= 0.55
        and _debug_float(ctx.bar_debug.get("scores", {}), "squat_front")
        >= 0.35
        and _debug_float(ctx.bar_debug.get("scores", {}), "overhead_squat")
        < 0.70
    )

    if front_squat_weak_router_recovery:
        protected_label = "squat_front"
        protected_conf = 0.80
        protected_reason = "front_squat_weak_router_recovery"

    early_pull_up_long_squat_collision = (
        protected_label == "pull_up"
        and ctx.squat_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and _float(ctx.squat_conf) >= 0.75
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_conf) >= 0.70
        and _int(ctx.bodyweight_debug.get("total_frames")) >= 250
        and _debug_float(ctx.bodyweight_debug, "wrist_y_range") >= 0.27
        and _debug_float(ctx.bodyweight_debug, "hip_y_range") >= 0.18
        and _debug_float(ctx.bodyweight_debug, "wrist_above_shoulder_ratio")
        >= 0.85
    )

    early_pull_up_long_overhead_collision = (
        protected_label == "pull_up"
        and ctx.olympic_pred == "clean_and_jerk"
        and _float(ctx.olympic_conf) >= 0.87
        and _int(ctx.bodyweight_debug.get("total_frames")) >= 300
        and _debug_float(ctx.bodyweight_debug, "shoulder_y_range", 999.0)
        <= 0.11
        and _debug_float(ctx.bodyweight_debug, "hip_y_range", 999.0) <= 0.10
        and _debug_float(ctx.bodyweight_debug, "wrist_y_range") >= 0.20
        and (
            ctx.raw_label == "push_press"
            or ctx.bio_label == "push_press"
        )
    )

    if (
        early_pull_up_long_squat_collision
        or early_pull_up_long_overhead_collision
    ):
        protected_label = None
        protected_conf = 0.0
        protected_reason = None

    return ProtectedEvidenceDecision(
        label=protected_label,
        confidence=protected_conf,
        reason=protected_reason,
        bench_model_consensus=bench_model_consensus,
    )
