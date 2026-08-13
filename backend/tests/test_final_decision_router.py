from types import SimpleNamespace

from app.ml.final_decision_router import (
    BodyweightFinalArbitrationContext,
    EarlyFinalContext,
    FallbackFinalContext,
    FinalArbitrationContext,
    FinalCleanOlympicAuthorityContext,
    FinalDeadliftProbeContext,
    FinalCleanBenchPushupContext,
    FinalCollisionRecoveryContext,
    FinalDecisionState,
    FinalMidArbitrationContext,
    FinalPreProbeArbitrationContext,
    FinalPostProbeAuthorityContext,
    FinalPressSquatPostProbeContext,
    FinalPressSquatPreProbeContext,
    FinalPushPressProbeContext,
    FinalPullupPushupAuthorityContext,
    FinalShapeAuthorityContext,
    FinalTailArbitrationContext,
    ProtectedEvidenceContext,
    RouterV5OverrideContext,
    YoloDeadliftRecoveryContext,
    apply_final_deadlift_probe_result,
    apply_final_push_press_probe_result,
    apply_yolo_deadlift_recovery_result,
    plan_final_deadlift_probe,
    plan_final_push_press_probe,
    plan_yolo_deadlift_recovery,
    run_final_arbitration,
    select_bodyweight_final_arbitration,
    select_early_final_decision,
    select_fallback_final_decision,
    select_final_clean_olympic_authority,
    select_final_clean_bench_pushup_authority,
    select_final_collision_recovery,
    select_final_mid_arbitration,
    select_final_post_probe_authority,
    select_final_press_squat_post_probe_authority,
    select_final_press_squat_pre_probe_authority,
    select_final_pre_probe_arbitration,
    select_final_tail_arbitration,
    select_final_pullup_pushup_authority,
    select_final_shape_authority,
    select_protected_evidence,
    select_router_v5_override,
)


def _ctx(**overrides):
    data = {
        "raw_label": "squat",
        "base_conf": 0.50,
        "bio_label": "squat",
        "bio_conf": 0.50,
        "bio_override": False,
        "bio_reason": "",
        "squat_label": "squat_back",
        "squat_conf": 0.50,
        "olympic_pred": None,
        "olympic_conf": 0.0,
        "run_oly_router": False,
        "explosive_score": 0.0,
        "wrist_overhead_ratio": 0.0,
        "router_v6_conf": 0.0,
        "strong_bench_evidence": False,
        "protection": SimpleNamespace(
            label=None,
            confidence=0.0,
            reason=None,
        ),
        "looks_clean_only": False,
        "looks_cj": False,
        "looks_split": False,
        "looks_strict": False,
        "looks_thruster": False,
        "looks_push_up": False,
        "looks_pull_up": False,
        "looks_handstand_push_up": False,
        "truly_explosive": False,
        "squat_confident": False,
        "deadlift_setup_geometry": False,
        "short_low_camera_bench_setup": False,
        "bodyweight_debug": {},
        "bar_debug": {},
    }
    data.update(overrides)
    return ProtectedEvidenceContext(**data)


def _early_ctx(**overrides):
    data = {
        "protected_label": None,
        "protected_conf": 0.0,
        "protected_reason": None,
        "strong_oly_lock": False,
        "bodyweight_router_label": None,
        "bodyweight_router_conf": 0.0,
        "raw_label": "squat",
        "base_conf": 0.50,
        "bio_label": "squat",
        "bio_conf": 0.50,
        "squat_label": "squat_back",
        "squat_conf": 0.50,
        "olympic_pred": None,
        "olympic_conf": 0.0,
        "run_oly_router": False,
        "explosive_score": 0.0,
        "wrist_overhead_ratio": 0.0,
        "router_v6_label": None,
        "router_v6_conf": 0.0,
        "pull_up_router_guard": False,
        "looks_cj": False,
        "looks_split": False,
        "truly_explosive": False,
        "strong_overhead": False,
        "bodyweight_debug": {},
    }
    data.update(overrides)
    return EarlyFinalContext(**data)


def _fallback_ctx(**overrides):
    data = {
        "raw_label": "squat",
        "base_conf": 0.50,
        "bio_label": "squat",
        "bio_conf": 0.50,
        "bio_override": False,
        "squat_label": "squat_back",
        "squat_conf": 0.50,
        "bar_conf": 0.0,
        "olympic_pred": None,
        "olympic_conf": 0.0,
        "run_oly_router": False,
        "explosive_score": 0.0,
        "wrist_overhead_ratio": 0.0,
        "router_v6_label": None,
        "router_v6_conf": 0.0,
        "squat_confident": False,
        "truly_explosive": False,
        "strong_overhead": False,
        "bar_says_overhead_squat": False,
        "has_real_squat_motion": False,
        "push_press_should_hold": False,
        "looks_clean_only": False,
        "looks_cj": False,
        "looks_split": False,
        "bodyweight_debug": {},
    }
    data.update(overrides)
    return FallbackFinalContext(**data)


def _router_v5_override_ctx(**overrides):
    data = {
        "final_label": "squat",
        "final_confidence": 0.60,
        "analysis_mode": "detailed_rep_analysis",
        "protected_label": None,
        "protected_confidence": 0.0,
        "protected_reason": None,
        "router_v5_label": "clean",
        "router_v5_confidence": 0.80,
        "router_v5_debug": {},
        "router_v6_label": None,
        "router_v6_confidence": 0.0,
        "raw_label": "squat",
        "base_confidence": 0.60,
        "bio_label": "squat",
        "bio_confidence": 0.60,
        "squat_label": "squat_back",
        "squat_confidence": 0.60,
        "olympic_pred": "clean",
        "olympic_confidence": 0.80,
        "explosive_score": 80.0,
        "clean_rescue_active": False,
        "upright_curl_signature": False,
        "router_v8_cj_lock": False,
        "clear_squat_should_hold": False,
        "looks_clean_only": False,
        "looks_cj": False,
        "looks_split": False,
        "truly_explosive": True,
        "bodyweight_debug": {},
    }
    data.update(overrides)
    return RouterV5OverrideContext(**data)


def _bodyweight_arbitration_ctx(**overrides):
    data = {
        "final_label": "squat_back",
        "final_confidence": 0.70,
        "analysis_mode": "squat_router_protected",
        "protected_label": None,
        "protected_confidence": 0.0,
        "protected_reason": None,
        "router_v6_label": "pull_up",
        "router_v6_confidence": 0.80,
        "bodyweight_router_label": "pull_up",
        "bodyweight_router_confidence": 0.96,
        "raw_label": "squat",
        "base_confidence": 0.50,
        "bio_label": "squat",
        "bio_confidence": 0.50,
        "squat_label": "squat_back",
        "squat_confidence": 0.70,
        "olympic_pred": None,
        "olympic_confidence": 0.0,
        "explosive_score": 0.0,
        "strong_oly_lock": False,
        "strong_bench_evidence": False,
        "credible_split_jerk": False,
        "bodyweight_debug": {},
    }
    data.update(overrides)
    return BodyweightFinalArbitrationContext(**data)


def _shape_authority_ctx(**overrides):
    data = {
        "forced_exercise_label": None,
        "final_label": "squat_back",
        "final_confidence": 0.70,
        "analysis_mode": "squat_router_protected",
        "protected_label": None,
        "protected_confidence": 0.0,
        "protected_reason": None,
        "raw_label": "squat",
        "base_confidence": 0.70,
        "bio_label": "squat",
        "bio_confidence": 0.70,
        "squat_label": "squat_back",
        "squat_confidence": 0.70,
        "router_v6_label": None,
        "router_v6_confidence": 0.0,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.80,
        "explosive_score": 100.0,
        "run_oly_router": True,
        "credible_split_jerk": False,
        "looks_cj": False,
        "looks_split": False,
        "looks_thruster": False,
        "bodyweight_debug": {},
        "router_v5_label": None,
        "router_v5_confidence": 0.0,
        "router_v5_debug": {},
    }
    data.update(overrides)
    return FinalShapeAuthorityContext(**data)


def _pre_probe_arbitration_ctx(**overrides):
    data = {
        "state": FinalDecisionState(
            final_label="squat_back",
            final_confidence=0.80,
            analysis_mode="detailed_rep_analysis",
            protected_label="squat_back",
            protected_confidence=0.80,
            protected_reason="squat_router_protected",
        ),
        "forced_exercise_label": None,
        "raw_label": "squat",
        "base_confidence": 0.96,
        "bio_label": "squat",
        "bio_confidence": 0.96,
        "squat_label": "squat_back",
        "squat_confidence": 0.90,
        "router_v6_label": "squat_back",
        "router_v6_confidence": 0.80,
        "bodyweight_router_label": None,
        "bodyweight_router_confidence": 0.0,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.84,
        "explosive_score": 120.0,
        "run_oly_router": True,
        "strong_oly_lock": False,
        "strong_bench_evidence": False,
        "credible_split_jerk": False,
        "looks_cj": True,
        "looks_split": False,
        "looks_thruster": False,
        "bodyweight_debug": {},
        "router_v5_label": None,
        "router_v5_confidence": 0.0,
        "router_v5_debug": {},
    }
    data.update(overrides)
    return FinalPreProbeArbitrationContext(**data)


def _mid_arbitration_ctx(**overrides):
    data = {
        "state": FinalDecisionState(
            final_label="clean_and_jerk",
            final_confidence=0.70,
            analysis_mode="router_v5",
            protected_label=None,
            protected_confidence=0.0,
            protected_reason=None,
        ),
        "forced_exercise_label": None,
        "raw_label": "squat",
        "base_confidence": 0.70,
        "bio_label": "squat",
        "bio_confidence": 0.70,
        "squat_label": "squat_back",
        "squat_confidence": 0.70,
        "router_v6_label": None,
        "router_v6_confidence": 0.0,
        "bodyweight_router_label": None,
        "bodyweight_router_confidence": 0.0,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.70,
        "explosive_score": 0.0,
        "wrist_overhead_ratio": 0.0,
        "looks_clean_only": False,
        "looks_cj": False,
        "looks_split": False,
        "looks_strict": False,
        "looks_thruster": False,
        "strong_front_squat_consensus": False,
        "bench_model_consensus": False,
        "pull_up_long_squat_barbell_collision": False,
        "pull_up_long_overhead_barbell_collision": False,
        "squat_knee_range": 0.0,
        "squat_hip_range": 0.0,
        "bodyweight_debug": {},
        "router_v5_label": None,
        "router_v5_debug": {},
    }
    data.update(overrides)
    return FinalMidArbitrationContext(**data)


def _tail_arbitration_ctx(**overrides):
    data = {
        "state": FinalDecisionState(
            final_label="squat_back",
            final_confidence=0.80,
            analysis_mode="squat_router_protected",
            protected_label="squat_back",
            protected_confidence=0.80,
            protected_reason=None,
        ),
        "forced_exercise_label": None,
        "raw_label": "squat",
        "base_confidence": 0.96,
        "bio_label": "squat",
        "bio_confidence": 0.96,
        "squat_label": "squat_back",
        "squat_confidence": 0.92,
        "router_v6_label": "squat_back",
        "router_v6_confidence": 0.90,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.60,
        "explosive_score": 90.0,
        "wrist_overhead_ratio": 0.10,
        "looks_cj": False,
        "looks_split": True,
        "bodyweight_debug": {
            "wrist_y_range": 0.40,
            "total_frames": 50,
        },
        "bar_debug": {},
        "router_v5_debug": {},
        "family_router_shadow": {},
        "learned_family_shadow_label": None,
        "learned_family_shadow_confidence": 0.0,
        "learned_family_shadow_trusted": False,
        "deadlift_probe": lambda state: state,
    }
    data.update(overrides)
    return FinalTailArbitrationContext(**data)


def _final_arbitration_ctx(**overrides):
    data = {
        "state": FinalDecisionState(
            final_label="squat_back",
            final_confidence=0.80,
            analysis_mode="detailed_rep_analysis",
            protected_label="squat_back",
            protected_confidence=0.80,
            protected_reason="squat_router_protected",
        ),
        "forced_exercise_label": None,
        "raw_label": "squat",
        "base_confidence": 0.96,
        "bio_label": "squat",
        "bio_confidence": 0.96,
        "squat_label": "squat_back",
        "squat_confidence": 0.90,
        "router_v6_label": "squat_back",
        "router_v6_confidence": 0.80,
        "bodyweight_router_label": None,
        "bodyweight_router_confidence": 0.0,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.84,
        "explosive_score": 120.0,
        "wrist_overhead_ratio": 0.10,
        "run_oly_router": True,
        "strong_oly_lock": False,
        "strong_bench_evidence": False,
        "credible_split_jerk": False,
        "looks_clean_only": False,
        "looks_cj": True,
        "looks_split": False,
        "looks_strict": False,
        "looks_thruster": False,
        "strong_front_squat_consensus": False,
        "bench_model_consensus": False,
        "squat_knee_range": 0.0,
        "squat_hip_range": 0.0,
        "bodyweight_debug": {},
        "bar_debug": {},
        "router_v5_label": None,
        "router_v5_confidence": 0.0,
        "router_v5_debug": {},
        "family_router_shadow": {},
        "learned_family_shadow_label": None,
        "learned_family_shadow_confidence": 0.0,
        "learned_family_shadow_trusted": False,
        "push_press_probe": lambda state: state,
        "yolo_deadlift_recovery": lambda state: state,
        "deadlift_probe": lambda state: state,
    }
    data.update(overrides)
    return FinalArbitrationContext(**data)


def _post_probe_authority_ctx(**overrides):
    data = {
        "forced_exercise_label": None,
        "final_label": "pull_up",
        "final_confidence": 0.80,
        "analysis_mode": "router_v6_bodyweight",
        "protected_label": "pull_up",
        "protected_confidence": 0.80,
        "protected_reason": None,
        "raw_label": "squat",
        "base_confidence": 0.70,
        "bio_label": "push_press",
        "bio_confidence": 0.80,
        "squat_label": "squat_back",
        "squat_confidence": 0.70,
        "router_v6_label": "pull_up",
        "olympic_pred": None,
        "olympic_confidence": 0.0,
        "explosive_score": 0.0,
        "wrist_overhead_ratio": 0.0,
        "bodyweight_router_confidence": 0.80,
        "looks_cj": False,
        "looks_split": False,
        "looks_thruster": False,
    }
    data.update(overrides)
    return FinalPostProbeAuthorityContext(**data)


def _clean_bench_pushup_ctx(**overrides):
    data = {
        "forced_exercise_label": None,
        "final_label": "clean_and_jerk",
        "final_confidence": 0.70,
        "analysis_mode": "router_v5",
        "protected_label": None,
        "protected_confidence": 0.0,
        "protected_reason": None,
        "raw_label": "squat",
        "base_confidence": 0.70,
        "bio_label": "squat",
        "bio_confidence": 0.70,
        "router_v6_label": None,
        "router_v6_confidence": 0.0,
        "bodyweight_router_label": None,
        "bodyweight_router_confidence": 0.0,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.70,
        "looks_clean_only": False,
        "looks_cj": False,
        "looks_split": False,
        "strong_front_squat_consensus": False,
        "router_v5_label": None,
        "router_v5_debug": {},
        "bodyweight_debug": {},
    }
    data.update(overrides)
    return FinalCleanBenchPushupContext(**data)


def _pullup_pushup_ctx(**overrides):
    data = {
        "forced_exercise_label": None,
        "final_label": "squat_back",
        "final_confidence": 0.70,
        "analysis_mode": "squat_router_protected",
        "protected_label": None,
        "protected_confidence": 0.0,
        "protected_reason": None,
        "raw_label": "squat",
        "bio_label": "squat",
        "base_confidence": 0.70,
        "bio_confidence": 0.70,
        "squat_label": "squat_back",
        "squat_confidence": 0.70,
        "router_v6_label": None,
        "router_v6_confidence": 0.0,
        "bodyweight_router_label": None,
        "bodyweight_router_confidence": 0.0,
        "olympic_pred": None,
        "olympic_confidence": 0.0,
        "explosive_score": 0.0,
        "bench_model_consensus": False,
        "pull_up_long_squat_barbell_collision": False,
        "pull_up_long_overhead_barbell_collision": False,
        "looks_cj": False,
        "looks_split": False,
        "looks_strict": False,
        "looks_thruster": False,
        "bodyweight_debug": {},
    }
    data.update(overrides)
    return FinalPullupPushupAuthorityContext(**data)


def _collision_ctx(**overrides):
    data = {
        "forced_exercise_label": None,
        "final_label": "clean_and_jerk",
        "final_confidence": 0.72,
        "analysis_mode": "router_v5",
        "protected_label": None,
        "protected_confidence": 0.0,
        "protected_reason": None,
        "raw_label": "squat",
        "base_confidence": 0.99,
        "bio_label": "squat",
        "bio_confidence": 0.99,
        "squat_label": "squat_front",
        "squat_confidence": 0.90,
        "router_v6_label": "squat",
        "router_v6_confidence": 0.99,
        "bodyweight_router_label": None,
        "bodyweight_router_confidence": 0.0,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.72,
        "explosive_score": 0.0,
        "bodyweight_debug": {},
    }
    data.update(overrides)
    return FinalCollisionRecoveryContext(**data)


def _press_squat_pre_ctx(**overrides):
    data = {
        "forced_exercise_label": None,
        "final_label": "clean_and_jerk",
        "final_confidence": 0.80,
        "analysis_mode": "router_v5",
        "protected_label": None,
        "protected_confidence": 0.0,
        "protected_reason": None,
        "raw_label": "push_press",
        "base_confidence": 0.96,
        "bio_label": "push_press",
        "bio_confidence": 0.96,
        "squat_label": "overhead_squat",
        "squat_confidence": 0.82,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.80,
        "explosive_score": 10.0,
        "looks_cj": False,
        "looks_split": False,
        "looks_strict": False,
        "looks_thruster": False,
        "squat_knee_range": 0.0,
        "squat_hip_range": 0.0,
        "bodyweight_debug": {},
    }
    data.update(overrides)
    return FinalPressSquatPreProbeContext(**data)


def _press_squat_post_ctx(**overrides):
    data = {
        "forced_exercise_label": None,
        "final_label": "split_jerk",
        "final_confidence": 0.75,
        "analysis_mode": "router_v5",
        "protected_label": None,
        "protected_confidence": 0.0,
        "protected_reason": None,
        "raw_label": "push_press",
        "base_confidence": 0.80,
        "bio_label": "push_press",
        "bio_confidence": 0.80,
        "squat_label": "squat_back",
        "squat_confidence": 0.80,
        "router_v6_label": "push_press",
        "router_v6_confidence": 0.80,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.60,
        "explosive_score": 80.0,
        "wrist_overhead_ratio": 0.0,
        "looks_cj": False,
        "bodyweight_debug": {},
        "bar_debug": {},
    }
    data.update(overrides)
    return FinalPressSquatPostProbeContext(**data)


def _clean_olympic_ctx(**overrides):
    data = {
        "forced_exercise_label": None,
        "final_label": "squat_back",
        "final_confidence": 0.80,
        "analysis_mode": "squat_router_protected",
        "protected_label": None,
        "protected_confidence": 0.0,
        "protected_reason": None,
        "raw_label": "squat",
        "base_confidence": 0.96,
        "bio_label": "squat",
        "bio_confidence": 0.96,
        "squat_label": "squat_back",
        "squat_confidence": 0.92,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.60,
        "explosive_score": 90.0,
        "wrist_overhead_ratio": 0.10,
        "looks_cj": False,
        "looks_split": True,
        "bodyweight_debug": {
            "wrist_y_range": 0.40,
            "total_frames": 50,
        },
        "router_v5_debug": {},
        "family_router_shadow": {},
        "learned_family_shadow_label": None,
        "learned_family_shadow_confidence": 0.0,
        "learned_family_shadow_trusted": False,
        "apply_segment_rules": True,
        "apply_olympic_authority": False,
    }
    data.update(overrides)
    return FinalCleanOlympicAuthorityContext(**data)


def _deadlift_probe_ctx(**overrides):
    data = {
        "forced_exercise_label": None,
        "final_label": "squat_back",
        "final_confidence": 0.90,
        "analysis_mode": "squat_router_protected",
        "protected_label": "squat_back",
        "protected_confidence": 0.90,
        "protected_reason": None,
        "raw_label": "squat",
        "base_confidence": 0.96,
        "bio_label": "squat",
        "bio_confidence": 0.96,
        "squat_label": "squat_back",
        "squat_confidence": 0.92,
        "router_v6_label": "squat_back",
        "router_v6_confidence": 0.90,
        "wrist_overhead_ratio": 0.01,
        "explosive_score": 40.0,
        "deadlift_knee_range": 0.0,
        "deadlift_hip_range": 0.0,
        "deadlift_torso_range": 0.0,
        "bodyweight_debug": {
            "avg_elbow": 170.0,
            "elbow_range": 20.0,
            "avg_torso_angle": 25.0,
            "wrist_y_range": 0.30,
            "total_frames": 150,
        },
        "bar_debug": {
            "front_rack_elbow_p25": 170.0,
            "wrist_height_above_shoulder": -0.20,
        },
    }
    data.update(overrides)
    return FinalDeadliftProbeContext(**data)


def _bench05_bodyweight_debug():
    return {
        "avg_elbow": 132.78941345214844,
        "avg_torso_angle": 73.81417846679688,
        "elbow_range": 171.68922424316406,
        "hip_y_range": 0.03856884688138962,
        "min_elbow": 4.319302558898926,
        "shoulder_y_range": 0.043359190225601196,
        "total_frames": 52,
        "wrist_above_shoulder_ratio": 0.19230769230769232,
        "wrist_y_range": 0.19166655838489532,
    }


def _push_press_probe_ctx(**overrides):
    data = {
        "forced_exercise_label": None,
        "final_label": "split_jerk",
        "final_confidence": 0.85,
        "analysis_mode": "router_v5",
        "protected_label": "split_jerk",
        "protected_confidence": 0.85,
        "protected_reason": "standalone_split_from_cj",
        "raw_label": "push_press",
        "base_confidence": 0.99,
        "bio_label": "push_press",
        "bio_confidence": 0.99,
        "squat_label": "overhead_squat",
        "squat_confidence": 0.82,
        "bodyweight_router_label": None,
        "bodyweight_router_confidence": 0.0,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.82,
        "explosive_score": 15.0,
        "bar_debug": {
            "overhead_ratio": 0.85,
            "total_frames": 300,
        },
    }
    data.update(overrides)
    return FinalPushPressProbeContext(**data)


def _yolo_deadlift_ctx(**overrides):
    data = {
        "use_yolo_tracking": True,
        "forced_exercise_label": None,
        "final_label": "squat_front",
        "final_confidence": 0.80,
        "analysis_mode": "detailed_rep_analysis",
        "protected_label": "squat_front",
        "protected_confidence": 0.80,
        "protected_reason": "squat_router_protected",
        "raw_label": "squat_front",
        "bio_label": "squat",
        "olympic_confidence": 0.60,
    }
    data.update(overrides)
    return YoloDeadliftRecoveryContext(**data)


def test_final_deadlift_probe_plans_squat_disagreement_probe():
    plan = plan_final_deadlift_probe(_deadlift_probe_ctx())

    assert plan.should_probe is True
    assert plan.recovery_reason == "deadlift_analyzer_disagreement_rescue"


def test_final_deadlift_probe_plans_bench_hinge_probe():
    plan = plan_final_deadlift_probe(
        _deadlift_probe_ctx(
            final_label="bench_press",
            raw_label="bench_press",
            bio_label="bench_press",
            router_v6_label="bench_press",
            router_v6_confidence=0.96,
            deadlift_knee_range=90.0,
            deadlift_hip_range=90.0,
            deadlift_torso_range=30.0,
            bodyweight_debug={
                "elbow_range": 10.0,
                "min_elbow": 160.0,
                "avg_elbow": 170.0,
                "avg_torso_angle": 25.0,
            },
        )
    )

    assert plan.should_probe is True
    assert plan.recovery_reason == "straight_arm_hinge_deadlift_recovery"


def test_final_deadlift_probe_applies_confirmed_reps():
    decision = apply_final_deadlift_probe_result(
        _deadlift_probe_ctx(),
        probe_reps=[{"rep": 1}],
        recovery_reason="deadlift_analyzer_disagreement_rescue",
    )

    assert decision.final_label == "deadlift"
    assert decision.final_confidence == 0.96
    assert decision.protected_reason == "deadlift_analyzer_disagreement_rescue"


def test_final_deadlift_probe_does_not_apply_without_reps():
    decision = apply_final_deadlift_probe_result(
        _deadlift_probe_ctx(),
        probe_reps=[],
        recovery_reason="deadlift_analyzer_disagreement_rescue",
    )

    assert decision.final_label == "squat_back"
    assert decision.final_confidence == 0.90


def test_final_collision_recovers_short_bench_from_push_up():
    decision = select_final_collision_recovery(
        _collision_ctx(
            final_label="push_up",
            raw_label="squat_front",
            base_confidence=0.993,
            bio_label="deadlift",
            bio_confidence=0.993,
            squat_label="squat_front",
            squat_confidence=0.97,
            router_v6_label="squat_front",
            router_v6_confidence=0.99,
            olympic_confidence=0.688,
            bodyweight_debug=_bench05_bodyweight_debug(),
        )
    )

    assert decision.final_label == "bench_press"
    assert decision.final_confidence == 0.86
    assert decision.protected_reason == "short_bench_over_pushup_final_recovery"


def test_final_push_press_probe_plans_split_jerk_rescue():
    plan = plan_final_push_press_probe(_push_press_probe_ctx())

    assert plan.should_probe is True
    assert plan.minimum_rep_count == 3
    assert plan.recovery_reason == "push_press_analyzer_over_split_authority"


def test_final_push_press_probe_plans_pull_up_rescue():
    plan = plan_final_push_press_probe(
        _push_press_probe_ctx(
            final_label="pull_up",
            protected_label="pull_up",
            protected_reason="pull_up_final_authority",
            squat_confidence=0.90,
            bodyweight_router_label="pull_up",
            bodyweight_router_confidence=0.96,
            olympic_confidence=0.60,
            explosive_score=20.0,
        )
    )

    assert plan.should_probe is True
    assert plan.minimum_rep_count == 2
    assert plan.recovery_reason == "push_press_analyzer_over_pull_up_authority"


def test_final_push_press_probe_applies_confirmed_reps():
    decision = apply_final_push_press_probe_result(
        _push_press_probe_ctx(),
        probe_reps=[{"rep": 1}, {"rep": 2}, {"rep": 3}],
        minimum_rep_count=3,
        recovery_reason="push_press_analyzer_over_split_authority",
    )

    assert decision.final_label == "push_press"
    assert decision.final_confidence == 0.99
    assert decision.protected_reason == "push_press_analyzer_over_split_authority"


def test_final_push_press_probe_does_not_apply_without_enough_reps():
    decision = apply_final_push_press_probe_result(
        _push_press_probe_ctx(),
        probe_reps=[{"rep": 1}, {"rep": 2}],
        minimum_rep_count=3,
        recovery_reason="push_press_analyzer_over_split_authority",
    )

    assert decision.final_label == "split_jerk"
    assert decision.final_confidence == 0.85


def test_yolo_deadlift_recovery_plans_busy_scene_probe():
    plan = plan_yolo_deadlift_recovery(_yolo_deadlift_ctx())

    assert plan.should_probe is True
    assert plan.squat_probe_label == "squat_front"


def test_yolo_deadlift_recovery_normalizes_generic_squat_probe_label():
    plan = plan_yolo_deadlift_recovery(
        _yolo_deadlift_ctx(final_label="squat", raw_label="squat")
    )

    assert plan.should_probe is True
    assert plan.squat_probe_label == "squat_back"


def test_yolo_deadlift_recovery_applies_rep_count_disagreement():
    decision = apply_yolo_deadlift_recovery_result(
        _yolo_deadlift_ctx(),
        deadlift_probe_reps=[{"rep": 1}, {"rep": 2}],
        squat_probe_reps=[
            {"rep": 1},
            {"rep": 2},
            {"rep": 3},
            {"rep": 4},
            {"rep": 5},
            {"rep": 6},
            {"rep": 7},
        ],
    )

    assert decision.final_label == "deadlift"
    assert decision.final_confidence == 0.86
    assert decision.protected_reason == "yolo_rep_count_deadlift_recovery"


def test_yolo_deadlift_recovery_does_not_apply_without_large_disagreement():
    decision = apply_yolo_deadlift_recovery_result(
        _yolo_deadlift_ctx(),
        deadlift_probe_reps=[{"rep": 1}, {"rep": 2}],
        squat_probe_reps=[{"rep": 1}, {"rep": 2}, {"rep": 3}],
    )

    assert decision.final_label == "squat_front"
    assert decision.final_confidence == 0.80


def test_final_clean_olympic_recovers_short_clean_segment():
    decision = select_final_clean_olympic_authority(_clean_olympic_ctx())

    assert decision.final_label == "clean"
    assert decision.final_confidence == 0.75
    assert decision.protected_reason == "short_explosive_clean_segment"


def test_final_clean_olympic_recovers_squat_protected_clean_segment():
    decision = select_final_clean_olympic_authority(
        _clean_olympic_ctx(
            final_label="squat_back",
            analysis_mode="squat_router_protected",
            olympic_pred="clean",
            olympic_confidence=0.55,
            wrist_overhead_ratio=0.05,
            looks_cj=False,
            family_router_shadow={
                "family": "olympic",
                "margin": 0.524,
            },
            learned_family_shadow_label="olympic",
            learned_family_shadow_confidence=0.772,
            learned_family_shadow_trusted=True,
        )
    )

    assert decision.final_label == "clean"
    assert decision.final_confidence == 0.772
    assert decision.analysis_mode == "shape_override"
    assert (
        decision.protected_reason
        == "squat_protected_clean_segment_recovery"
    )


def test_final_clean_olympic_recovers_clean_segment_with_trusted_family():
    decision = select_final_clean_olympic_authority(
        _clean_olympic_ctx(
            final_label="squat_back",
            analysis_mode="squat_router_protected",
            olympic_pred="clean",
            olympic_confidence=0.25,
            wrist_overhead_ratio=0.05,
            looks_cj=False,
            learned_family_shadow_label="olympic",
            learned_family_shadow_confidence=0.772,
            learned_family_shadow_trusted=True,
        )
    )

    assert decision.final_label == "clean"
    assert decision.final_confidence == 0.772
    assert decision.analysis_mode == "shape_override"
    assert (
        decision.protected_reason
        == "squat_protected_clean_segment_recovery"
    )


def test_final_clean_olympic_recovers_segmented_clean_from_weak_cj():
    decision = select_final_clean_olympic_authority(
        _clean_olympic_ctx(
            final_label="clean_and_jerk",
            squat_label="squat_front",
            squat_confidence=0.50,
            olympic_confidence=0.55,
            explosive_score=50.0,
            wrist_overhead_ratio=0.30,
            looks_cj=True,
            bodyweight_debug={"total_frames": 100},
            router_v5_debug={
                "features": {
                    "catch_overhead": 0.0,
                    "extension_to_catch": 5.0,
                    "catch_to_finish": 60.0,
                    "lockout_duration": 30.0,
                }
            },
        )
    )

    assert decision.final_label == "clean"
    assert decision.protected_reason == "segmented_clean_from_weak_cj"


def test_final_clean_olympic_recovers_compact_clean_and_jerk():
    decision = select_final_clean_olympic_authority(
        _clean_olympic_ctx(
            bio_label="push_press",
            bio_confidence=0.80,
            olympic_confidence=0.78,
            wrist_overhead_ratio=0.70,
            bodyweight_debug={"total_frames": 120},
            router_v5_debug={
                "decision": "agreement",
                "features": {
                    "catch_to_finish": 60.0,
                    "lockout_duration": 50.0,
                },
            },
        )
    )

    assert decision.final_label == "clean_and_jerk"
    assert decision.final_confidence == 0.78
    assert decision.protected_reason == "compact_clean_and_jerk_final_authority"


def test_final_clean_olympic_preserves_extra_snatch_authority_case():
    decision = select_final_clean_olympic_authority(
        _clean_olympic_ctx(
            final_label="squat_back",
            raw_label="squat",
            bio_label="push_press",
            olympic_pred="snatch",
            olympic_confidence=0.55,
            explosive_score=20.0,
            apply_segment_rules=False,
            apply_olympic_authority=True,
        )
    )

    assert decision.final_label == "snatch"
    assert decision.final_confidence == 0.75
    assert decision.analysis_mode == "olympic_final_authority"
    assert decision.protected_reason == "specialized_olympic_final_authority"


def test_final_clean_olympic_recovers_learned_snatch_over_overhead_squat():
    decision = select_final_clean_olympic_authority(
        _clean_olympic_ctx(
            final_label="overhead_squat",
            final_confidence=0.811,
            analysis_mode="detailed_rep_analysis",
            raw_label="deadlift",
            base_confidence=0.759,
            bio_label="push_press",
            bio_confidence=0.78,
            squat_label="overhead_squat",
            squat_confidence=0.811,
            olympic_pred="clean_and_jerk",
            olympic_confidence=0.436,
            wrist_overhead_ratio=0.698,
            explosive_score=13.37,
            looks_cj=True,
            looks_split=True,
            learned_family_shadow_label="olympic",
            learned_family_shadow_confidence=0.848,
            learned_family_shadow_trusted=True,
            apply_segment_rules=False,
            apply_olympic_authority=True,
        )
    )

    assert decision.final_label == "snatch"
    assert decision.final_confidence == 0.75
    assert decision.analysis_mode == "olympic_final_authority"


def test_final_press_squat_pre_recovers_controlled_overhead_squat():
    decision = select_final_press_squat_pre_probe_authority(
        _press_squat_pre_ctx(
            bodyweight_debug={
                "wrist_above_shoulder_ratio": 0.90,
                "avg_elbow": 160.0,
                "min_elbow": 40.0,
                "total_frames": 220,
            }
        )
    )

    assert decision.final_label == "overhead_squat"
    assert decision.final_confidence == 0.82
    assert decision.protected_reason == (
        "controlled_overhead_squat_final_recovery"
    )


def test_final_press_squat_pre_recovers_strict_press():
    decision = select_final_press_squat_pre_probe_authority(
        _press_squat_pre_ctx(
            final_label="push_press",
            looks_strict=True,
            squat_knee_range=10.0,
            squat_hip_range=5.0,
        )
    )

    assert decision.final_label == "strict_press"
    assert decision.final_confidence == 0.86
    assert decision.analysis_mode == "shape_override"
    assert decision.protected_reason == (
        "strict_press_low_leg_drive_final_authority"
    )


def test_final_press_squat_post_recovers_push_press_over_weak_cj_split():
    decision = select_final_press_squat_post_probe_authority(
        _press_squat_post_ctx(
            bodyweight_debug={
                "wrist_y_range": 0.60,
                "shoulder_y_range": 0.10,
            }
        )
    )

    assert decision.final_label == "push_press"
    assert decision.final_confidence == 0.86
    assert decision.protected_reason == "push_press_over_weak_cj_split"


def test_final_press_squat_post_recovers_push_press_over_back_squat():
    decision = select_final_press_squat_post_probe_authority(
        _press_squat_post_ctx(
            final_label="squat_back",
            raw_label="squat",
            base_confidence=0.70,
            bio_label="squat",
            bio_confidence=0.70,
            squat_confidence=0.92,
            olympic_pred="split_jerk",
            olympic_confidence=0.85,
            explosive_score=10.0,
            bodyweight_debug={
                "shoulder_y_range": 0.10,
                "hip_y_range": 0.10,
                "wrist_y_range": 0.30,
            },
        )
    )

    assert decision.final_label == "push_press"
    assert decision.final_confidence == 0.86
    assert decision.protected_reason == (
        "low_motion_push_press_over_back_squat"
    )


def test_final_press_squat_post_recovers_sustained_overhead_squat():
    decision = select_final_press_squat_post_probe_authority(
        _press_squat_post_ctx(
            final_label="clean_and_jerk",
            squat_label="overhead_squat",
            squat_confidence=0.84,
            olympic_confidence=0.80,
            wrist_overhead_ratio=0.95,
            bodyweight_debug={"total_frames": 320},
        )
    )

    assert decision.final_label == "overhead_squat"
    assert decision.final_confidence == 0.84
    assert decision.protected_reason == (
        "sustained_overhead_squat_final_authority"
    )


def test_final_collision_recovery_recovers_long_squat_over_cj():
    decision = select_final_collision_recovery(
        _collision_ctx(
            bodyweight_debug={
                "avg_torso_angle": 2.0,
                "wrist_above_shoulder_ratio": 0.10,
                "shoulder_y_range": 0.60,
                "hip_y_range": 0.45,
                "elbow_range": 150.0,
                "avg_elbow": 55.0,
                "total_frames": 600,
            }
        )
    )

    assert decision.final_label == "squat_back"
    assert decision.final_confidence == 0.90
    assert decision.protected_reason == "long_squat_over_cj_final_recovery"


def test_final_collision_recovery_recovers_long_deadlift_over_squat():
    decision = select_final_collision_recovery(
        _collision_ctx(
            final_label="squat_back",
            raw_label="push_press",
            base_confidence=0.86,
            bio_label="squat",
            bio_confidence=0.86,
            squat_label="squat_back",
            squat_confidence=0.97,
            router_v6_label="squat_back",
            router_v6_confidence=0.60,
            olympic_pred="snatch",
            olympic_confidence=0.60,
            bodyweight_debug={
                "avg_torso_angle": 25.0,
                "wrist_y_range": 0.35,
                "shoulder_y_range": 0.35,
                "hip_y_range": 0.15,
                "elbow_range": 45.0,
                "avg_elbow": 175.0,
                "min_elbow": 130.0,
                "wrist_above_shoulder_ratio": 0.01,
                "total_frames": 900,
            },
        )
    )

    assert decision.final_label == "deadlift"
    assert decision.final_confidence == 0.82
    assert decision.protected_reason == "long_deadlift_over_squat_final_recovery"


def test_final_collision_recovery_recovers_short_deadlift_over_snatch():
    decision = select_final_collision_recovery(
        _collision_ctx(
            final_label="snatch",
            raw_label="deadlift",
            base_confidence=0.90,
            bio_label="squat",
            bodyweight_router_label="push_up",
            bodyweight_router_confidence=0.90,
            router_v6_label="push_up",
            explosive_score=90.0,
            bodyweight_debug={
                "avg_torso_angle": 65.0,
                "wrist_above_shoulder_ratio": 0.01,
                "elbow_range": 20.0,
                "min_elbow": 160.0,
                "wrist_y_range": 0.20,
                "hip_y_range": 0.10,
                "total_frames": 60,
            },
        )
    )

    assert decision.final_label == "deadlift"
    assert decision.final_confidence == 0.90
    assert decision.protected_reason == "short_deadlift_over_snatch_final_recovery"


def test_final_collision_recovery_recovers_long_snatch_over_squat():
    decision = select_final_collision_recovery(
        _collision_ctx(
            final_label="squat_back",
            raw_label="squat",
            base_confidence=0.50,
            bio_label="squat",
            bio_confidence=0.50,
            squat_label="squat_back",
            squat_confidence=0.94,
            router_v6_label="squat_back",
            router_v6_confidence=0.60,
            olympic_pred="clean_and_jerk",
            olympic_confidence=0.58,
            explosive_score=140.0,
            bodyweight_debug={
                "wrist_y_range": 0.70,
                "shoulder_y_range": 0.30,
                "hip_y_range": 0.35,
                "elbow_range": 170.0,
                "avg_elbow": 160.0,
                "min_elbow": 5.0,
                "wrist_above_shoulder_ratio": 0.50,
                "total_frames": 1300,
            },
        )
    )

    assert decision.final_label == "snatch"
    assert decision.final_confidence == 0.70
    assert decision.protected_reason == "long_snatch_over_squat_final_recovery"


def test_final_pullup_pushup_recovers_horizontal_push_up():
    decision = select_final_pullup_pushup_authority(
        _pullup_pushup_ctx(
            final_label="clean_and_jerk",
            raw_label="deadlift",
            bio_label="squat",
            bodyweight_router_label="push_up",
            bodyweight_router_confidence=0.80,
            bodyweight_debug={
                "avg_torso_angle": 170.0,
                "shoulder_y_range": 0.35,
                "hip_y_range": 0.25,
                "elbow_range": 160.0,
                "wrist_above_shoulder_ratio": 0.10,
            },
        )
    )

    assert decision.final_label == "push_up"
    assert decision.final_confidence == 0.86
    assert decision.protected_reason == "horizontal_push_up_final_recovery"


def test_final_pullup_pushup_recovers_low_motion_pull_up():
    decision = select_final_pullup_pushup_authority(
        _pullup_pushup_ctx(
            final_label="push_press",
            raw_label="push_press",
            bio_label="push_press",
            router_v6_label="push_press",
            bodyweight_router_label="pull_up",
            bodyweight_router_confidence=0.88,
            protected_reason="push_press_pattern_detected",
            bodyweight_debug={
                "wrist_above_shoulder_ratio": 0.01,
                "mean_wrist_minus_shoulder_y": 0.25,
                "elbow_range": 20.0,
                "min_elbow": 150.0,
                "shoulder_y_range": 0.02,
                "hip_y_range": 0.02,
                "avg_wrist_forward": 0.06,
                "total_frames": 80,
            },
        )
    )

    assert decision.final_label == "pull_up"
    assert decision.final_confidence == 0.88
    assert decision.protected_reason == "low_motion_pull_up_final_recovery"


def test_final_pullup_pushup_applies_pull_up_authority():
    decision = select_final_pullup_pushup_authority(
        _pullup_pushup_ctx(
            final_label="snatch",
            router_v6_label="pull_up",
            router_v6_confidence=0.82,
            bodyweight_router_label="pull_up",
            bodyweight_router_confidence=0.95,
        )
    )

    assert decision.final_label == "pull_up"
    assert decision.final_confidence == 0.95
    assert decision.protected_reason == "pull_up_final_authority"


def test_final_pullup_pushup_rejects_barbell_path_false_pull_up():
    decision = select_final_pullup_pushup_authority(
        _pullup_pushup_ctx(
            final_label="overhead_squat",
            final_confidence=0.84,
            raw_label="push_press",
            bio_label="push_press",
            squat_label="overhead_squat",
            squat_confidence=0.82,
            olympic_pred="clean_and_jerk",
            router_v6_label="pull_up",
            router_v6_confidence=0.91,
            bodyweight_router_label="pull_up",
            bodyweight_router_confidence=0.99,
            bodyweight_debug={
                "total_frames": 255,
                "wrist_above_shoulder_ratio": 0.886,
                "mean_wrist_minus_shoulder_y": -0.099,
                "wrist_y_range": 0.357,
                "shoulder_y_range": 0.218,
            },
        )
    )

    assert decision.final_label == "overhead_squat"
    assert decision.protected_reason is None


def test_final_pullup_pushup_preserves_protected_push_press():
    decision = select_final_pullup_pushup_authority(
        _pullup_pushup_ctx(
            final_label="push_press",
            protected_reason="push_press_pattern_detected",
            router_v6_label="pull_up",
            router_v6_confidence=0.82,
            bodyweight_router_label="pull_up",
            bodyweight_router_confidence=0.95,
        )
    )

    assert decision.final_label == "push_press"
    assert decision.protected_reason == "push_press_pattern_detected"


def test_final_clean_bench_pushup_recovers_clean_only_shape():
    decision = select_final_clean_bench_pushup_authority(
        _clean_bench_pushup_ctx(looks_clean_only=True)
    )

    assert decision.final_label == "clean"
    assert decision.final_confidence == 0.75
    assert decision.analysis_mode == "shape_override"
    assert decision.protected_reason == "clean_only_shape_final_authority"


def test_final_clean_bench_pushup_recovers_short_horizontal_bench():
    decision = select_final_clean_bench_pushup_authority(
        _clean_bench_pushup_ctx(
            final_label="push_press",
            final_confidence=0.78,
            raw_label="push_press",
            base_confidence=0.80,
            bio_label="push_press",
            bio_confidence=0.82,
            router_v6_label="push_press",
            protected_reason="push_press_pattern_detected",
            bodyweight_debug={
                "wrist_above_shoulder_ratio": 0.01,
                "avg_torso_angle": 10.0,
                "elbow_range": 160.0,
                "wrist_y_range": 0.18,
                "shoulder_y_range": 0.20,
                "hip_y_range": 0.20,
                "total_frames": 40,
            },
        )
    )

    assert decision.final_label == "bench_press"
    assert decision.final_confidence == 0.86
    assert decision.protected_reason == "short_horizontal_bench_final_recovery"


def test_final_clean_bench_pushup_recovers_push_up_authority():
    decision = select_final_clean_bench_pushup_authority(
        _clean_bench_pushup_ctx(
            final_label="clean",
            bodyweight_router_label="push_up",
            bodyweight_router_confidence=0.90,
            router_v6_label="push_up",
            router_v6_confidence=0.70,
            bodyweight_debug={
                "wrist_above_shoulder_ratio": 0.05,
            },
        )
    )

    assert decision.final_label == "push_up"
    assert decision.final_confidence == 0.90
    assert decision.analysis_mode == "router_v6_bodyweight"
    assert decision.protected_reason == "push_up_final_authority"


def test_final_post_probe_authority_accepts_high_conf_cj():
    decision = select_final_post_probe_authority(
        _post_probe_authority_ctx(
            final_label="split_jerk",
            raw_label="push_press",
            bio_label="push_press",
            olympic_pred="clean_and_jerk",
            olympic_confidence=0.98,
        )
    )

    assert decision.final_label == "clean_and_jerk"
    assert decision.final_confidence == 0.98
    assert decision.protected_reason == (
        "clean_and_jerk_high_conf_final_authority"
    )


def test_final_post_probe_authority_recovers_push_press_from_false_cj():
    decision = select_final_post_probe_authority(
        _post_probe_authority_ctx(
            final_label="clean_and_jerk",
            raw_label="push_press",
            base_confidence=0.50,
            bio_label="squat",
            squat_confidence=0.50,
            olympic_pred="clean_and_jerk",
            olympic_confidence=0.80,
            looks_thruster=True,
        )
    )

    assert decision.final_label == "push_press"
    assert decision.final_confidence == 0.76
    assert decision.analysis_mode == "detailed_rep_analysis"
    assert decision.protected_reason == "push_press_from_false_cj_agreement"


def test_final_post_probe_authority_recovers_ring_muscle_up():
    decision = select_final_post_probe_authority(
        _post_probe_authority_ctx(
            final_label="pull_up",
            raw_label="squat",
            bio_label="push_press",
            bio_confidence=0.80,
            router_v6_label="pull_up",
            bodyweight_router_confidence=0.98,
            explosive_score=95.0,
            wrist_overhead_ratio=0.70,
            olympic_confidence=0.60,
        )
    )

    assert decision.final_label == "muscle_up"
    assert decision.final_confidence == 0.98
    assert decision.protected_reason == "ring_muscle_up_final_recovery"


def test_final_post_probe_authority_recovers_bar_muscle_up():
    decision = select_final_post_probe_authority(
        _post_probe_authority_ctx(
            final_label="pull_up",
            raw_label="squat_front",
            base_confidence=0.96,
            bio_label="squat_front",
            bio_confidence=0.97,
            squat_label="overhead_squat",
            squat_confidence=0.92,
            router_v6_label="pull_up",
            bodyweight_router_confidence=0.95,
            explosive_score=20.0,
            wrist_overhead_ratio=0.50,
        )
    )

    assert decision.final_label == "muscle_up"
    assert decision.final_confidence == 0.95
    assert decision.protected_reason == "bar_muscle_up_final_recovery"


def test_final_shape_authority_recovers_clean_and_jerk_shape():
    decision = select_final_shape_authority(
        _shape_authority_ctx(looks_cj=True)
    )

    assert decision.final_label == "clean_and_jerk"
    assert decision.final_confidence == 0.86
    assert decision.protected_reason == "clean_and_jerk_shape_final_recovery"


def test_final_shape_authority_recovers_standalone_split_shape():
    decision = select_final_shape_authority(
        _shape_authority_ctx(
            final_label="bench_press",
            credible_split_jerk=True,
            olympic_pred="split_jerk",
            olympic_confidence=0.83,
            explosive_score=40.0,
        )
    )

    assert decision.final_label == "split_jerk"
    assert decision.final_confidence == 0.83
    assert decision.protected_reason == "standalone_split_shape_recovery"


def test_final_shape_authority_recovers_low_explosive_push_press():
    decision = select_final_shape_authority(
        _shape_authority_ctx(
            final_label="squat_back",
            raw_label="squat_front",
            base_confidence=0.99,
            bio_label="push_press",
            bio_confidence=0.99,
            credible_split_jerk=True,
            olympic_confidence=0.82,
            explosive_score=10.0,
            looks_split=True,
        )
    )

    assert decision.final_label == "push_press"
    assert decision.final_confidence == 0.99
    assert decision.analysis_mode == "biomechanics_override"
    assert decision.protected_reason == "low_explosive_push_press_over_split"


def test_final_pre_probe_arbitration_runs_bodyweight_then_shape_authority():
    decision = select_final_pre_probe_arbitration(
        _pre_probe_arbitration_ctx()
    )

    assert decision.state.final_label == "clean_and_jerk"
    assert decision.state.final_confidence == 0.86
    assert (
        decision.state.protected_reason
        == "clean_and_jerk_shape_final_recovery"
    )
    assert decision.pull_up_long_squat_barbell_collision is False
    assert decision.pull_up_long_overhead_barbell_collision is False


def test_final_mid_arbitration_runs_clean_bench_pushup_authority():
    decision = select_final_mid_arbitration(
        _mid_arbitration_ctx(
            looks_clean_only=True,
            olympic_confidence=0.70,
        )
    )

    assert decision.state.final_label == "clean"
    assert decision.state.final_confidence == 0.75
    assert decision.state.protected_reason == "clean_only_shape_final_authority"


def test_final_tail_arbitration_runs_clean_segment_authority():
    decision = select_final_tail_arbitration(_tail_arbitration_ctx())

    assert decision.state.final_label == "clean"
    assert decision.state.final_confidence == 0.75
    assert decision.state.protected_reason == "short_explosive_clean_segment"


def test_run_final_arbitration_runs_full_sequence():
    decision = run_final_arbitration(_final_arbitration_ctx())

    assert decision.state.final_label == "clean_and_jerk"
    assert decision.state.final_confidence == 0.84
    assert (
        decision.state.protected_reason
        == "specialized_olympic_final_authority"
    )
    assert decision.pull_up_long_squat_barbell_collision is False
    assert decision.pull_up_long_overhead_barbell_collision is False


def test_bodyweight_arbitration_accepts_matching_high_conf_router():
    decision = select_bodyweight_final_arbitration(
        _bodyweight_arbitration_ctx(
            router_v6_label="handstand_push_up",
            router_v6_confidence=0.82,
            bodyweight_router_label="handstand_push_up",
            bodyweight_router_confidence=0.97,
        )
    )

    assert decision.allowed is True
    assert decision.final_label == "handstand_push_up"
    assert decision.final_confidence == 0.97
    assert decision.analysis_mode == "router_v6_bodyweight"
    assert decision.protected_reason == "router_v6_bodyweight_winner"


def test_bodyweight_arbitration_rejects_bench_pull_up_collision():
    decision = select_bodyweight_final_arbitration(
        _bodyweight_arbitration_ctx(
            router_v6_label="pull_up",
            bodyweight_router_label="pull_up",
            raw_label="bench_press",
            base_confidence=0.80,
            bio_label="bench_press",
            bio_confidence=0.82,
        )
    )

    assert decision.allowed is False
    assert decision.final_label == "squat_back"


def test_bodyweight_arbitration_rejects_barbell_path_pull_up_collision():
    decision = select_bodyweight_final_arbitration(
        _bodyweight_arbitration_ctx(
            final_label="squat_back",
            router_v6_label="pull_up",
            router_v6_confidence=0.94,
            bodyweight_router_label="pull_up",
            bodyweight_router_confidence=0.99,
            raw_label="squat",
            bio_label="push_press",
            squat_label="squat_back",
            squat_confidence=0.70,
            olympic_pred="split_jerk",
            olympic_confidence=0.62,
            bodyweight_debug={
                "total_frames": 450,
                "wrist_above_shoulder_ratio": 0.969,
                "mean_wrist_minus_shoulder_y": -0.033,
                "wrist_y_range": 0.274,
                "shoulder_y_range": 0.241,
            },
        )
    )

    assert decision.allowed is False
    assert decision.final_label == "squat_back"
    assert decision.pull_up_long_squat_barbell_collision is True


def test_bodyweight_arbitration_keeps_true_pull_up_shape():
    decision = select_bodyweight_final_arbitration(
        _bodyweight_arbitration_ctx(
            final_label="squat_back",
            router_v6_label="pull_up",
            router_v6_confidence=0.94,
            bodyweight_router_label="pull_up",
            bodyweight_router_confidence=0.99,
            raw_label="squat",
            bio_label="squat",
            squat_label="squat_back",
            bodyweight_debug={
                "total_frames": 176,
                "wrist_above_shoulder_ratio": 0.972,
                "mean_wrist_minus_shoulder_y": -0.178,
                "wrist_y_range": 0.058,
                "shoulder_y_range": 0.242,
            },
        )
    )

    assert decision.allowed is True
    assert decision.final_label == "pull_up"


def test_bodyweight_arbitration_rejects_strict_press_protection():
    decision = select_bodyweight_final_arbitration(
        _bodyweight_arbitration_ctx(
            protected_label="strict_press",
            protected_confidence=0.95,
            protected_reason="strict_press_pattern_detected",
        )
    )

    assert decision.allowed is False
    assert decision.protected_label == "strict_press"


def test_router_v5_override_applies_olympic_router_label():
    decision = select_router_v5_override(
        _router_v5_override_ctx(
            final_label="squat",
            router_v5_label="clean",
            router_v5_confidence=0.82,
            protected_label=None,
        )
    )

    assert decision.final_label == "clean"
    assert decision.final_confidence == 0.82
    assert decision.analysis_mode == "router_v5"


def test_router_v5_override_protects_clear_squat_from_weak_oly():
    decision = select_router_v5_override(
        _router_v5_override_ctx(
            raw_label="squat",
            base_confidence=0.91,
            squat_label="squat_back",
            squat_confidence=0.94,
            router_v5_label="clean_and_jerk",
            router_v5_confidence=0.70,
        )
    )

    assert decision.final_label == "squat_back"
    assert decision.final_confidence == 0.94
    assert decision.analysis_mode == "squat_router_protected"


def test_router_v5_override_rejects_upright_curl_false_olympic():
    decision = select_router_v5_override(
        _router_v5_override_ctx(
            router_v5_label="clean_and_jerk",
            router_v5_confidence=0.82,
            router_v5_debug={},
            upright_curl_signature=True,
        )
    )

    assert decision.final_label == "unknown"
    assert decision.final_confidence == 0.50
    assert decision.analysis_mode == "insufficient_signal"
    assert decision.protected_label is None
    assert decision.router_v5_debug["decision"] == (
        "rejected_upright_curl_signature"
    )


def test_protected_evidence_prefers_bench_model_consensus():
    decision = select_protected_evidence(
        _ctx(
            raw_label="bench_press",
            base_conf=0.84,
            bio_label="bench_press",
            bio_conf=0.85,
        )
    )

    assert decision.label == "bench_press"
    assert decision.confidence == 0.95
    assert decision.reason == "bench_model_consensus"
    assert decision.bench_model_consensus is True


def test_protected_evidence_blocks_false_pull_up_on_snatch_shape():
    decision = select_protected_evidence(
        _ctx(
            raw_label="squat",
            bio_label="push_press",
            olympic_pred="snatch",
            olympic_conf=0.65,
            explosive_score=80.0,
            router_v6_conf=0.70,
            protection=SimpleNamespace(
                label="pull_up",
                confidence=0.86,
                reason="pull_up_bodyweight_pattern",
            ),
        )
    )

    assert decision.label is None
    assert decision.reason is None


def test_protected_evidence_keeps_front_squat_weak_router_recovery():
    decision = select_protected_evidence(
        _ctx(
            raw_label="squat",
            base_conf=0.98,
            bio_label="squat",
            bio_conf=0.98,
            squat_label="squat_back",
            squat_conf=0.60,
            wrist_overhead_ratio=0.40,
            bar_debug={
                "scores": {
                    "squat_front": 0.40,
                    "overhead_squat": 0.20,
                }
            },
        )
    )

    assert decision.label == "squat_front"
    assert decision.confidence == 0.80
    assert decision.reason == "front_squat_weak_router_recovery"


def test_protected_evidence_blocks_thruster_shape_short_squat_bench_rescue():
    decision = select_protected_evidence(
        _ctx(
            raw_label="squat",
            base_conf=0.935,
            bio_label="push_press",
            bio_conf=0.935,
            squat_label="squat_back",
            squat_conf=0.953,
            olympic_pred="split_jerk",
            olympic_conf=0.62,
            run_oly_router=True,
            wrist_overhead_ratio=0.969,
            explosive_score=105.13,
            looks_thruster=True,
            bodyweight_debug={
                "shoulder_y_range": 0.241,
                "hip_y_range": 0.222,
            },
            bar_debug={
                "squat_frames_used": 180,
            },
        )
    )

    assert decision.label is None
    assert decision.reason is None


def test_protected_evidence_rejects_long_barbell_pull_up_collision():
    decision = select_protected_evidence(
        _ctx(
            squat_label="overhead_squat",
            squat_conf=0.90,
            olympic_pred="clean_and_jerk",
            olympic_conf=0.75,
            protection=SimpleNamespace(
                label="pull_up",
                confidence=0.86,
                reason="pull_up_bodyweight_pattern",
            ),
            bodyweight_debug={
                "total_frames": 300,
                "wrist_y_range": 0.30,
                "hip_y_range": 0.20,
                "wrist_above_shoulder_ratio": 0.90,
            },
        )
    )

    assert decision.label is None
    assert decision.confidence == 0.0
    assert decision.reason is None


def test_early_final_decision_applies_protected_evidence_first():
    decision = select_early_final_decision(
        _early_ctx(
            protected_label="bench_press",
            protected_conf=0.95,
            protected_reason="bench_model_consensus",
        )
    )

    assert decision.label == "bench_press"
    assert decision.confidence == 0.95
    assert decision.mode == "biomechanics_override"
    assert decision.protected_reason == "bench_model_consensus"


def test_early_final_decision_respects_strong_olympic_lock():
    decision = select_early_final_decision(
        _early_ctx(
            protected_label="bench_press",
            protected_conf=0.95,
            strong_oly_lock=True,
        )
    )

    assert decision is None


def test_early_final_decision_accepts_high_conf_pull_up_router():
    decision = select_early_final_decision(
        _early_ctx(
            bodyweight_router_label="pull_up",
            bodyweight_router_conf=0.97,
            raw_label="push_press",
            bio_label="push_press",
            bodyweight_debug={
                "wrist_above_shoulder_ratio": 0.90,
            },
        )
    )

    assert decision.label == "pull_up"
    assert decision.mode == "bodyweight_router"
    assert decision.protected_reason == "bodyweight_router_pull_up_high_conf"


def test_early_final_decision_locks_clean_and_jerk():
    decision = select_early_final_decision(
        _early_ctx(
            raw_label="squat",
            olympic_pred="clean_and_jerk",
            olympic_conf=0.80,
            run_oly_router=True,
            explosive_score=30.0,
        )
    )

    assert decision.label == "clean_and_jerk"
    assert decision.mode == "olympic_locked"


def test_early_final_decision_preserves_push_press_over_split_shape():
    decision = select_early_final_decision(
        _early_ctx(
            raw_label="push_press",
            base_conf=0.90,
            bio_label="push_press",
            bio_conf=0.91,
            router_v6_label="push_press",
            router_v6_conf=0.90,
            looks_split=True,
            explosive_score=40.0,
            olympic_pred="clean_and_jerk",
            olympic_conf=0.60,
            bodyweight_debug={
                "wrist_y_range": 0.30,
                "shoulder_y_range": 0.10,
            },
        )
    )

    assert decision.label == "push_press"
    assert decision.mode == "biomechanics_override"


def test_early_final_decision_raw_front_squat_consensus():
    decision = select_early_final_decision(
        _early_ctx(
            raw_label="squat_front",
            base_conf=0.93,
            bio_label="squat",
            bio_conf=0.88,
            truly_explosive=False,
            strong_overhead=False,
            olympic_conf=0.50,
        )
    )

    assert decision.label == "squat_front"
    assert decision.mode == "squat_raw_consensus"


def test_fallback_final_decision_selects_clear_squat():
    decision = select_fallback_final_decision(
        _fallback_ctx(
            raw_label="squat",
            bio_label="squat",
            squat_label="squat_back",
            squat_conf=0.82,
            squat_confident=True,
            truly_explosive=False,
            strong_overhead=False,
        )
    )

    assert decision.label == "squat_back"
    assert decision.mode == "detailed_rep_analysis"


def test_fallback_final_decision_selects_olympic_router():
    decision = select_fallback_final_decision(
        _fallback_ctx(
            olympic_pred="snatch",
            olympic_conf=0.70,
            run_oly_router=True,
        )
    )

    assert decision.label == "snatch"
    assert decision.mode == "olympic_locked"


def test_fallback_final_decision_preserves_overhead_squat_guard():
    decision = select_fallback_final_decision(
        _fallback_ctx(
            raw_label="push_press",
            squat_label="overhead_squat",
            squat_conf=0.92,
            olympic_conf=0.40,
            bodyweight_debug={
                "wrist_above_shoulder_ratio": 0.90,
                "mean_wrist_minus_shoulder_y": -0.12,
                "elbow_range": 130.0,
                "min_elbow": 20.0,
                "avg_torso_angle": 5.0,
                "avg_wrist_forward": 0.01,
                "wrist_y_range": 0.10,
            },
        )
    )

    assert decision.label == "unknown"
    assert decision.mode == "insufficient_signal"


def test_fallback_final_decision_shape_clean():
    decision = select_fallback_final_decision(
        _fallback_ctx(
            looks_clean_only=True,
            truly_explosive=True,
            looks_cj=False,
            looks_split=False,
        )
    )

    assert decision.label == "clean"
    assert decision.confidence == 0.75
    assert decision.mode == "shape_override"


def test_fallback_final_decision_push_press_base_fallback():
    decision = select_fallback_final_decision(
        _fallback_ctx(
            raw_label="push_press",
            base_conf=0.70,
            looks_split=False,
        )
    )

    assert decision.label == "push_press"
    assert decision.mode == "base_model_locked"
