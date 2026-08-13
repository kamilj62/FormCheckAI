from app.ml.final_decision_router import (
    RouterV5AdjustmentContext,
    adjust_router_v5_prediction,
)


def _ctx(**overrides):
    data = {
        "router_v5_label": "clean",
        "router_v5_conf": 0.60,
        "router_v5_debug": {},
        "raw_label": "squat",
        "base_conf": 0.50,
        "bio_label": "squat",
        "bio_conf": 0.50,
        "squat_label": "squat_back",
        "squat_conf": 0.50,
        "olympic_pred": None,
        "olympic_conf": 0.0,
        "explosive_score": 0.0,
        "wrist_overhead_ratio": 0.0,
        "looks_clean_only": False,
        "looks_cj": False,
        "looks_split": False,
        "truly_explosive": False,
        "bodyweight_debug": {},
    }
    data.update(overrides)
    return RouterV5AdjustmentContext(**data)


def test_router_v5_adjustment_recovers_standalone_split_from_cj():
    result = adjust_router_v5_prediction(
        _ctx(
            router_v5_label="clean_and_jerk",
            router_v5_conf=0.70,
            router_v5_debug={
                "split_features": {
                    "lockout_duration": 220.0,
                    "catch_to_finish": 320.0,
                }
            },
            raw_label="push_press",
            bio_label="push_press",
            olympic_pred="clean_and_jerk",
            olympic_conf=0.85,
            looks_split=True,
        )
    )

    assert result.label == "split_jerk"
    assert result.confidence == 0.80
    assert result.decision == "standalone_split_from_cj"


def test_router_v5_adjustment_restores_high_conf_clean_and_jerk():
    result = adjust_router_v5_prediction(
        _ctx(
            router_v5_label="split_jerk",
            router_v5_conf=0.70,
            router_v5_debug={"split_features": {}},
            olympic_pred="clean_and_jerk",
            olympic_conf=0.92,
        )
    )

    assert result.label == "clean_and_jerk"
    assert result.confidence == 0.92
    assert result.decision == "clean_and_jerk_high_conf_rescue"


def test_router_v5_adjustment_recovers_full_cj_sequence_from_clean():
    result = adjust_router_v5_prediction(
        _ctx(
            router_v5_label="clean",
            router_v5_conf=0.65,
            router_v5_debug={
                "events": {
                    "clean_extension": 1,
                    "clean_catch": 2,
                    "clean_recovery": 3,
                    "jerk_dip": 4,
                    "jerk_drive": 5,
                    "jerk_catch": 6,
                    "lockout": 7,
                },
                "features": {
                    "has_overhead": 1.0,
                    "catch_overhead": 1.0,
                    "lockout_duration": 25.0,
                    "catch_to_finish": 100.0,
                },
            },
            olympic_pred="clean_and_jerk",
            olympic_conf=0.55,
            explosive_score=30.0,
        )
    )

    assert result.label == "clean_and_jerk"
    assert result.confidence == 0.80
    assert result.decision == "clean_and_jerk_full_sequence_rescue"


def test_router_v5_adjustment_recovers_bench_from_short_split_press():
    result = adjust_router_v5_prediction(
        _ctx(
            router_v5_label="split_jerk",
            router_v5_conf=0.60,
            router_v5_debug={
                "split_features": {
                    "lockout_duration": 40.0,
                    "catch_to_finish": 50.0,
                }
            },
            raw_label="deadlift",
            base_conf=0.90,
            bio_label="deadlift",
            bio_conf=0.88,
            olympic_conf=0.60,
            bodyweight_debug={
                "avg_torso_angle": 70.0,
                "elbow_range": 120.0,
                "avg_wrist_forward": 0.02,
                "mean_hip_minus_shoulder_y": 0.05,
                "wrist_above_shoulder_ratio": 0.20,
            },
        )
    )

    assert result.label == "bench_press"
    assert result.confidence == 0.90
    assert result.decision == "bench_press_short_split_rescue"


def test_router_v5_adjustment_preserves_clean_rescue_active_flag():
    result = adjust_router_v5_prediction(
        _ctx(
            router_v5_label="clean",
            router_v5_conf=0.72,
            router_v5_debug={
                "decision": "clean_rescue_from_weak_snatch",
            },
            truly_explosive=True,
        )
    )

    assert result.label == "clean"
    assert result.clean_rescue_active is True
