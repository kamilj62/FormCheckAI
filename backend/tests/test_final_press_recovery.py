from app.ml.final_press_recovery import (
    should_recover_controlled_push_press,
    should_recover_explosive_push_press_authority,
    should_recover_push_press_over_back_squat,
    should_recover_push_press_over_weak_cj_split,
    should_recover_strict_press,
)


def test_controlled_push_press_recovery_requires_no_cj_shape():
    bodyweight_debug = {
        "shoulder_y_range": 0.08,
        "hip_y_range": 0.06,
        "wrist_above_shoulder_ratio": 0.90,
        "total_frames": 260,
    }

    assert should_recover_controlled_push_press(
        forced_exercise_label=None,
        final_label="clean_and_jerk",
        bio_label="push_press",
        bio_conf=0.80,
        squat_label="squat_back",
        squat_conf=0.94,
        olympic_pred="clean_and_jerk",
        olympic_conf=0.88,
        looks_cj=False,
        looks_split=False,
        explosive_score=10.0,
        bodyweight_debug=bodyweight_debug,
    )

    assert not should_recover_controlled_push_press(
        forced_exercise_label=None,
        final_label="clean_and_jerk",
        bio_label="push_press",
        bio_conf=0.80,
        squat_label="squat_back",
        squat_conf=0.94,
        olympic_pred="clean_and_jerk",
        olympic_conf=0.88,
        looks_cj=True,
        looks_split=False,
        explosive_score=10.0,
        bodyweight_debug=bodyweight_debug,
    )


def test_strict_press_recovery_requires_low_leg_drive():
    assert should_recover_strict_press(
        forced_exercise_label=None,
        final_label="push_press",
        raw_label="push_press",
        bio_label="push_press",
        looks_strict=True,
        looks_split=False,
        looks_thruster=False,
        explosive_score=8.0,
        squat_knee_range=10.0,
        squat_hip_range=8.0,
    )

    assert not should_recover_strict_press(
        forced_exercise_label=None,
        final_label="push_press",
        raw_label="push_press",
        bio_label="push_press",
        looks_strict=True,
        looks_split=False,
        looks_thruster=False,
        explosive_score=8.0,
        squat_knee_range=35.0,
        squat_hip_range=8.0,
    )


def test_push_press_recoveries_preserve_existing_reasons():
    bodyweight_debug = {
        "wrist_y_range": 0.45,
        "shoulder_y_range": 0.10,
        "hip_y_range": 0.08,
    }

    assert should_recover_push_press_over_weak_cj_split(
        forced_exercise_label=None,
        final_label="split_jerk",
        raw_label="push_press",
        base_conf=0.80,
        bio_label="push_press",
        bio_conf=0.80,
        router_v6_label="push_press",
        router_v6_conf=0.72,
        olympic_pred="clean_and_jerk",
        olympic_conf=0.60,
        looks_cj=False,
        explosive_score=85.0,
        bodyweight_debug=bodyweight_debug,
    )

    assert should_recover_push_press_over_back_squat(
        forced_exercise_label=None,
        final_label="squat_back",
        raw_label="squat",
        base_conf=0.70,
        bio_label="squat",
        bio_conf=0.70,
        squat_label="squat_back",
        squat_conf=0.92,
        olympic_pred="split_jerk",
        olympic_conf=0.82,
        looks_cj=False,
        explosive_score=10.0,
        bodyweight_debug=bodyweight_debug,
    ) == "low_motion_push_press_over_back_squat"

    assert should_recover_push_press_over_back_squat(
        forced_exercise_label=None,
        final_label="squat_back",
        raw_label="squat",
        base_conf=0.65,
        bio_label="push_press",
        bio_conf=0.80,
        squat_label="squat_back",
        squat_conf=0.96,
        olympic_pred="snatch",
        olympic_conf=0.40,
        looks_cj=False,
        explosive_score=120.0,
        bodyweight_debug=bodyweight_debug,
    ) == "explosive_push_press_over_back_squat"


def test_explosive_push_press_authority_requires_overhead_bar_evidence():
    assert should_recover_explosive_push_press_authority(
        forced_exercise_label=None,
        final_label="overhead_squat",
        raw_label="push_press",
        base_conf=0.995,
        bio_label="push_press",
        bio_conf=0.995,
        squat_label="overhead_squat",
        squat_conf=0.80,
        olympic_conf=0.60,
        explosive_score=120.0,
        bar_debug={"overhead_ratio": 0.96},
    )

    assert not should_recover_explosive_push_press_authority(
        forced_exercise_label=None,
        final_label="overhead_squat",
        raw_label="push_press",
        base_conf=0.995,
        bio_label="push_press",
        bio_conf=0.995,
        squat_label="overhead_squat",
        squat_conf=0.80,
        olympic_conf=0.60,
        explosive_score=120.0,
        bar_debug={"overhead_ratio": 0.50},
    )
