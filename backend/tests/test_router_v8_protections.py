from app.ml.router_v8.protections import (
    bodyweight_protections,
    strength_protections,
)


def _pull_up_inputs(**overrides):
    data = {
        "raw_label": "push_press",
        "squat_label": "overhead_squat",
        "bodyweight_debug": {
            "total_frames": 176,
            "wrist_above_shoulder_ratio": 0.972,
            "mean_wrist_minus_shoulder_y": -0.178,
            "wrist_y_range": 0.058,
            "shoulder_y_range": 0.242,
            "elbow_range": 176.0,
            "min_elbow": 1.8,
            "avg_torso_angle": 1.4,
            "avg_wrist_forward": 0.004,
            "hip_y_range": 0.309,
        },
        "bodyweight_router_label": "pull_up",
        "bodyweight_router_conf": 0.99,
        "strong_bench_evidence": False,
        "looks_push_up": False,
        "looks_pull_up": True,
        "looks_handstand_push_up": False,
        "looks_muscle_up": False,
        "looks_burpee": False,
        "credible_split_jerk": False,
    }
    data.update(overrides)
    return data


def test_bodyweight_protection_keeps_true_pull_up_shape():
    result = bodyweight_protections(**_pull_up_inputs())

    assert result.label == "pull_up"
    assert result.reason == "pull_up_bodyweight_pattern"


def test_bodyweight_protection_rejects_barbell_path_false_pull_up():
    result = bodyweight_protections(
        **_pull_up_inputs(
            raw_label="squat",
            squat_label="squat_back",
            bodyweight_debug={
                "total_frames": 450,
                "wrist_above_shoulder_ratio": 0.969,
                "mean_wrist_minus_shoulder_y": -0.033,
                "wrist_y_range": 0.274,
                "shoulder_y_range": 0.241,
                "elbow_range": 172.0,
                "min_elbow": 1.0,
                "avg_torso_angle": 1.6,
                "avg_wrist_forward": 0.016,
                "hip_y_range": 0.222,
            },
        )
    )

    assert result.label is None
    assert result.reason is None


def test_bodyweight_protection_rejects_overhead_squat_false_pull_up():
    result = bodyweight_protections(
        **_pull_up_inputs(
            raw_label="push_press",
            squat_label="overhead_squat",
            bodyweight_debug={
                "total_frames": 255,
                "wrist_above_shoulder_ratio": 0.886,
                "mean_wrist_minus_shoulder_y": -0.099,
                "wrist_y_range": 0.357,
                "shoulder_y_range": 0.218,
                "elbow_range": 137.0,
                "min_elbow": 41.6,
                "avg_torso_angle": 2.0,
                "avg_wrist_forward": 0.010,
                "hip_y_range": 0.185,
            },
        )
    )

    assert result.label is None
    assert result.reason is None


def test_strength_protection_rejects_low_explosive_overhead_squat_push_press_hold():
    result = strength_protections(
        raw_label="push_press",
        base_conf=0.999,
        bio_label="push_press",
        bio_conf=0.999,
        squat_label="overhead_squat",
        squat_conf=0.811,
        explosive_score=21.45,
        bodyweight_debug={
            "wrist_y_range": 0.357,
            "shoulder_y_range": 0.218,
            "hip_y_range": 0.185,
        },
        looks_strict=False,
        looks_thruster=False,
        looks_clean_only=False,
        looks_cj=False,
        looks_split=False,
    )

    assert result.label is None
    assert result.reason is None


def test_strength_protection_recovers_controlled_strict_press_shape():
    result = strength_protections(
        raw_label="squat_front",
        base_conf=0.995,
        bio_label="push_press",
        bio_conf=0.995,
        squat_label="overhead_squat",
        squat_conf=0.811,
        explosive_score=7.13,
        bodyweight_debug={
            "wrist_y_range": 0.384,
            "shoulder_y_range": 0.171,
            "hip_y_range": 0.024,
        },
        looks_strict=False,
        looks_thruster=False,
        looks_clean_only=False,
        looks_cj=False,
        looks_split=False,
    )

    assert result.label == "strict_press"
    assert result.reason == "controlled_strict_press_pattern"
