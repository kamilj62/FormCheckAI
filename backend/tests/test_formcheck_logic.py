import pytest

from app.logic import classify_with_biomechanics, build_set_summary
from app.ml.squat_variant_recovery import (
    should_recover_front_squat_from_back_router,
)

def test_trusts_raw_bench_press_prediction():
    summary = {
        "min_knee_angle": 90,
        "max_knee_angle": 178,
        "min_hip_angle": 7,
        "max_hip_angle": 178,
        "min_torso_angle": 0,
        "max_torso_angle": 179,
        "min_elbow_angle": 40,
        "max_elbow_angle": 175,
        "wrist_above_shoulder_ratio": 0.67,
    }

    label, confidence, override_used, reason = classify_with_biomechanics(
        raw_label="bench_press",
        confidence=0.415,
        summary=summary,
        pose_frames=118,
    )

    assert label == "bench_press"
    assert confidence >= 0.80
    assert reason == "trusted_model_bench_press"


def test_push_press_detected_from_overhead_and_knee_drive():
    summary = {
        "min_knee_angle": 120,
        "max_knee_angle": 160,
        "min_hip_angle": 130,
        "max_hip_angle": 150,
        "min_torso_angle": 80,
        "max_torso_angle": 90,
        "min_elbow_angle": 70,
        "max_elbow_angle": 175,
        "wrist_above_shoulder_ratio": 0.80,
    }

    label, confidence, override_used, reason = classify_with_biomechanics(
        raw_label="squat",
        confidence=0.40,
        summary=summary,
        pose_frames=50,
    )

    assert label == "push_press"
    assert override_used is True


def test_low_pose_data_does_not_override():
    summary = {}

    label, confidence, override_used, reason = classify_with_biomechanics(
        raw_label="bench_press",
        confidence=0.30,
        summary=summary,
        pose_frames=5,
    )

    assert label == "bench_press"
    assert override_used is False
    assert reason == "low_pose_data"


def test_build_set_summary_empty_reps():
    summary = build_set_summary([])

    assert summary["detected_reps"] == 0
    assert summary["avg_rep_score"] == 0
    assert summary["trend"] == "No clear reps detected."


def test_recovers_front_squat_from_back_router_rack_confusion():
    assert should_recover_front_squat_from_back_router(
        forced_exercise_label=None,
        final_label="squat_back",
        raw_label="squat",
        bio_label="push_press",
        squat_label="squat_back",
        squat_conf=0.95,
        olympic_pred="clean_and_jerk",
        olympic_conf=0.72,
        truly_explosive=False,
        looks_clean_only=False,
        looks_cj=False,
        looks_split=False,
    )


def test_recovers_explosive_front_squat_from_back_router_rack_confusion():
    assert should_recover_front_squat_from_back_router(
        forced_exercise_label=None,
        final_label="squat_back",
        raw_label="squat",
        bio_label="push_press",
        squat_label="squat_back",
        squat_conf=0.95,
        olympic_pred="split_jerk",
        olympic_conf=0.62,
        truly_explosive=True,
        looks_clean_only=False,
        looks_cj=False,
        looks_split=False,
        looks_thruster=True,
        bar_debug={
            "front_rack_elbow_p25": 3.9,
            "avg_elbow_angle_sq": 37.9,
            "squat_frames_used": 180,
            "scores": {
                "squat_front": 0.60,
                "squat_back": 0.15,
                "overhead_squat": 0.80,
            },
        },
    )


def test_front_squat_recovery_does_not_override_forced_or_olympic_sequence():
    common = {
        "final_label": "squat_back",
        "raw_label": "squat",
        "bio_label": "push_press",
        "squat_label": "squat_back",
        "squat_conf": 0.95,
        "olympic_pred": "clean_and_jerk",
        "olympic_conf": 0.72,
        "looks_clean_only": False,
        "looks_cj": False,
        "looks_split": False,
    }

    assert not should_recover_front_squat_from_back_router(
        forced_exercise_label="squat_back",
        truly_explosive=False,
        **common,
    )
    assert not should_recover_front_squat_from_back_router(
        forced_exercise_label=None,
        truly_explosive=True,
        **common,
    )
    assert not should_recover_front_squat_from_back_router(
        forced_exercise_label=None,
        truly_explosive=False,
        looks_cj=True,
        **{k: v for k, v in common.items() if k != "looks_cj"},
    )
