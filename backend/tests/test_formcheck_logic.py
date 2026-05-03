import pytest

from app.logic import classify_with_biomechanics, build_set_summary

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