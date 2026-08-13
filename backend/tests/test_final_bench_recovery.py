from app.ml.final_bench_recovery import (
    should_recover_short_bench_over_pushup,
)


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


def test_recovers_short_bench_clip_from_push_up_collision():
    assert should_recover_short_bench_over_pushup(
        forced_exercise_label=None,
        final_label="push_up",
        raw_label="squat_front",
        base_conf=0.993,
        bio_label="deadlift",
        bio_conf=0.993,
        squat_label="squat_front",
        squat_conf=0.97,
        router_v6_label="squat_front",
        router_v6_conf=0.99,
        olympic_pred="clean_and_jerk",
        olympic_conf=0.688,
        bodyweight_debug=_bench05_bodyweight_debug(),
    )


def test_does_not_recover_when_floor_push_up_geometry_is_stronger():
    bodyweight_debug = _bench05_bodyweight_debug()
    bodyweight_debug["hip_y_range"] = 0.20

    assert not should_recover_short_bench_over_pushup(
        forced_exercise_label=None,
        final_label="push_up",
        raw_label="squat_front",
        base_conf=0.993,
        bio_label="deadlift",
        bio_conf=0.993,
        squat_label="squat_front",
        squat_conf=0.97,
        router_v6_label="squat_front",
        router_v6_conf=0.99,
        olympic_pred="clean_and_jerk",
        olympic_conf=0.688,
        bodyweight_debug=bodyweight_debug,
    )
