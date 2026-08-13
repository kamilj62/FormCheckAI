from __future__ import annotations


def should_recover_front_squat_from_back_router(
    *,
    forced_exercise_label,
    final_label,
    raw_label,
    bio_label,
    squat_label,
    squat_conf,
    olympic_pred,
    olympic_conf,
    truly_explosive,
    looks_clean_only,
    looks_cj,
    looks_split,
    looks_thruster=False,
    bar_debug=None,
) -> bool:
    """
    Recover front squats that the squat subtype router calls back squat.

    Front-rack posture can look like a press to the broad biomechanics model
    and like a weak C&J to the Olympic router. Keep this narrow so explosive
    Olympic sequences and already-forced analyses are left alone.
    """
    bar_debug = bar_debug or {}
    front_rack_bar_evidence = (
        float((bar_debug.get("scores") or {}).get("squat_front") or 0.0)
        >= 0.55
        and float(bar_debug.get("front_rack_elbow_p25", 180.0)) <= 45.0
        and float(bar_debug.get("avg_elbow_angle_sq", 180.0)) <= 70.0
        and float(bar_debug.get("squat_frames_used", 0.0)) >= 80.0
    )
    classic_clean_confusion = (
        olympic_pred == "clean_and_jerk"
        and float(olympic_conf or 0.0) < 0.90
        and not bool(truly_explosive)
    )
    explosive_front_rack_confusion = (
        olympic_pred in {"split_jerk", "clean_and_jerk"}
        and float(olympic_conf or 0.0) < 0.75
        and bool(looks_thruster)
        and front_rack_bar_evidence
    )

    return (
        not forced_exercise_label
        and final_label == "squat_back"
        and raw_label == "squat"
        and bio_label == "push_press"
        and squat_label == "squat_back"
        and float(squat_conf or 0.0) >= 0.85
        and (classic_clean_confusion or explosive_front_rack_confusion)
        and not bool(looks_clean_only)
        and not bool(looks_cj)
        and not bool(looks_split)
    )
