from app.ml.final_arbitration_adapters import FinalArbitrationProbeAdapters
from app.ml.final_decision_router import FinalDecisionState


def _adapter(**overrides):
    data = {
        "biomechanics": [],
        "forced_exercise_label": None,
        "raw_label": "push_press",
        "base_confidence": 0.99,
        "bio_label": "push_press",
        "bio_confidence": 0.99,
        "squat_label": "overhead_squat",
        "squat_confidence": 0.82,
        "router_v6_label": "push_press",
        "router_v6_confidence": 0.90,
        "bodyweight_router_label": None,
        "bodyweight_router_confidence": 0.0,
        "olympic_pred": "clean_and_jerk",
        "olympic_confidence": 0.82,
        "wrist_overhead_ratio": 0.0,
        "explosive_score": 15.0,
        "bodyweight_debug": {},
        "bar_debug": {
            "overhead_ratio": 0.85,
            "total_frames": 300,
        },
        "use_yolo_tracking": False,
        "summarize_biomechanics": lambda biomechanics: {},
        "analyze_push_press_reps": (
            lambda biomechanics, label: ([{"rep": 1}, {"rep": 2}, {"rep": 3}], {})
        ),
        "analyze_deadlift_reps": lambda biomechanics: [],
        "analyze_yolo_deadlift_reps": lambda biomechanics: [],
        "analyze_squat_reps": lambda biomechanics, label: ([], {}),
    }
    data.update(overrides)
    return FinalArbitrationProbeAdapters(**data)


def test_push_press_probe_recovers_confirmed_split_confusion():
    state = _adapter().push_press_probe(
        FinalDecisionState(
            final_label="split_jerk",
            final_confidence=0.85,
            analysis_mode="router_v5",
            protected_label="split_jerk",
            protected_confidence=0.85,
            protected_reason="standalone_split_from_cj",
        )
    )

    assert state.final_label == "push_press"
    assert state.final_confidence == 0.99
    assert state.protected_reason == "push_press_analyzer_over_split_authority"
