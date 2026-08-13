from app.ml.family_router_shadow import classify_family_shadow
from app.ml.press_variant_shadow import classify_press_variant_shadow


def test_protected_push_up_evidence_wins_bodyweight_family():
    result = classify_family_shadow(
        candidates={
            "base": {"label": "squat", "confidence": 0.86},
            "biomechanics": {"label": "squat", "confidence": 0.86},
            "squat_router": {"label": "squat_front", "confidence": 0.90},
            "bodyweight_router": {"label": "push_up", "confidence": 0.55},
            "protected_evidence": {
                "label": "push_up",
                "confidence": 0.86,
                "reason": "push_up_bodyweight_pattern",
            },
        },
        truly_explosive=False,
        explosive_score=0.0,
        looks_clean_only=False,
        looks_cj=False,
        looks_split=False,
        looks_thruster=False,
        strong_overhead=False,
    )

    assert result["family"] == "bodyweight"
    assert result["signals"]["protected_bodyweight_supported"] is True


def test_false_pull_up_protection_does_not_steal_press_family():
    result = classify_family_shadow(
        candidates={
            "base": {"label": "push_press", "confidence": 0.84},
            "biomechanics": {"label": "push_press", "confidence": 0.82},
            "bodyweight_router": {"label": "pull_up", "confidence": 0.99},
            "protected_evidence": {
                "label": "pull_up",
                "confidence": 0.99,
                "reason": "router_v6_bodyweight_winner",
            },
        },
        truly_explosive=False,
        explosive_score=10.0,
        looks_clean_only=False,
        looks_cj=False,
        looks_split=False,
        looks_thruster=False,
        strong_overhead=True,
    )

    assert result["family"] == "press"
    assert result["signals"]["false_pull_up_during_press"] is True
    assert result["signals"]["protected_bodyweight_supported"] is False


def test_push_press_consensus_beats_ambiguous_thruster_shape():
    result = classify_press_variant_shadow(
        family="press",
        biomechanics_summary={
            "avg_torso_angle": 12.0,
            "min_knee_angle": 95.0,
            "max_knee_angle": 160.0,
            "min_hip_angle": 100.0,
            "max_hip_angle": 150.0,
        },
        bodyweight_summary={
            "shoulder_y_range": 0.15,
            "hip_y_range": 0.15,
            "wrist_above_shoulder_ratio": 0.75,
        },
        routing_candidates={
            "base": {"label": "push_press", "confidence": 0.82},
            "biomechanics": {"label": "push_press", "confidence": 0.80},
            "squat_router": {"label": "squat_front", "confidence": 0.84},
            "protected_evidence": {
                "label": "push_press",
                "confidence": 0.82,
                "reason": "learned_press_hierarchy_authority",
            },
        },
        explosive_score=22.0,
        looks_strict=False,
        looks_thruster=True,
        strong_overhead=True,
    )

    assert result["label"] == "push_press"
    assert result["features"]["push_press_consensus"] is True


def test_strict_press_geometry_beats_push_press_bias():
    result = classify_press_variant_shadow(
        family="press",
        biomechanics_summary={
            "avg_torso_angle": 10.0,
            "min_knee_angle": 171.0,
            "max_knee_angle": 180.0,
            "min_hip_angle": 172.0,
            "max_hip_angle": 180.0,
        },
        bodyweight_summary={
            "shoulder_y_range": 0.08,
            "hip_y_range": 0.03,
            "wrist_above_shoulder_ratio": 0.82,
        },
        routing_candidates={
            "base": {"label": "push_press", "confidence": 0.76},
            "biomechanics": {"label": "strict_press", "confidence": 0.78},
            "protected_evidence": {
                "label": "strict_press",
                "confidence": 0.86,
                "reason": "strict_press_pattern_detected",
            },
        },
        explosive_score=4.0,
        looks_strict=True,
        looks_thruster=False,
        strong_overhead=True,
    )

    assert result["label"] == "strict_press"
    assert result["features"]["strict_press_geometry"] is True
    assert result["features"]["strict_press_consensus"] is True
