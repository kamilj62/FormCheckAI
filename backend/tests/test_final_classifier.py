from app.ml.final_classifier import simplify_final_classification


def test_simplified_classifier_promotes_protected_push_up():
    decision = simplify_final_classification(
        current_label="squat_front",
        current_confidence=0.97,
        current_mode="detailed_rep_analysis",
        forced_label=None,
        family_shadow={
            "family": "bodyweight",
            "score": 1.8,
            "margin": 0.8,
            "signals": {
                "protected_bodyweight_supported": True,
            },
        },
        press_variant_shadow={},
        hierarchical_shadow={
            "eligible": True,
            "family": "bodyweight",
            "label": "push_up",
            "confidence": 0.86,
            "source": "protected_evidence",
        },
    )

    assert decision.changed is True
    assert decision.label == "push_up"
    assert decision.mode == "simplified_classifier"
    assert decision.reason == "protected_bodyweight_hierarchy"


def test_simplified_classifier_promotes_handstand_push_up_as_bodyweight():
    decision = simplify_final_classification(
        current_label="push_press",
        current_confidence=0.82,
        current_mode="biomechanics_override",
        forced_label=None,
        family_shadow={
            "family": "bodyweight",
            "score": 1.9,
            "margin": 0.6,
            "signals": {
                "protected_bodyweight_supported": True,
            },
        },
        press_variant_shadow={
            "label": "push_press",
            "score": 0.7,
            "margin": 0.1,
            "features": {},
        },
        hierarchical_shadow={
            "eligible": True,
            "family": "bodyweight",
            "label": "handstand_push_up",
            "confidence": 0.88,
            "source": "protected_evidence",
        },
    )

    assert decision.changed is True
    assert decision.label == "handstand_push_up"
    assert decision.reason == "protected_bodyweight_hierarchy"


def test_simplified_classifier_keeps_false_pull_up_over_press_blocked():
    decision = simplify_final_classification(
        current_label="push_press",
        current_confidence=0.84,
        current_mode="biomechanics_override",
        forced_label=None,
        family_shadow={
            "family": "press",
            "score": 1.6,
            "margin": 0.5,
            "signals": {
                "false_pull_up_during_press": True,
                "protected_bodyweight_supported": False,
            },
        },
        press_variant_shadow={
            "label": "push_press",
            "score": 1.8,
            "margin": 0.4,
            "features": {
                "push_press_consensus": True,
            },
        },
        hierarchical_shadow={
            "eligible": True,
            "family": "press",
            "label": "push_press",
            "confidence": 1.8,
            "source": "press_variant_shadow",
        },
    )

    assert decision.changed is False
    assert decision.label == "push_press"


def test_simplified_classifier_promotes_clear_press_variant():
    decision = simplify_final_classification(
        current_label="thruster",
        current_confidence=0.80,
        current_mode="biomechanics_override",
        forced_label=None,
        family_shadow={
            "family": "press",
            "score": 2.1,
            "margin": 0.7,
            "signals": {},
        },
        press_variant_shadow={
            "label": "push_press",
            "score": 2.0,
            "margin": 0.3,
            "features": {
                "push_press_consensus": True,
            },
        },
        hierarchical_shadow={
            "eligible": True,
            "family": "press",
            "label": "push_press",
            "confidence": 2.0,
            "source": "press_variant_shadow",
        },
    )

    assert decision.changed is True
    assert decision.label == "push_press"
    assert decision.reason == "press_hierarchy_variant"


def test_simplified_classifier_promotes_clear_strict_press():
    decision = simplify_final_classification(
        current_label="push_press",
        current_confidence=0.78,
        current_mode="biomechanics_override",
        forced_label=None,
        family_shadow={
            "family": "press",
            "score": 2.0,
            "margin": 0.6,
            "signals": {},
        },
        press_variant_shadow={
            "label": "strict_press",
            "score": 2.2,
            "margin": 0.5,
            "features": {
                "strict_press_geometry": True,
                "strict_press_consensus": True,
            },
        },
        hierarchical_shadow={
            "eligible": True,
            "family": "press",
            "label": "strict_press",
            "confidence": 2.2,
            "source": "press_variant_shadow",
        },
    )

    assert decision.changed is True
    assert decision.label == "strict_press"
    assert decision.reason == "press_hierarchy_variant"


def test_simplified_classifier_does_not_override_forced_label():
    decision = simplify_final_classification(
        current_label="deadlift",
        current_confidence=0.90,
        current_mode="forced",
        forced_label="deadlift",
        family_shadow={
            "family": "bodyweight",
            "score": 2.0,
            "margin": 1.0,
            "signals": {
                "protected_bodyweight_supported": True,
            },
        },
        press_variant_shadow={},
        hierarchical_shadow={
            "eligible": True,
            "family": "bodyweight",
            "label": "push_up",
            "confidence": 0.86,
            "source": "protected_evidence",
        },
    )

    assert decision.changed is False
    assert decision.label == "deadlift"
