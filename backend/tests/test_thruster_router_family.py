from app.ml.central_router_shadow import arbitrate_shadow
from app.ml.family_router_shadow import classify_family_shadow
from app.ml.specialist_router_stack import classify_specialist_routers


def test_family_shadow_routes_thruster_to_press_not_olympic():
    result = classify_family_shadow(
        candidates={
            "base": {"label": "thruster", "confidence": 0.78},
            "biomechanics": {"label": "thruster", "confidence": 0.76},
            "olympic_router": {"label": "clean", "confidence": 0.62},
            "protected_evidence": {
                "label": "thruster",
                "confidence": 0.86,
                "reason": "thruster_pattern_detected",
            },
        },
        truly_explosive=False,
        explosive_score=22.0,
        looks_clean_only=False,
        looks_cj=False,
        looks_split=False,
        looks_thruster=True,
        strong_overhead=True,
    )

    assert result["family"] == "press"
    assert result["scores"]["press"] > result["scores"].get("olympic", 0.0)


def test_specialist_stack_keeps_thruster_in_press_lane():
    result = classify_specialist_routers(
        candidates={
            "base": {"label": "thruster", "confidence": 0.78},
            "biomechanics": {"label": "thruster", "confidence": 0.76},
            "olympic_router": {"label": "clean_and_jerk", "confidence": 0.64},
        },
        press_variant={
            "eligible": True,
            "label": "thruster",
            "score": 1.55,
            "features": {
                "deep_squat": True,
                "large_vertical_travel": True,
            },
        },
        family_shadow={
            "signals": {
                "false_pull_up_during_press": False,
            },
        },
    )

    assert result["winner"]["router"] == "press"
    assert result["winner"]["label"] == "thruster"
    assert result["routers"]["olympic"]["label"] == "clean_and_jerk"


def test_central_shadow_has_press_gate_for_thruster_shape():
    result = arbitrate_shadow(
        candidates={
            "base": {"label": "thruster", "confidence": 0.74},
            "biomechanics": {"label": "thruster", "confidence": 0.72},
            "olympic_router": {"label": "clean", "confidence": 0.58},
            "router_v5": {
                "label": "clean",
                "confidence": 0.58,
                "decision": "low_confidence",
            },
        },
        truly_explosive=False,
        explosive_score=30.0,
        looks_clean_only=False,
        looks_cj=False,
        looks_split=False,
        looks_thruster=True,
        strong_overhead=True,
        wrist_overhead_ratio=0.55,
    )

    assert result["label"] == "thruster"
    assert result["eligibility"]["thruster"] is True
    assert result["eligibility"]["olympic"] is False
