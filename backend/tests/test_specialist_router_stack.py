from app.ml.specialist_router_stack import classify_specialist_routers


def test_specialist_stack_has_four_router_lanes():
    result = classify_specialist_routers(
        candidates={
            "base": {"label": "push_press", "confidence": 0.80},
            "biomechanics": {"label": "push_press", "confidence": 0.82},
            "bodyweight_router": {"label": "pull_up", "confidence": 0.99},
        },
        press_variant={
            "eligible": True,
            "label": "push_press",
            "score": 1.7,
            "features": {
                "push_press_consensus": True,
            },
        },
        family_shadow={
            "signals": {
                "false_pull_up_during_press": True,
            },
        },
    )

    assert set(result["routers"]) == {
        "press",
        "bodyweight",
        "olympic",
        "squat",
    }
    assert result["winner"]["router"] == "press"
    assert result["winner"]["label"] == "push_press"
    assert result["routers"]["bodyweight"]["eligible"] is False


def test_specialist_stack_accepts_contextual_bodyweight_router():
    result = classify_specialist_routers(
        candidates={
            "bodyweight_router": {
                "label": "push_up",
                "confidence": 0.72,
            },
            "protected_evidence": {
                "label": "push_up",
                "confidence": 0.86,
                "reason": "push_up_bodyweight_pattern",
            },
            "squat_router": {
                "label": "squat_front",
                "confidence": 0.84,
            },
        },
        press_variant={
            "eligible": False,
            "label": None,
        },
        family_shadow={
            "signals": {
                "false_pull_up_during_press": False,
            },
        },
    )

    assert result["winner"]["router"] == "bodyweight"
    assert result["winner"]["label"] == "push_up"


def test_specialist_stack_routes_handstand_push_up_as_bodyweight():
    result = classify_specialist_routers(
        candidates={
            "bodyweight_router": {
                "label": "handstand_push_up",
                "confidence": 0.78,
            },
            "protected_evidence": {
                "label": "handstand_push_up",
                "confidence": 0.88,
                "reason": "handstand_push_up_bodyweight_pattern",
            },
            "base": {
                "label": "push_press",
                "confidence": 0.72,
            },
        },
        press_variant={
            "eligible": False,
            "label": None,
        },
        family_shadow={
            "signals": {
                "false_pull_up_during_press": False,
            },
        },
    )

    assert result["winner"]["router"] == "bodyweight"
    assert result["winner"]["label"] == "handstand_push_up"
    assert result["routers"]["press"]["label"] == "push_press"
    assert result["routers"]["bodyweight"]["source"] == "protected_evidence"


def test_specialist_stack_prefers_verified_olympic_router_v5():
    result = classify_specialist_routers(
        candidates={
            "router_v5": {
                "label": "split_jerk",
                "confidence": 0.80,
                "decision": "standalone_split_from_cj",
            },
            "olympic_router": {
                "label": "clean_and_jerk",
                "confidence": 0.82,
            },
            "squat_router": {
                "label": "overhead_squat",
                "confidence": 0.78,
            },
        },
        press_variant={
            "eligible": False,
            "label": None,
        },
    )

    assert result["winner"]["router"] == "olympic"
    assert result["winner"]["label"] == "split_jerk"
    assert result["routers"]["olympic"]["source"] == "router_v5"


def test_specialist_stack_keeps_squat_variant_inside_squat_router():
    result = classify_specialist_routers(
        candidates={
            "base": {"label": "squat", "confidence": 0.90},
            "biomechanics": {"label": "squat", "confidence": 0.88},
            "squat_router": {
                "label": "squat_back",
                "confidence": 0.92,
            },
        },
        press_variant={
            "eligible": False,
            "label": None,
        },
    )

    assert result["winner"]["router"] == "squat"
    assert result["winner"]["label"] == "squat_back"
