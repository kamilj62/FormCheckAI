from app.ml.router_audit import (
    build_router_score_flags,
    finalize_router_scores,
    initialize_router_audit,
    populate_router_scores,
)


def test_router_audit_downweights_generic_models_for_high_conf_bodyweight():
    strong_bench, bodyweight_high_conf = build_router_score_flags(
        raw_label="push_press",
        base_conf=0.99,
        bio_label="push_press",
        bio_conf=0.99,
        looks_thruster=False,
        bodyweight_router_label="handstand_push_up",
        bodyweight_router_conf=0.96,
    )

    assert strong_bench is False
    assert bodyweight_high_conf is True

    _, router_scores, _, add_router_score = initialize_router_audit(
        raw_label="push_press",
        base_conf=0.99,
        bio_label="push_press",
        bio_conf=0.99,
        bio_reason="",
        squat_label=None,
        squat_conf=0.0,
        olympic_pred=None,
        olympic_conf=0.0,
        bodyweight_router_label="handstand_push_up",
        bodyweight_router_conf=0.96,
    )

    populate_router_scores(
        add_router_score,
        raw_label="push_press",
        base_conf=0.99,
        bio_label="push_press",
        bio_conf=0.99,
        squat_label=None,
        squat_conf=0.0,
        olympic_pred=None,
        olympic_conf=0.0,
        bodyweight_router_label="handstand_push_up",
        bodyweight_router_conf=0.96,
        bodyweight_high_conf=bodyweight_high_conf,
        truly_explosive=False,
    )

    (
        winner,
        score,
        router_v6_label,
        router_v6_conf,
        decision,
    ) = finalize_router_scores(router_scores)

    assert winner == "handstand_push_up"
    assert score == 1.44
    assert router_v6_label == "handstand_push_up"
    assert router_v6_conf == 0.72
    assert decision == "score_winner"


def test_router_audit_empty_scores_are_no_scores():
    assert finalize_router_scores({}) == (
        None,
        0.0,
        None,
        0.0,
        "no_scores",
    )
