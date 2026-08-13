from __future__ import annotations

from typing import Any, Callable

from app.ml.movement_signatures import (
    OLYMPIC_LABELS,
)


ROUTER_SCORE_BODYWEIGHT_LABELS = {
    "push_up",
    "pull_up",
    "handstand_push_up",
}

ROUTER_SCORE_SQUAT_LABELS = {
    "squat_back",
    "squat_front",
    "overhead_squat",
}


def build_router_score_flags(
    *,
    raw_label: str | None,
    base_conf: float,
    bio_label: str | None,
    bio_conf: float,
    looks_thruster: bool,
    bodyweight_router_label: str | None,
    bodyweight_router_conf: float,
) -> tuple[bool, bool]:
    """Build flags used to weight the audit router scores."""
    strong_bench_evidence = (
        raw_label == "bench_press"
        and float(base_conf or 0.0) >= 0.95
        and bio_label == "bench_press"
        and float(bio_conf or 0.0) >= 0.95
        and not bool(looks_thruster)
    )

    bodyweight_high_conf = (
        bodyweight_router_label in ROUTER_SCORE_BODYWEIGHT_LABELS
        and float(bodyweight_router_conf or 0.0) >= 0.95
        and not strong_bench_evidence
    )

    return strong_bench_evidence, bodyweight_high_conf


def populate_router_scores(
    add_router_score: Callable[[str | None, float, str], None],
    *,
    raw_label: str | None,
    base_conf: float,
    bio_label: str | None,
    bio_conf: float,
    squat_label: str | None,
    squat_conf: float,
    olympic_pred: str | None,
    olympic_conf: float,
    bodyweight_router_label: str | None,
    bodyweight_router_conf: float,
    bodyweight_high_conf: bool,
    truly_explosive: bool,
) -> None:
    """Populate audit scores without changing routing decisions."""
    raw_weight = 0.35 if bodyweight_high_conf else 1.0
    bio_weight = 0.35 if bodyweight_high_conf else 1.0

    add_router_score(
        raw_label,
        float(base_conf or 0.0) * raw_weight,
        "raw_model",
    )
    add_router_score(
        bio_label,
        float(bio_conf or 0.0) * bio_weight,
        "bio_model",
    )
    add_router_score(squat_label, squat_conf, "squat_router")
    add_router_score(olympic_pred, olympic_conf, "olympic_router")
    add_router_score(
        bodyweight_router_label,
        bodyweight_router_conf,
        "bodyweight_router",
    )

    if bodyweight_router_label in ROUTER_SCORE_BODYWEIGHT_LABELS:
        add_router_score(
            bodyweight_router_label,
            float(bodyweight_router_conf or 0.0) * 0.50,
            "bodyweight_bonus",
        )

    if squat_label in ROUTER_SCORE_SQUAT_LABELS:
        add_router_score(
            squat_label,
            float(squat_conf or 0.0) * 0.25,
            "squat_bonus",
        )

    if olympic_pred in OLYMPIC_LABELS and truly_explosive:
        add_router_score(
            olympic_pred,
            float(olympic_conf or 0.0) * 0.35,
            "olympic_explosive_bonus",
        )


def finalize_router_scores(
    router_scores: dict[str, dict[str, Any]],
) -> tuple[str | None, float, str | None, float, str]:
    """Select the audit score winner and derive the Router V6 confidence."""
    router_score_winner = None
    router_score_value = 0.0
    router_v6_label = None
    router_v6_conf = 0.0
    router_v6_decision = "no_scores"

    if router_scores:
        router_score_winner, score_info = max(
            router_scores.items(),
            key=lambda item: float(item[1].get("score", 0.0)),
        )

        router_score_value = float(score_info.get("score", 0.0))

        router_v6_label = router_score_winner
        router_v6_conf = min(
            0.99,
            max(0.01, router_score_value / 2.0),
        )
        router_v6_decision = "score_winner"

    return (
        router_score_winner,
        router_score_value,
        router_v6_label,
        router_v6_conf,
        router_v6_decision,
    )


def initialize_router_audit(
    *,
    raw_label: str | None,
    base_conf: float,
    bio_label: str | None,
    bio_conf: float,
    bio_reason: str | None,
    squat_label: str | None,
    squat_conf: float,
    olympic_pred: str | None,
    olympic_conf: float,
    bodyweight_router_label: str | None,
    bodyweight_router_conf: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], Callable, Callable]:
    """Initialize routing trace and score accumulation for audit/debug."""
    routing_trace: list[dict[str, Any]] = []

    def trace_route(
        stage: str,
        label: str | None = None,
        conf: float | None = None,
        reason: str | None = None,
    ) -> None:
        routing_trace.append({
            "stage": stage,
            "label": str(label if label is not None else ""),
            "conf": round(float(conf or 0.0), 3),
            "reason": str(reason or ""),
        })

    trace_route("raw_model", raw_label, base_conf)
    trace_route("bio_model", bio_label, bio_conf, bio_reason)
    trace_route("squat_router", squat_label, squat_conf)
    trace_route("olympic_router", olympic_pred, olympic_conf)
    trace_route(
        "bodyweight_router",
        bodyweight_router_label,
        bodyweight_router_conf,
    )

    router_scores: dict[str, Any] = {}

    def add_router_score(
        label: str | None,
        score: float,
        source: str,
    ) -> None:
        if not label:
            return

        label = str(label)
        score = float(score or 0.0)

        if label not in router_scores:
            router_scores[label] = {
                "score": 0.0,
                "sources": [],
            }

        router_scores[label]["score"] += score
        router_scores[label]["sources"].append({
            "source": source,
            "score": round(score, 3),
        })

    return routing_trace, router_scores, trace_route, add_router_score
