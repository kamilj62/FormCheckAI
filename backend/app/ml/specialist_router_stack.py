from __future__ import annotations

from typing import Any

from app.ml.movement_signatures import (
    BODYWEIGHT_LABELS,
    OLYMPIC_LABELS,
    PRESS_LABELS,
    SQUAT_LABELS,
)

VERIFIED_OLY_DECISIONS = {
    "agreement",
    "clean_rescue_from_weak_snatch",
    "standalone_split_from_cj",
    "clean_and_jerk_high_conf_rescue",
    "clean_and_jerk_shape_rescue",
}


def _candidate(
    candidates: dict[str, Any],
    source: str,
) -> tuple[str | None, float, str]:
    item = candidates.get(source) or {}

    label = item.get("label")
    decision = str(item.get("decision") or "")

    try:
        confidence = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return label, confidence, decision


def _signals(value: dict[str, Any] | None) -> dict[str, Any]:
    signals = (value or {}).get("signals") or {}
    return signals if isinstance(signals, dict) else {}


def _route(
    *,
    router: str,
    label: str | None,
    score: float,
    source: str | None,
    evidence: list[str],
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "router": router,
        "label": label,
        "score": round(float(score or 0.0), 3),
        "source": source,
        "evidence": evidence,
        "features": features or {},
        "eligible": bool(label and score > 0.0),
    }


def classify_specialist_routers(
    *,
    candidates: dict[str, Any] | None,
    press_variant: dict[str, Any] | None,
    family_shadow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Four-router classification stack.

    Each specialist proposes one exact label. The winner is chosen only among
    these lanes: press, bodyweight, Olympic, and squat.
    """

    candidate_map = candidates or {}
    family_signals = _signals(family_shadow)

    base_label, base_conf, _ = _candidate(candidate_map, "base")
    bio_label, bio_conf, _ = _candidate(candidate_map, "biomechanics")
    squat_label, squat_conf, _ = _candidate(candidate_map, "squat_router")
    olympic_label, olympic_conf, _ = _candidate(
        candidate_map,
        "olympic_router",
    )
    v5_label, v5_conf, v5_decision = _candidate(
        candidate_map,
        "router_v5",
    )
    bodyweight_label, bodyweight_conf, _ = _candidate(
        candidate_map,
        "bodyweight_router",
    )
    protected_label, protected_conf, _ = _candidate(
        candidate_map,
        "protected_evidence",
    )
    protected_reason = str(
        (candidate_map.get("protected_evidence") or {}).get("reason")
        or ""
    )

    routes: dict[str, dict[str, Any]] = {}

    press_result = press_variant or {}
    press_label = press_result.get("label")
    press_score = 0.0
    press_evidence: list[str] = []

    if press_result.get("eligible") and press_label in PRESS_LABELS:
        try:
            press_score = float(press_result.get("score") or 0.0)
        except (TypeError, ValueError):
            press_score = 0.0
        press_evidence.append("press variant router")

    for label, confidence, source in (
        (base_label, base_conf, "base"),
        (bio_label, bio_conf, "biomechanics"),
        (protected_label, protected_conf, "protected_evidence"),
    ):
        if label in PRESS_LABELS:
            if press_label is None:
                press_label = label
            press_score += 0.35 * float(confidence or 0.0)
            press_evidence.append(f"{source} supports {label}")

    routes["press"] = _route(
        router="press",
        label=press_label if press_label in PRESS_LABELS else None,
        score=press_score,
        source=(
            "press_variant_router"
            if press_result.get("eligible")
            else None
        ),
        evidence=press_evidence,
        features=press_result.get("features") or {},
    )

    false_pull_up_during_press = bool(
        family_signals.get("false_pull_up_during_press")
    )
    bodyweight_choice = None
    bodyweight_score = 0.0
    bodyweight_evidence: list[str] = []

    if bodyweight_label in BODYWEIGHT_LABELS and not false_pull_up_during_press:
        bodyweight_choice = bodyweight_label
        bodyweight_score += bodyweight_conf
        bodyweight_evidence.append("bodyweight model")

    if (
        protected_label in BODYWEIGHT_LABELS
        and not false_pull_up_during_press
        and protected_reason in {
            "push_up_bodyweight_pattern",
            "handstand_push_up_bodyweight_pattern",
            "pull_up_bodyweight_pattern",
            "router_v6_bodyweight_winner",
            "muscle_up_bodyweight_pattern",
            "burpee_bodyweight_pattern",
        }
    ):
        bodyweight_choice = protected_label
        bodyweight_score += max(protected_conf, 0.86)
        bodyweight_evidence.append(f"protected {protected_reason}")

    routes["bodyweight"] = _route(
        router="bodyweight",
        label=bodyweight_choice,
        score=bodyweight_score,
        source=(
            "protected_evidence"
            if protected_label == bodyweight_choice
            else "bodyweight_router"
        ),
        evidence=bodyweight_evidence,
        features={
            "false_pull_up_during_press": false_pull_up_during_press,
        },
    )

    olympic_choice = None
    olympic_score = 0.0
    olympic_source = None
    olympic_evidence: list[str] = []

    if v5_label in OLYMPIC_LABELS and v5_decision in VERIFIED_OLY_DECISIONS:
        olympic_choice = v5_label
        olympic_score += v5_conf + 0.35
        olympic_source = "router_v5"
        olympic_evidence.append(f"verified router_v5 {v5_decision}")

    if olympic_label in OLYMPIC_LABELS:
        if olympic_choice is None:
            olympic_choice = olympic_label
            olympic_source = "olympic_router"

        olympic_score += 0.75 * olympic_conf
        olympic_evidence.append("olympic model")

    routes["olympic"] = _route(
        router="olympic",
        label=olympic_choice,
        score=olympic_score,
        source=olympic_source,
        evidence=olympic_evidence,
        features={
            "v5_decision": v5_decision,
        },
    )

    squat_choice = None
    squat_score = 0.0
    squat_evidence: list[str] = []

    if squat_label in SQUAT_LABELS:
        squat_choice = squat_label
        squat_score += squat_conf
        squat_evidence.append("squat model")

        if base_label in SQUAT_LABELS or bio_label in SQUAT_LABELS:
            squat_score += 0.35
            squat_evidence.append("broad squat support")

    routes["squat"] = _route(
        router="squat",
        label=squat_choice,
        score=squat_score,
        source="squat_router" if squat_choice else None,
        evidence=squat_evidence,
    )

    ranked = sorted(
        routes.values(),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )

    winner = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None

    winner_score = float((winner or {}).get("score") or 0.0)
    runner_score = float((runner_up or {}).get("score") or 0.0)

    return {
        "version": "specialist_router_stack_v1",
        "winner": winner,
        "runner_up": runner_up,
        "margin": round(winner_score - runner_score, 3),
        "routers": routes,
        "eligible": bool(winner and winner.get("eligible")),
    }
