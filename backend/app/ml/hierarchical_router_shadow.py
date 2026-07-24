from __future__ import annotations

from typing import Any


SQUAT_LABELS = {
    "squat",
    "squat_back",
    "squat_front",
    "overhead_squat",
}

OLYMPIC_LABELS = {
    "clean",
    "clean_and_jerk",
    "snatch",
    "split_jerk",
}

BODYWEIGHT_LABELS = {
    "pull_up",
    "push_up",
    "handstand_push_up",
    "burpee",
    "muscle_up",
}

PRESS_LABELS = {
    "bench_press",
    "strict_press",
    "push_press",
    "thruster",
}


def _candidate(
    candidates: dict[str, Any],
    source: str,
) -> tuple[str | None, float, str | None]:
    item = candidates.get(source) or {}

    label = item.get("label")
    decision = item.get("decision")

    try:
        confidence = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return label, confidence, decision


def classify_hierarchical_shadow(
    *,
    family_shadow: dict[str, Any] | None,
    press_variant_shadow: dict[str, Any] | None,
    routing_candidates: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Shadow-only hierarchical exact-label router.

    Stage 1 chooses a movement family.
    Stage 2 chooses a label only from that family.

    This does not modify the production result.
    """

    family_result = family_shadow or {}
    press_result = press_variant_shadow or {}
    candidates = routing_candidates or {}

    family = family_result.get("family")

    base_label, base_conf, _ = _candidate(
        candidates,
        "base",
    )
    bio_label, bio_conf, _ = _candidate(
        candidates,
        "biomechanics",
    )
    squat_label, squat_conf, _ = _candidate(
        candidates,
        "squat_router",
    )
    olympic_label, olympic_conf, _ = _candidate(
        candidates,
        "olympic_router",
    )
    v5_label, v5_conf, v5_decision = _candidate(
        candidates,
        "router_v5",
    )
    bodyweight_label, bodyweight_conf, _ = _candidate(
        candidates,
        "bodyweight_router",
    )
    protected_label, protected_conf, _ = _candidate(
        candidates,
        "protected_evidence",
    )
    protected_reason = str(
        (candidates.get("protected_evidence") or {}).get("reason")
        or ""
    )

    label = None
    confidence = 0.0
    source = None
    reason = None
    alternatives: list[dict[str, Any]] = []

    if family == "press":
        press_label = press_result.get("label")

        try:
            press_score = float(
                press_result.get("score") or 0.0
            )
        except (TypeError, ValueError):
            press_score = 0.0

        if press_label in PRESS_LABELS:
            label = press_label
            confidence = press_score
            source = "press_variant_shadow"
            reason = "press family variant selection"

    elif family == "squat":
        if squat_label in SQUAT_LABELS:
            label = squat_label
            confidence = squat_conf
            source = "squat_router"
            reason = "specialized squat variant"

        if v5_label in SQUAT_LABELS:
            alternatives.append({
                "label": v5_label,
                "confidence": round(v5_conf, 3),
                "source": "router_v5",
                "decision": v5_decision,
            })

        if label is None and v5_label in SQUAT_LABELS:
            label = v5_label
            confidence = v5_conf
            source = "router_v5"
            reason = "V5 squat fallback"

        if label is None:
            for candidate_label, candidate_conf, candidate_source in (
                (base_label, base_conf, "base"),
                (bio_label, bio_conf, "biomechanics"),
            ):
                if candidate_label in SQUAT_LABELS:
                    label = candidate_label
                    confidence = candidate_conf
                    source = candidate_source
                    reason = "broad squat fallback"
                    break

    elif family == "olympic":
        # Router V5 contains phase-based Olympic decisions and should
        # take priority when it returns an Olympic label.
        if v5_label in OLYMPIC_LABELS:
            label = v5_label
            confidence = v5_conf
            source = "router_v5"
            reason = (
                f"phase-aware Olympic decision: "
                f"{v5_decision or 'unspecified'}"
            )

        if olympic_label in OLYMPIC_LABELS:
            alternatives.append({
                "label": olympic_label,
                "confidence": round(olympic_conf, 3),
                "source": "olympic_router",
            })

        if label is None and olympic_label in OLYMPIC_LABELS:
            label = olympic_label
            confidence = olympic_conf
            source = "olympic_router"
            reason = "specialized Olympic router"

    elif family == "bodyweight":
        contextual_bodyweight_labels = {
            "muscle_up",
            "burpee",
            "handstand_push_up",
        }

        # Existing geometry detectors can distinguish variants that the
        # generic bodyweight router collapses into pull_up or push_up.
        if protected_label in contextual_bodyweight_labels:
            label = protected_label
            confidence = protected_conf
            source = "protected_evidence"
            reason = (
                f"contextual bodyweight detector: "
                f"{protected_reason or 'unspecified'}"
            )

        if v5_label in BODYWEIGHT_LABELS:
            alternatives.append({
                "label": v5_label,
                "confidence": round(v5_conf, 3),
                "source": "router_v5",
                "decision": v5_decision,
            })

            if label is None:
                label = v5_label
                confidence = v5_conf
                source = "router_v5"
                reason = "contextual bodyweight variant"

        if bodyweight_label in BODYWEIGHT_LABELS:
            alternatives.append({
                "label": bodyweight_label,
                "confidence": round(bodyweight_conf, 3),
                "source": "bodyweight_router",
            })

        if label is None and bodyweight_label in BODYWEIGHT_LABELS:
            label = bodyweight_label
            confidence = bodyweight_conf
            source = "bodyweight_router"
            reason = "specialized bodyweight router"

    elif family == "hinge":
        label = "deadlift"
        confidence = max(
            base_conf if base_label == "deadlift" else 0.0,
            bio_conf if bio_label == "deadlift" else 0.0,
        )
        source = "hinge_family"
        reason = "deadlift is the current hinge-family variant"

    return {
        "version": "hierarchical_router_shadow_v2",
        "family": family,
        "label": label,
        "confidence": round(float(confidence), 3),
        "source": source,
        "reason": reason,
        "family_score": family_result.get("score"),
        "family_margin": family_result.get("margin"),
        "alternatives": alternatives,
        "eligible": label is not None,
    }
