from __future__ import annotations

from collections import defaultdict
from typing import Any


LABEL_TO_FAMILY = {
    "squat": "squat",
    "squat_back": "squat",
    "squat_front": "squat",
    "overhead_squat": "squat",

    "clean": "olympic",
    "clean_and_jerk": "olympic",
    "snatch": "olympic",
    "split_jerk": "olympic",

    "bench_press": "press",
    "strict_press": "press",
    "push_press": "press",
    "thruster": "press",

    "pull_up": "bodyweight",
    "push_up": "bodyweight",
    "handstand_push_up": "bodyweight",
    "burpee": "bodyweight",
    "muscle_up": "bodyweight",

    "deadlift": "hinge",
}

VERIFIED_OLY_DECISIONS = {
    "agreement",
    "clean_rescue_from_weak_snatch",
    "standalone_split_from_cj",
    "clean_and_jerk_high_conf_rescue",
    "clean_and_jerk_shape_rescue",
}

BODYWEIGHT_LABELS = {
    "pull_up",
    "push_up",
    "handstand_push_up",
    "burpee",
    "muscle_up",
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


def classify_family_shadow(
    *,
    candidates: dict[str, Any],
    truly_explosive: bool,
    explosive_score: float,
    looks_clean_only: bool,
    looks_cj: bool,
    looks_split: bool,
    looks_thruster: bool,
    strong_overhead: bool,
) -> dict[str, Any]:
    """
    Shadow family router V2.

    Broad classifiers supply weak family evidence.
    Specialized routers and verified movement events supply strong evidence.
    No production result is changed.
    """

    base_label, base_conf, _ = _candidate(candidates, "base")
    bio_label, bio_conf, _ = _candidate(candidates, "biomechanics")
    squat_label, squat_conf, _ = _candidate(candidates, "squat_router")
    oly_label, oly_conf, _ = _candidate(candidates, "olympic_router")
    v5_label, v5_conf, v5_decision = _candidate(
        candidates,
        "router_v5",
    )
    bw_label, bw_conf, _ = _candidate(
        candidates,
        "bodyweight_router",
    )

    base_family = LABEL_TO_FAMILY.get(base_label)
    bio_family = LABEL_TO_FAMILY.get(bio_label)
    squat_family = LABEL_TO_FAMILY.get(squat_label)
    oly_family = LABEL_TO_FAMILY.get(oly_label)
    v5_family = LABEL_TO_FAMILY.get(v5_label)
    bw_family = LABEL_TO_FAMILY.get(bw_label)

    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(
        family: str | None,
        score: float,
        source: str,
        reason: str,
    ) -> None:
        if not family or score <= 0.0:
            return

        scores[family] += float(score)
        evidence[family].append({
            "source": source,
            "score": round(float(score), 3),
            "reason": reason,
        })

    # ---------------------------------------------------------
    # Broad evidence: one discounted contribution per source.
    # ---------------------------------------------------------
    add(
        base_family,
        0.35 * base_conf,
        "base",
        f"discounted broad label: {base_label}",
    )
    add(
        bio_family,
        0.35 * bio_conf,
        "biomechanics",
        f"discounted broad label: {bio_label}",
    )

    if base_family and base_family == bio_family:
        add(
            base_family,
            0.15,
            "broad_agreement",
            "base and biomechanics agree on family",
        )

    # ---------------------------------------------------------
    # Specialized squat evidence.
    # Router V5 may return the same squat variant as a fallback.
    # ---------------------------------------------------------
    squat_supported = (
        squat_family == "squat"
        and (
            base_family == "squat"
            or bio_family == "squat"
            or v5_family == "squat"
        )
    )

    if squat_supported:
        add(
            "squat",
            squat_conf,
            "squat_router",
            f"specialized squat variant: {squat_label}",
        )

        if v5_family == "squat":
            add(
                "squat",
                0.35 * v5_conf,
                "router_v5",
                f"V5 squat support: {v5_label}",
            )

    if (
        squat_label == "overhead_squat"
        and squat_conf >= 0.70
        and strong_overhead
        and v5_label == "overhead_squat"
    ):
        add(
            "squat",
            1.10,
            "overhead_squat_event",
            "sustained overhead squat with specialized agreement",
        )

    # ---------------------------------------------------------
    # Verified Olympic evidence.
    # ---------------------------------------------------------
    verified_v5_olympic = (
        v5_family == "olympic"
        and v5_decision in VERIFIED_OLY_DECISIONS
    )

    strong_olympic_event = (
        verified_v5_olympic
        or (
            truly_explosive
            and explosive_score >= 60.0
            and (
                looks_clean_only
                or looks_cj
                or looks_split
                or (
                    oly_family == "olympic"
                    and oly_conf >= 0.75
                )
            )
        )
    )

    if strong_olympic_event:
        if oly_family == "olympic":
            add(
                "olympic",
                0.70 * oly_conf,
                "olympic_router",
                f"Olympic candidate: {oly_label}",
            )

        if verified_v5_olympic:
            add(
                "olympic",
                v5_conf,
                "router_v5",
                f"verified V5 decision: {v5_decision}",
            )
            add(
                "olympic",
                0.45,
                "verified_event",
                "verified Olympic phase decision",
            )

        if (
            oly_family == "olympic"
            and v5_family == "olympic"
            and oly_label == v5_label
        ):
            add(
                "olympic",
                0.25,
                "olympic_agreement",
                "Olympic Router and V5 agree",
            )

    # ---------------------------------------------------------
    # Bodyweight evidence.
    # Strong specialized confidence can override orientation errors.
    # ---------------------------------------------------------
    verified_split_jerk = (
        verified_v5_olympic
        and v5_decision == "standalone_split_from_cj"
    )

    false_pull_up_during_press = (
        bw_label == "pull_up"
        and base_family == "press"
        and bio_family == "press"
        and v5_family != "bodyweight"
    )

    bodyweight_supported = (
        bw_family == "bodyweight"
        and bw_conf >= 0.90
        and not verified_split_jerk
        and not false_pull_up_during_press
    )

    bodyweight_clean_conflict = (
        bodyweight_supported
        and bw_conf >= 0.98
        and v5_decision == "clean_rescue_from_weak_snatch"
    )

    if bodyweight_supported:
        add(
            "bodyweight",
            bw_conf,
            "bodyweight_router",
            f"specialized bodyweight candidate: {bw_label}",
        )

        if v5_family == "bodyweight":
            add(
                "bodyweight",
                v5_conf,
                "router_v5",
                f"V5 bodyweight support: {v5_label}",
            )

        if bw_conf >= 0.95:
            add(
                "bodyweight",
                0.45,
                "bodyweight_strength",
                "very high bodyweight confidence",
            )

        if bw_label in {
            "push_up",
            "handstand_push_up",
            "burpee",
            "muscle_up",
        }:
            add(
                "bodyweight",
                0.40,
                "distinct_bodyweight_variant",
                f"distinct bodyweight variant: {bw_label}",
            )

    # ---------------------------------------------------------
    # Press evidence.
    # Do not count broad press confidence twice.
    # ---------------------------------------------------------
    press_sources = [
        confidence
        for family, confidence in (
            (base_family, base_conf),
            (bio_family, bio_conf),
            (v5_family, v5_conf),
        )
        if family == "press"
    ]

    if press_sources:
        add(
            "press",
            max(press_sources),
            "press_specialist",
            "strongest press-family candidate",
        )

        if base_family == "press" and bio_family == "press":
            add(
                "press",
                0.20,
                "press_agreement",
                "base and biomechanics agree on press family",
            )

    # ---------------------------------------------------------
    # Hinge evidence.
    # Do not re-add the same base/biomechanics confidence.
    # ---------------------------------------------------------
    if base_family == "hinge" and bio_family == "hinge":
        add(
            "hinge",
            max(base_conf, bio_conf),
            "hinge_consensus",
            "base and biomechanics agree on hinge",
        )
        add(
            "hinge",
            0.20,
            "hinge_agreement",
            "two-source hinge agreement",
        )
    elif base_family == "hinge":
        add(
            "hinge",
            0.60 * base_conf,
            "base",
            "single-source hinge evidence",
        )
    elif bio_family == "hinge":
        add(
            "hinge",
            0.60 * bio_conf,
            "biomechanics",
            "single-source hinge evidence",
        )

    # ---------------------------------------------------------
    # Family-level suppression from strong specialized evidence.
    # ---------------------------------------------------------
    if bodyweight_supported:
        for family in ("squat", "press", "hinge", "olympic"):
            scores[family] *= 0.60

    if verified_v5_olympic and not bodyweight_clean_conflict:
        for family in ("squat", "press", "hinge", "bodyweight"):
            scores[family] *= 0.60

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    winner = ranked[0] if ranked else ("unknown", 0.0)
    runner_up = ranked[1] if len(ranked) > 1 else (None, 0.0)

    return {
        "version": "family_router_shadow_v2",
        "family": winner[0],
        "score": round(float(winner[1]), 3),
        "margin": round(
            float(winner[1] - runner_up[1]),
            3,
        ),
        "runner_up": {
            "family": runner_up[0],
            "score": round(float(runner_up[1]), 3),
        },
        "scores": {
            family: round(float(score), 3)
            for family, score in ranked
        },
        "evidence": dict(evidence),
        "signals": {
            "truly_explosive": bool(truly_explosive),
            "explosive_score": round(
                float(explosive_score),
                3,
            ),
            "looks_clean_only": bool(looks_clean_only),
            "looks_cj": bool(looks_cj),
            "looks_split": bool(looks_split),
            "looks_thruster": bool(looks_thruster),
            "strong_overhead": bool(strong_overhead),
            "verified_v5_olympic": verified_v5_olympic,
            "verified_split_jerk": verified_split_jerk,
            "false_pull_up_during_press": false_pull_up_during_press,
            "bodyweight_supported": bodyweight_supported,
            "bodyweight_clean_conflict": bodyweight_clean_conflict,
            "v5_decision": v5_decision,
        },
    }
