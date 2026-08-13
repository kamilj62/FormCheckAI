from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.ml.movement_signatures import (
    BODYWEIGHT_LABELS,
    OLYMPIC_LABELS,
    PRESS_LABELS,
    SQUAT_LABELS,
)


SQUAT_VARIANTS = SQUAT_LABELS - {"squat", "back_squat", "front_squat"}
OLY_LABELS = OLYMPIC_LABELS


def _candidate(
    candidates: dict[str, Any],
    source: str,
) -> tuple[str | None, float]:
    item = candidates.get(source) or {}
    label = item.get("label")

    try:
        confidence = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return label, confidence


def _decision(
    candidates: dict[str, Any],
    source: str,
) -> str:
    item = candidates.get(source) or {}
    return str(item.get("decision") or "")


def arbitrate_shadow(
    *,
    candidates: dict[str, Any],
    truly_explosive: bool,
    explosive_score: float,
    looks_clean_only: bool,
    looks_cj: bool,
    looks_split: bool,
    looks_thruster: bool,
    strong_overhead: bool,
    wrist_overhead_ratio: float,
) -> dict[str, Any]:
    """
    Experimental centralized router V2.

    Shadow-only. This function does not alter the current production result.

    Design:
    1. Broad classifiers provide discounted family evidence.
    2. Specialized routers vote only when their family is eligible.
    3. Correlated Olympic Router/V5 predictions are not added twice.
    4. Biomechanical event shapes provide eligibility and modest bonuses.
    """

    base_label, base_conf = _candidate(candidates, "base")
    bio_label, bio_conf = _candidate(candidates, "biomechanics")
    squat_label, squat_conf = _candidate(candidates, "squat_router")
    olympic_label, olympic_conf = _candidate(
        candidates,
        "olympic_router",
    )
    v5_label, v5_conf = _candidate(candidates, "router_v5")
    v5_decision = _decision(candidates, "router_v5")

    bodyweight_label, bodyweight_conf = _candidate(
        candidates,
        "bodyweight_router",
    )

    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(
        label: str | None,
        score: float,
        source: str,
        reason: str,
    ) -> None:
        if not label or score <= 0.0:
            return

        scores[label] += float(score)
        evidence[label].append(
            {
                "source": source,
                "score": round(float(score), 3),
                "reason": reason,
            }
        )

    # ---------------------------------------------------------
    # Broad evidence
    # ---------------------------------------------------------
    # Broad classifiers are useful for family detection but are frequently
    # overconfident about the exact movement, so their votes are discounted.
    add(
        base_label,
        0.50 * base_conf,
        "base",
        "discounted broad classifier",
    )
    add(
        bio_label,
        0.50 * bio_conf,
        "biomechanics",
        "discounted broad biomechanics classifier",
    )

    if base_label and base_label == bio_label:
        add(
            base_label,
            0.20,
            "broad_agreement",
            "base and biomechanics agree",
        )

    # ---------------------------------------------------------
    # Olympic event strength
    # ---------------------------------------------------------
    verified_v5_olympic = (
        v5_label in OLY_LABELS
        and v5_decision in {
            "agreement",
            "clean_rescue_from_weak_snatch",
            "standalone_split_from_cj",
            "clean_and_jerk_high_conf_rescue",
        }
    )

    strong_olympic_event = (
        bool(truly_explosive)
        and float(explosive_score) >= 80.0
        and (
            (
                olympic_label in OLY_LABELS
                and olympic_conf >= 0.80
            )
            or (
                verified_v5_olympic
                and v5_conf >= 0.70
            )
        )
    )

    # ---------------------------------------------------------
    # Squat family
    # ---------------------------------------------------------
    broad_squat_support = (
        base_label in SQUAT_LABELS
        or bio_label in SQUAT_LABELS
    )

    squat_eligible = (
        squat_label in SQUAT_VARIANTS
        and broad_squat_support
        and not strong_olympic_event
    )

    if squat_eligible:
        add(
            squat_label,
            squat_conf,
            "squat_router",
            "squat family eligible",
        )

        exact_variant_agreement = (
            squat_label == base_label
            or squat_label == bio_label
        )

        generic_squat_agreement = (
            base_label == "squat"
            or bio_label == "squat"
        )

        if exact_variant_agreement or generic_squat_agreement:
            add(
                squat_label,
                0.35,
                "squat_family_agreement",
                "broad squat evidence supports specialized variant",
            )

    # Overhead squat is often misread as push press or pull-up by broad
    # models. Sustained overhead posture plus a confident squat variant is
    # stronger evidence than those broad labels.
    overhead_squat_eligible = (
        squat_label == "overhead_squat"
        and squat_conf >= 0.70
        and bool(strong_overhead)
        and not strong_olympic_event
    )

    if overhead_squat_eligible:
        add(
            "overhead_squat",
            squat_conf + 0.75,
            "overhead_squat_gate",
            "sustained overhead posture with squat variant",
        )

    # ---------------------------------------------------------
    # Olympic family
    # ---------------------------------------------------------
    olympic_eligible = (
        strong_olympic_event
        or (
            bool(truly_explosive)
            and float(explosive_score) >= 60.0
            and (
                bool(looks_clean_only)
                or bool(looks_cj)
                or bool(looks_split)
            )
        )
    )

    if olympic_eligible:
        # Do not add Olympic Router and Router V5 as two independent full
        # votes when they agree. Router V5 frequently derives from the same
        # Olympic prediction.
        if (
            olympic_label in OLY_LABELS
            and olympic_label == v5_label
        ):
            specialist_score = max(olympic_conf, v5_conf)

            add(
                olympic_label,
                specialist_score,
                "olympic_consensus",
                "maximum correlated Olympic confidence",
            )

            if v5_decision == "agreement":
                add(
                    olympic_label,
                    0.35,
                    "olympic_agreement",
                    "Olympic Router and Router V5 agree",
                )
        else:
            if olympic_label in OLY_LABELS:
                add(
                    olympic_label,
                    olympic_conf,
                    "olympic_router",
                    "Olympic family eligible",
                )

            if verified_v5_olympic:
                add(
                    v5_label,
                    v5_conf,
                    "router_v5",
                    f"verified Router V5 decision: {v5_decision}",
                )

    if (
        bool(looks_clean_only)
        and bool(truly_explosive)
        and v5_label == "clean"
        and v5_decision == "clean_rescue_from_weak_snatch"
    ):
        add(
            "clean",
            0.60,
            "clean_shape",
            "verified explosive clean-only event",
        )

    if (
        bool(looks_cj)
        and bool(truly_explosive)
        and (
            olympic_label == "clean_and_jerk"
            or v5_label == "clean_and_jerk"
        )
    ):
        add(
            "clean_and_jerk",
            0.50,
            "cj_shape",
            "explosive clean-and-jerk event",
        )

    if (
        bool(looks_split)
        and bool(truly_explosive)
        and v5_label == "split_jerk"
    ):
        add(
            "split_jerk",
            0.50,
            "split_shape",
            "verified explosive split recovery",
        )

    # ---------------------------------------------------------
    # Press family
    # ---------------------------------------------------------
    # Push press can be mistaken for C&J by both Olympic routers because
    # those predictions are correlated. Require broad push-press support
    # and absence of a verified Olympic phase sequence.
    push_press_support = max(
        base_conf if base_label == "push_press" else 0.0,
        bio_conf if bio_label == "push_press" else 0.0,
    )

    push_press_eligible = (
        push_press_support >= 0.50
        and not bool(looks_clean_only)
        and not bool(looks_cj)
        and not bool(looks_split)
    )

    if push_press_eligible:
        add(
            "push_press",
            push_press_support + 0.75,
            "push_press_gate",
            "press evidence without verified Olympic sequence",
        )

    # Thruster is a press-family movement in this app. It shares squat and
    # overhead features with Olympic clips, so give the press lane an explicit
    # thruster vote when the thruster shape is present and no clean/split/C&J
    # sequence has been verified.
    thruster_support = max(
        base_conf if base_label == "thruster" else 0.0,
        bio_conf if bio_label == "thruster" else 0.0,
    )

    thruster_eligible = (
        bool(looks_thruster)
        and not bool(looks_clean_only)
        and not bool(looks_cj)
        and not bool(looks_split)
        and not strong_olympic_event
    )

    if thruster_eligible:
        add(
            "thruster",
            max(thruster_support, 0.70) + 0.65,
            "thruster_press_gate",
            "thruster belongs to press family without verified Olympic sequence",
        )

    # ---------------------------------------------------------
    # Bodyweight family
    # ---------------------------------------------------------
    bodyweight_eligible = (
        bodyweight_label in BODYWEIGHT_LABELS
        and (
            base_label in BODYWEIGHT_LABELS
            or bio_label in BODYWEIGHT_LABELS
        )
    )

    if bodyweight_eligible:
        add(
            bodyweight_label,
            bodyweight_conf,
            "bodyweight_router",
            "bodyweight family supported by broad classifier",
        )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    winner = ranked[0] if ranked else ("unknown", 0.0)
    runner_up = ranked[1] if len(ranked) > 1 else (None, 0.0)

    return {
        "version": "central_router_shadow_v2",
        "label": winner[0],
        "score": round(float(winner[1]), 3),
        "margin": round(
            float(winner[1] - runner_up[1]),
            3,
        ),
        "runner_up": {
            "label": runner_up[0],
            "score": round(float(runner_up[1]), 3),
        },
        "scores": {
            label: round(float(score), 3)
            for label, score in ranked
        },
        "evidence": dict(evidence),
        "eligibility": {
            "squat": squat_eligible,
            "overhead_squat": overhead_squat_eligible,
            "olympic": olympic_eligible,
            "push_press": push_press_eligible,
            "thruster": thruster_eligible,
            "bodyweight": bodyweight_eligible,
        },
        "signals": {
            "truly_explosive": bool(truly_explosive),
            "explosive_score": round(
                float(explosive_score),
                3,
            ),
            "strong_olympic_event": strong_olympic_event,
            "v5_decision": v5_decision,
            "looks_clean_only": bool(looks_clean_only),
            "looks_cj": bool(looks_cj),
            "looks_split": bool(looks_split),
            "looks_thruster": bool(looks_thruster),
            "strong_overhead": bool(strong_overhead),
            "wrist_overhead_ratio": round(
                float(wrist_overhead_ratio),
                3,
            ),
        },
    }
