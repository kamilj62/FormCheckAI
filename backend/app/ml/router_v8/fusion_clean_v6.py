from __future__ import annotations

from collections import defaultdict

from .locks import get_locks
from .models import RouterPrediction
from .state import RouterState


LABEL_FAMILY = {
    "squat": "squat",
    "squat_back": "squat",
    "squat_front": "squat",
    "overhead_squat": "squat",

    "clean": "olympic",
    "clean_and_jerk": "olympic",
    "snatch": "olympic",
    "split_jerk": "olympic",

    "bench_press": "press",
    "push_press": "press",
    "strict_press": "press",
    "thruster": "press",

    "push_up": "bodyweight",
    "pull_up": "bodyweight",
    "handstand_push_up": "bodyweight",
    "muscle_up": "bodyweight",
    "burpee": "bodyweight",

    "deadlift": "pull",
}


SPECIALIST_ROUTERS = {
    "squat",
    "olympic",
    "bodyweight",
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _empty_result() -> dict:
    return {
        "label": None,
        "confidence": 0.0,
        "decision": "no_evidence",
        "winning_family": None,
        "family_scores": {},
        "scores": {},
        "evidence": {},
        "locks": [],
    }


def _merged_generic(
    predictions: list[RouterPrediction],
) -> tuple[str | None, float, dict]:
    """
    Collapse base and biomechanics into one correlated opinion.
    """
    base = next(
        (
            p for p in predictions
            if p.router == "base"
            and p.label in LABEL_FAMILY
        ),
        None,
    )

    bio = next(
        (
            p for p in predictions
            if p.router == "biomechanics"
            and p.label in LABEL_FAMILY
        ),
        None,
    )

    if base is None and bio is None:
        return None, 0.0, {}

    if base is None:
        return (
            bio.label,
            _clamp(bio.confidence) * 0.85,
            {
                "source": "biomechanics",
                "reason": "single_generic_source",
            },
        )

    if bio is None:
        return (
            base.label,
            _clamp(base.confidence),
            {
                "source": "base",
                "reason": "single_generic_source",
            },
        )

    base_conf = _clamp(base.confidence)
    bio_conf = _clamp(bio.confidence)

    if base.label == bio.label:
        return (
            base.label,
            (base_conf + bio_conf) / 2.0,
            {
                "source": "base_biomechanics",
                "reason": "correlated_agreement",
            },
        )

    if base_conf >= bio_conf:
        return (
            base.label,
            base_conf * 0.70,
            {
                "source": "base",
                "reason": "generic_disagreement",
            },
        )

    return (
        bio.label,
        bio_conf * 0.70,
        {
            "source": "biomechanics",
            "reason": "generic_disagreement",
        },
    )


def _specialist_multiplier(
    prediction: RouterPrediction,
    state: RouterState,
) -> float:
    """
    General movement-family context.

    These multipliers adjust router credibility but never create labels.
    """
    multiplier = 1.0

    if prediction.router == "squat":
        generic_support = (
            state.raw_label in {
                "squat",
                "squat_back",
                "squat_front",
                "overhead_squat",
            }
            or state.bio_label in {
                "squat",
                "squat_back",
                "squat_front",
                "overhead_squat",
            }
        )

        if generic_support:
            multiplier += 0.08

        if (
            prediction.label == "overhead_squat"
            and float(state.wrist_overhead or 0.0) >= 0.70
        ):
            multiplier += 0.12


    elif prediction.router == "olympic":
        motion_support = (
            state.looks_clean
            or state.looks_cj
            or state.looks_split
            or state.truly_explosive
            or float(state.explosive_score or 0.0) >= 20.0
        )

        if motion_support:
            multiplier += 0.18

        if _clamp(prediction.confidence) >= 0.90:
            multiplier += 0.08

    elif prediction.router == "bodyweight":
        generic_support = (
            state.raw_label in {
                "push_up",
                "pull_up",
                "handstand_push_up",
                "muscle_up",
                "burpee",
            }
            or state.bio_label in {
                "push_up",
                "pull_up",
                "handstand_push_up",
                "muscle_up",
                "burpee",
            }
        )

        if generic_support:
            multiplier += 0.10
        else:
            multiplier -= 0.20

    return max(0.65, min(1.30, multiplier))


def fuse_predictions(
    predictions: list[RouterPrediction],
    state: RouterState | None = None,
) -> dict:
    if state is None:
        state = RouterState()

    locks = get_locks(state)

    if locks:
        winner_lock = locks[0]

        return {
            "label": winner_lock["label"],
            "confidence": float(winner_lock["confidence"]),
            "decision": "context_lock",
            "winning_family": LABEL_FAMILY.get(
                winner_lock["label"]
            ),
            "family_scores": {},
            "scores": {},
            "evidence": {},
            "locks": locks,
            "lock": winner_lock,
        }

    generic_label, generic_conf, generic_meta = _merged_generic(
        predictions
    )

    generic_family = (
        LABEL_FAMILY.get(generic_label)
        if generic_label
        else None
    )

    specialist_scores: dict[str, float] = {}
    label_scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[dict]] = defaultdict(list)

    for prediction in predictions:
        if prediction.router not in SPECIALIST_ROUTERS:
            continue

        if (
            not prediction.label
            or prediction.label not in LABEL_FAMILY
        ):
            continue

        confidence = _clamp(prediction.confidence)
        multiplier = _specialist_multiplier(
            prediction,
            state,
        )
        score = confidence * multiplier
        family = LABEL_FAMILY[prediction.label]

        # One specialist router represents each specialist family.
        specialist_scores[family] = max(
            specialist_scores.get(family, 0.0),
            score,
        )

        label_scores[prediction.label] = max(
            label_scores.get(prediction.label, 0.0),
            score,
        )

        evidence[prediction.label].append({
            "router": prediction.router,
            "confidence": round(confidence, 3),
            "multiplier": round(multiplier, 3),
            "effective_score": round(score, 3),
            "reason": prediction.reason,
        })

    # Specialist-backed families take priority when at least one specialist
    # has credible evidence. Generic press/pull predictions control the family
    # only when no specialist reaches the credibility threshold.
    generic_only_score = generic_conf
    specialist_threshold = 0.75

    family_scores = dict(specialist_scores)

    strongest_specialist_score = max(
        specialist_scores.values(),
        default=0.0,
    )

    credible_specialist_exists = (
        strongest_specialist_score >= specialist_threshold
    )

    if (
        generic_family in {"press", "pull"}
        and not credible_specialist_exists
    ):
        family_scores[generic_family] = max(
            family_scores.get(generic_family, 0.0),
            generic_only_score,
        )

        label_scores[generic_label] = max(
            label_scores.get(generic_label, 0.0),
            generic_only_score,
        )

    if not family_scores:
        if not generic_label:
            return _empty_result()

        family_scores[generic_family] = generic_conf
        label_scores[generic_label] = generic_conf

    # Generic evidence is only a tie-breaker for specialist-backed families.
    # It is not added as a second full vote.
    if (
        generic_family in family_scores
        and generic_family not in {"press", "pull"}
    ):
        family_scores[generic_family] += generic_conf * 0.08

    # ----------------------------------------------------------
    # Close-family tie-break
    # ----------------------------------------------------------
    #
    # Specialist scores can be nearly identical for squat receiving positions
    # and Olympic lifts. Resolve only close squat-vs-Olympic contests using
    # independent motion and generic-family evidence.
    winning_family = max(
        family_scores,
        key=lambda family: float(family_scores[family]),
    )

    squat_score = float(family_scores.get("squat", 0.0))
    olympic_score = float(family_scores.get("olympic", 0.0))
    close_margin = 0.10

    if (
        squat_score > 0.0
        and olympic_score > 0.0
        and abs(squat_score - olympic_score) <= close_margin
    ):
        # High-confidence Olympic specialist evidence plus meaningful motion
        # should win a close family contest. Do not require truly_explosive:
        # that derived flag can be false on valid Olympic clips.
        strong_olympic_motion = (
            float(state.olympic_conf or 0.0) >= 0.88
            and float(state.explosive_score or 0.0) >= 20.0
        )

        credible_squat_support = (
            generic_family == "squat"
            and float(generic_conf or 0.0) >= 0.65
            and float(state.squat_conf or 0.0) >= 0.65
        )

        if strong_olympic_motion:
            winning_family = "olympic"
        elif credible_squat_support:
            winning_family = "squat"

    # ----------------------------------------------------------
    # Explosive pull-to-catch family promotion
    # ----------------------------------------------------------
    #
    # A clean can resemble a deadlift or squat at individual frames. Promote
    # the Olympic family only when the generic prediction is uncertain and
    # independent motion evidence shows a genuinely explosive clean-shaped
    # movement with at least credible Olympic specialist support.
    explosive_clean_family = (
        bool(state.truly_explosive)
        and bool(state.looks_clean)
        and float(state.explosive_score or 0.0) >= 60.0
        and float(generic_conf or 0.0) < 0.70
        and float(state.olympic_conf or 0.0) >= 0.50
    )

    if explosive_clean_family:
        winning_family = "olympic"

    candidates = {
        label: score
        for label, score in label_scores.items()
        if LABEL_FAMILY.get(label) == winning_family
    }

    if not candidates:
        if generic_family == winning_family and generic_label:
            candidates[generic_label] = generic_conf
        else:
            return _empty_result()

    winner = max(
        candidates,
        key=lambda label: float(candidates[label]),
    )

    # ----------------------------------------------------------
    # Olympic subtype hierarchy
    # ----------------------------------------------------------
    #
    # Family selection and subtype selection are separate decisions:
    #   1. clear jerk sequence
    #   2. strong snatch specialist
    #   3. explosive clean-shaped fallback
    if winning_family == "olympic" and bool(state.truly_explosive):
        original_winner = winner
        original_score = float(candidates.get(original_winner, 0.0))

        if bool(state.looks_cj):
            winner = "clean_and_jerk"
        elif (
            state.olympic_label == "snatch"
            and float(state.olympic_conf or 0.0) >= 0.80
        ):
            winner = "snatch"
        elif bool(state.looks_clean):
            winner = "clean"

        if winner not in candidates:
            candidates[winner] = max(
                original_score,
                float(state.olympic_conf or 0.0),
            )

    if (
        winner == "squat"
        and winning_family == "squat"
        and state.squat_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and float(state.squat_conf or 0.0) >= 0.55
    ):
        winner = state.squat_label
        candidates[winner] = max(
            candidates.get(winner, 0.0),
            float(state.squat_conf or 0.0),
        )

    winner_score = float(candidates[winner])
    candidate_total = (
        sum(max(0.0, float(v)) for v in candidates.values())
        or 1.0
    )
    confidence = min(1.0, winner_score / candidate_total)

    if generic_label:
        evidence[generic_label].append({
            "router": "generic_merged",
            "confidence": round(generic_conf, 3),
            "effective_score": round(
                generic_conf * 0.08
                if generic_family not in {"press", "pull"}
                else generic_conf,
                3,
            ),
            "reason": generic_meta.get("reason", ""),
        })

    return {
        "label": winner,
        "confidence": confidence,
        "decision": "specialist_family_then_subtype",
        "winning_family": winning_family,
        "family_scores": {
            key: round(value, 3)
            for key, value in sorted(
                family_scores.items(),
                key=lambda item: -item[1],
            )
        },
        "scores": {
            key: round(value, 3)
            for key, value in sorted(
                label_scores.items(),
                key=lambda item: -item[1],
            )
        },
        "evidence": dict(evidence),
        "locks": locks,
    }
