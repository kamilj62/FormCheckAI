from collections import defaultdict

from .locks_v38 import get_locks
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


ROUTER_WEIGHTS = {
    "base": 1.00,
    "biomechanics": 0.95,
    "squat": 0.90,
    "olympic": 0.90,
    "bodyweight": 0.80,
}


def _empty_result():
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


def _normalize_confidence(winner_score, all_scores):
    total = sum(max(0.0, float(v)) for v in all_scores.values()) or 1.0
    return min(1.0, max(0.0, float(winner_score) / total))


def fuse_predictions(
    predictions: list[RouterPrediction],
    state: RouterState | None = None,
) -> dict:
    """
    Hierarchical Router V8 fusion.

    V8 stays shadow-only. It first applies contextual locks, then selects a
    movement family, and finally selects a subtype inside that family.
    """

    if state is None:
        state = RouterState()

    label_scores = defaultdict(float)
    family_scores = defaultdict(float)
    evidence = defaultdict(list)

    # ----------------------------------------------------------
    # Context-supported locks
    # ----------------------------------------------------------
    locks = get_locks(state)

    if locks:
        winner_lock = locks[0]

        return {
            "label": winner_lock["label"],
            "confidence": float(winner_lock["confidence"]),
            "decision": "context_lock",
            "winning_family": LABEL_FAMILY.get(winner_lock["label"]),
            "family_scores": {},
            "scores": {},
            "evidence": {},
            "locks": locks,
            "lock": winner_lock,
        }

    # ----------------------------------------------------------
    # Router evidence
    # ----------------------------------------------------------
    for prediction in predictions:
        if not prediction.label:
            continue

        label = str(prediction.label)
        family = LABEL_FAMILY.get(label)

        if family is None:
            continue

        confidence = float(prediction.confidence or 0.0)
        weight = ROUTER_WEIGHTS.get(prediction.router, 0.75)
        score = confidence * weight

        # Base and biomechanics frequently repeat the same upstream decision.
        # Treat the biomechanics copy as correlated evidence rather than a
        # second independent vote.
        if (
            prediction.router == "biomechanics"
            and state.raw_label
            and label == state.raw_label
        ):
            score *= 0.15

        # Specialist routers contribute strongly only when supporting context
        # exists. Otherwise they remain evidence rather than global authority.
        if prediction.router == "squat":
            if (
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
            ):
                score *= 1.20
            else:
                score *= 0.55

        elif prediction.router == "olympic":
            olympic_context = (
                state.looks_clean
                or state.looks_cj
                or state.looks_split
                or state.truly_explosive
            )
            score *= 1.25 if olympic_context else 0.55

        elif prediction.router == "bodyweight":
            bodyweight_context = (
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
            score *= 1.20 if bodyweight_context else 0.60

        label_scores[label] += score
        family_scores[family] += score

        evidence[label].append({
            "router": prediction.router,
            "confidence": round(confidence, 3),
            "weight": round(weight, 3),
            "effective_score": round(score, 3),
            "reason": prediction.reason,
        })

    if not label_scores or not family_scores:
        return _empty_result()

    # ----------------------------------------------------------
    # Shape-based family evidence
    # ----------------------------------------------------------
    explosive = float(state.explosive_score or 0.0)

    # Shape signals are useful only when motion context supports them.
    # They are intentionally gated because looks_clean / looks_split can fire
    # on ordinary squats and vertical bodyweight movements.

    if (
        state.looks_cj
        and explosive >= 60.0
        and state.truly_explosive
    ):
        family_scores["olympic"] += 1.25
        label_scores["clean_and_jerk"] += 1.25

    split_press_context = (
        state.raw_label == "push_press"
        or state.bio_label == "push_press"
    )

    if (
        state.looks_split
        and not state.looks_cj
        and split_press_context
        and state.olympic_label in {"clean_and_jerk", "split_jerk"}
        and float(state.olympic_conf or 0.0) >= 0.75
        and 25.0 <= explosive <= 100.0
    ):
        family_scores["olympic"] += 1.10
        label_scores["split_jerk"] += 1.10

    if (
        state.looks_clean
        and not state.looks_cj
        and not state.looks_split
        and explosive >= 20.0
        and (
            state.truly_explosive
            or state.olympic_label == "clean"
        )
    ):
        family_scores["olympic"] += 0.90
        label_scores["clean"] += 0.90

    if state.looks_strict:
        family_scores["press"] += 0.90
        label_scores["strict_press"] += 0.90

    if state.looks_thruster:
        family_scores["press"] += 1.00
        label_scores["thruster"] += 1.00

    if state.truly_explosive and explosive >= 20.0:
        family_scores["olympic"] += 0.25

    # Base + biomechanics agreement is stronger family evidence than one
    # specialist router acting by itself.
    if (
        state.raw_label
        and state.raw_label == state.bio_label
        and state.raw_label in LABEL_FAMILY
    ):
        agreed_label = state.raw_label
        agreed_family = LABEL_FAMILY[agreed_label]
        # Raw and biomechanics agreement is useful, but these signals are
        # correlated and have already contributed above. Keep only a small
        # confirmation bonus instead of effectively counting the same opinion
        # for a third time.
        agreement_bonus = min(
            0.15,
            (
                float(state.raw_conf or 0.0)
                + float(state.bio_conf or 0.0)
            ) / 2.0,
        )

        family_scores[agreed_family] += agreement_bonus
        label_scores[agreed_label] += agreement_bonus

    # ----------------------------------------------------------
    # Contextual family corrections
    # ----------------------------------------------------------

    # Strong non-explosive squat evidence should defeat noisy Olympic shape
    # signals, including false looks_clean / looks_split detections.
    if (
        state.raw_label in {"squat", "squat_back", "squat_front"}
        and state.squat_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and float(state.squat_conf or 0.0) >= 0.70
        and not state.truly_explosive
        and explosive < 20.0
    ):
        family_scores["squat"] += 1.25
        label_scores[state.squat_label] += 1.25

    # Overhead squat rescue: push-press classifiers often fire because the
    # wrists stay overhead. Sustained overhead position plus a credible squat
    # subtype and low explosion is stronger evidence for overhead squat.
    if (
        state.squat_label == "overhead_squat"
        and float(state.squat_conf or 0.0) >= 0.75
        and float(state.wrist_overhead or 0.0) >= 0.75
        and explosive < 30.0
        and not state.truly_explosive
    ):
        # Must overcome the false push-press family agreement produced by
        # sustained overhead wrists. The surrounding conditions are deliberately
        # narrow: strong OHS router, very high overhead ratio, and low explosion.
        family_scores["squat"] += 3.75
        label_scores["overhead_squat"] += 3.75

    # Strong Olympic snatch prediction should not lose to a generic split-shape
    # signal when there is no push-press context.
    if (
        state.olympic_label == "snatch"
        and float(state.olympic_conf or 0.0) >= 0.82
        and state.truly_explosive
        and not split_press_context
    ):
        # Keep family and subtype evidence aligned. Previously the subtype
        # became strongest while its family still lost to generic squat evidence.
        # Strong, truly explosive snatch evidence must overcome generic
        # squat-family agreement from the receiving squat phase.
        family_scores["olympic"] += 2.00
        label_scores["snatch"] += 1.20

    # ----------------------------------------------------------
    # Strong specialist family authority
    # ----------------------------------------------------------

    # A very strong Olympic specialist should defeat correlated generic squat
    # evidence from the receiving position of a clean, jerk, or snatch.
    if (
        state.olympic_label in {
            "clean",
            "clean_and_jerk",
            "snatch",
            "split_jerk",
        }
        and float(state.olympic_conf or 0.0) >= 0.90
        and not (
            state.squat_label == "overhead_squat"
            and float(state.squat_conf or 0.0) >= 0.88
            and not state.truly_explosive
            and explosive < 20.0
        )
    ):
        family_scores["olympic"] += 1.35
        label_scores[state.olympic_label] += 1.35

    # Strong squat specialist rescue. A credible squat subtype should defeat
    # generic press/clean shape signals when the Olympic specialist itself is
    # not strong enough to establish Olympic-family authority.
    if (
        state.squat_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and float(state.squat_conf or 0.0) >= 0.85
        and float(state.olympic_conf or 0.0) < 0.75
    ):
        family_scores["squat"] += 1.75
        label_scores[state.squat_label] += 1.75

    # Strong squat subtype evidence should defeat false press, clean, and
    # thruster shape signals when the movement is not meaningfully explosive.
    if (
        state.squat_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and float(state.squat_conf or 0.0) >= 0.82
        and not state.truly_explosive
        and explosive < 20.0
    ):
        family_scores["squat"] += 1.25
        label_scores[state.squat_label] += 1.25

    # ----------------------------------------------------------
    # Narrow explosive clean-family rescue
    # ----------------------------------------------------------
    #
    # A clean can be mistaken for deadlift/squat from isolated frames.
    # Promote Olympic only for an uncertain generic prediction with:
    #   - genuinely explosive clean-shaped motion
    #   - credible but non-authoritative Olympic evidence
    #   - a front-rack receiving window
    #
    # The wrist window excludes floor-level deadlifts and higher squat/press
    # receiving positions.
    narrow_clean_rescue = (
        bool(state.truly_explosive)
        and bool(state.looks_clean)
        and not bool(state.looks_cj)
        and explosive >= 60.0
        and float(state.raw_conf or 0.0) < 0.70
        and 0.50 <= float(state.olympic_conf or 0.0) < 0.80
        and 0.10 <= float(state.wrist_overhead or 0.0) <= 0.40
    )

    if narrow_clean_rescue:
        family_scores["olympic"] += 2.00
        label_scores["clean"] += 2.00
        evidence["clean"].append({
            "router": "clean_shape_rescue",
            "confidence": round(float(state.olympic_conf or 0.0), 3),
            "effective_score": 2.0,
        })

    # ----------------------------------------------------------
    # Family first, subtype second
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # Context-supported snatch family authority
    # ----------------------------------------------------------
    # Strong C&J specialist evidence should outrank accumulated squat
    # evidence when the movement has a non-clean, thruster-like shape.
    # Unanimous high-confidence deadlift evidence should outrank
    # conflicting squat-family accumulation after protection is released.
    unanimous_deadlift_authority = (
        state.raw_label == "deadlift"
        and state.bio_label == "deadlift"
        and float(state.raw_conf or 0.0) >= 0.90
        and float(state.bio_conf or 0.0) >= 0.90
    )

    if unanimous_deadlift_authority:
        strongest_competing_family = max(
            float(family_scores.get("squat", 0.0)),
            float(family_scores.get("press", 0.0)),
            float(family_scores.get("olympic", 0.0)),
            float(family_scores.get("bodyweight", 0.0)),
        )

        family_scores["pull"] = max(
            float(family_scores.get("pull", 0.0)),
            strongest_competing_family + 0.01,
        )

        label_scores["deadlift"] = max(
            float(label_scores.get("deadlift", 0.0)),
            float(state.raw_conf or 0.0),
            float(state.bio_conf or 0.0),
        )


    # V29: once the trusted bench lock is vetoed, strong C&J specialist
    # evidence must also receive enough family authority to beat press.
    # V30: strong raw deadlift evidence should beat false squat-family
    # accumulation unless the Olympic specialist has strong snatch evidence.
    strong_snatch_evidence = (
        state.olympic_label == "snatch"
        and float(state.olympic_conf or 0.0) >= 0.75
    )

    strong_raw_deadlift_authority = (
        state.raw_label == "deadlift"
        and float(state.raw_conf or 0.0) >= 0.90
        and state.bio_label in {"deadlift", "squat"}
        and float(state.bio_conf or 0.0) >= 0.90
        and state.squat_label == "squat_back"
        and state.protected_label is None
        and not bool(state.looks_split)
        and not bool(state.looks_thruster)
        and not strong_snatch_evidence
    )

    if strong_raw_deadlift_authority:
        strongest_competing_family = max(
            float(family_scores.get("squat", 0.0)),
            float(family_scores.get("press", 0.0)),
            float(family_scores.get("olympic", 0.0)),
            float(family_scores.get("bodyweight", 0.0)),
        )

        family_scores["pull"] = max(
            float(family_scores.get("pull", 0.0)),
            strongest_competing_family + 0.01,
        )

        label_scores["deadlift"] = max(
            float(label_scores.get("deadlift", 0.0)),
            float(state.raw_conf or 0.0),
            float(state.bio_conf or 0.0),
        )

    # V31: very strong C&J evidence should beat accumulated squat-family
    # scoring when the squat subtype is overhead squat.
    strong_cj_overhead_squat_authority = (
        state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.96
        and state.squat_label == "overhead_squat"
    )

    if strong_cj_overhead_squat_authority:
        strongest_competing_family = max(
            float(family_scores.get("squat", 0.0)),
            float(family_scores.get("press", 0.0)),
            float(family_scores.get("pull", 0.0)),
            float(family_scores.get("bodyweight", 0.0)),
        )

        family_scores["olympic"] = max(
            float(family_scores.get("olympic", 0.0)),
            strongest_competing_family + 0.01,
        )

        label_scores["clean_and_jerk"] = max(
            float(label_scores.get("clean_and_jerk", 0.0)),
            float(state.olympic_conf or 0.0),
        )

    strong_cj_trusted_bench_authority = (
        state.protected_label == "bench_press"
        and state.protected_reason == "trusted_base_bench_press"
        and state.raw_label == "bench_press"
        and state.bio_label == "bench_press"
        and float(state.raw_conf or 0.0) >= 0.90
        and float(state.bio_conf or 0.0) >= 0.90
        and state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.87
    )

    if strong_cj_trusted_bench_authority:
        strongest_competing_family = max(
            float(family_scores.get("squat", 0.0)),
            float(family_scores.get("press", 0.0)),
            float(family_scores.get("pull", 0.0)),
            float(family_scores.get("bodyweight", 0.0)),
        )

        family_scores["olympic"] = max(
            float(family_scores.get("olympic", 0.0)),
            strongest_competing_family + 0.01,
        )

        label_scores["clean_and_jerk"] = max(
            float(label_scores.get("clean_and_jerk", 0.0)),
            float(state.olympic_conf or 0.0),
        )

    cj_family_authority = (
        state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.80
        and not bool(state.looks_clean)
        and bool(state.looks_thruster)
    )

    if cj_family_authority:
        family_scores["olympic"] = max(
            float(family_scores.get("olympic", 0.0)),
            float(family_scores.get("squat", 0.0)) + 0.01,
        )
        label_scores["clean_and_jerk"] = max(
            float(label_scores.get("clean_and_jerk", 0.0)),
            float(state.olympic_conf or 0.0),
        )

    snatch_family_authority = (
        state.olympic_label == "snatch"
        and not bool(state.looks_clean)
        and (
            (
                float(state.olympic_conf or 0.0) >= 0.80
                and bool(state.looks_split)
            )
            or (
                float(state.olympic_conf or 0.0) >= 0.70
                and bool(state.truly_explosive)
            )
            or (
                float(state.olympic_conf or 0.0) >= 0.55
                and bool(state.looks_split)
                and bool(state.looks_thruster)
                and float(state.wrist_overhead or 0.0) >= 0.40
            )
        )
    )

    if snatch_family_authority:
        family_scores["olympic"] = max(
            float(family_scores.get("olympic", 0.0)),
            float(family_scores.get("squat", 0.0)) + 0.01,
        )
        label_scores["snatch"] = max(
            float(label_scores.get("snatch", 0.0)),
            float(state.olympic_conf or 0.0),
        )

    winning_family = max(
        family_scores.items(),
        key=lambda item: float(item[1]),
    )[0]

    family_labels = {
        label: score
        for label, score in label_scores.items()
        if LABEL_FAMILY.get(label) == winning_family
    }

    if not family_labels:
        return _empty_result()

    winner = max(
        family_labels.items(),
        key=lambda item: float(item[1]),
    )[0]

    # Strong Olympic specialist predictions retain their subtype.
    # Shape flags such as looks_cj are not authoritative enough to replace
    # a high-confidence snatch, clean-and-jerk, split-jerk, or clean result.
    if (
        winning_family == "olympic"
        and state.olympic_label in {
            "clean",
            "clean_and_jerk",
            "snatch",
            "split_jerk",
        }
        and float(state.olympic_conf or 0.0) >= 0.80
    ):
        winner = state.olympic_label
        family_labels[winner] = max(
            float(family_labels.get(winner, 0.0)),
            float(label_scores.get(winner, 0.0)),
            float(state.olympic_conf or 0.0),
        )

    # Generic "squat" is a family-level prediction, not a final subtype.
    # Once the squat family wins, use the specialist subtype when credible.
    if (
        winning_family == "squat"
        and winner == "squat"
        and state.squat_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and float(state.squat_conf or 0.0) >= 0.65
    ):
        winner = state.squat_label

    # ----------------------------------------------------------
    # Front-vs-back squat subtype reconciliation
    # ----------------------------------------------------------
    #
    # The squat specialist can over-predict front squat when generic models
    # only identify the broader squat family. Require either direct generic
    # front-squat support or a front-rack-like wrist position before retaining
    # the front-squat subtype.
    unsupported_front_squat = (
        winning_family == "squat"
        and winner == "squat_front"
        and state.squat_label == "squat_front"
        and float(state.squat_conf or 0.0) >= 0.85
        and state.raw_label != "squat_front"
        and state.bio_label != "squat_front"
        and float(state.wrist_overhead or 0.0) < 0.35
        and float(state.olympic_conf or 0.0) < 0.85
    )

    if unsupported_front_squat:
        winner = "squat_back"
        family_labels[winner] = max(
            float(family_labels.get(winner, 0.0)),
            float(state.squat_conf or 0.0) * 0.95,
        )

    # Ensure any subtype override has a score entry before confidence
    # normalization.
    if winner not in family_labels:
        family_labels[winner] = max(
            float(label_scores.get(winner, 0.0)),
            float(state.olympic_conf or 0.0)
            if winning_family == "olympic"
            else 0.0,
        )

    winner_score = float(family_labels[winner])
    confidence = _normalize_confidence(winner_score, family_labels)

    return {
        "label": winner,
        "confidence": confidence,
        "decision": "family_then_subtype",
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
