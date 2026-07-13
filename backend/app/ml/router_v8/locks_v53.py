from .state import RouterState


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
    "push_up",
    "pull_up",
    "handstand_push_up",
    "muscle_up",
    "burpee",
}

SUPPORTED_LABELS = (
    SQUAT_LABELS
    | OLYMPIC_LABELS
    | BODYWEIGHT_LABELS
    | {
        "bench_press",
        "deadlift",
        "push_press",
        "strict_press",
        "thruster",
    }
)


def _add_lock(
    locks: list[dict],
    *,
    label: str,
    confidence: float,
    reason: str,
    priority: int,
) -> None:
    if not label:
        return

    locks.append({
        "label": str(label),
        "confidence": round(float(confidence or 0.0), 3),
        "reason": str(reason),
        "priority": int(priority),
    })


def get_locks(state: RouterState) -> list[dict]:
    """
    Return context-supported Router V8 locks.

    Specialist confidence alone is not enough to create a global lock.
    Existing deterministic protection evidence is allowed because it represents
    a specific movement-pattern detector rather than the final production label.
    """

    locks: list[dict] = []

    raw_conf = float(state.raw_conf or 0.0)
    bio_conf = float(state.bio_conf or 0.0)
    squat_conf = float(state.squat_conf or 0.0)
    olympic_conf = float(state.olympic_conf or 0.0)
    bodyweight_conf = float(state.bodyweight_conf or 0.0)
    explosive = float(state.explosive_score or 0.0)
    wrist_overhead = float(state.wrist_overhead or 0.0)

    olympic_shape_present = bool(
        state.looks_clean
        or state.looks_cj
        or state.looks_split
    )

    # ==========================================================
    # 1. Deterministic protection evidence
    # ==========================================================
    # A protected bodyweight label is normally strong context evidence.
    # However, pull-up protection can be a false positive on Olympic lifts
    # with sustained overhead motion. In that narrow conflict, allow normal
    # fusion to decide instead of creating an unbeatable protection lock.
    strong_olympic_pull_up_conflict = (
        state.protected_label == "pull_up"
        and state.olympic_label in OLYMPIC_LABELS
        and olympic_conf >= 0.90
        and wrist_overhead >= 0.40
    )

    # Bench protection can be a false positive on explosive Olympic clips.
    # Release only the narrow case where the Olympic specialist explicitly
    # identifies snatch and the movement is truly explosive.
    strong_cj_pull_up_conflict = (
        state.protected_label == "pull_up"
        and state.olympic_label == "clean_and_jerk"
        and olympic_conf >= 0.60
        and not bool(state.looks_clean)
        and bool(state.looks_thruster)
        and float(state.wrist_overhead or 0.0) >= 0.50

        # Preserve unanimous push-press evidence.
        and not (
            state.raw_label == "push_press"
            and state.bio_label == "push_press"
        )

        # Avoid turning deadlift-led clips into authoritative C&J.
        and state.raw_label != "deadlift"

        # Preserve strong overhead-squat evidence. This also protects
        # true pull-ups that the squat router reads as overhead squat.
        and not (
            state.squat_label == "overhead_squat"
            and float(state.squat_conf or 0.0) >= 0.85
        )
    )

    # Release false bench protection when the base classifier and
    # high-confidence squat router both support the squat family.
    # Suppress an overconfident front-squat context lock when
    # generic squat evidence conflicts with non-clean Olympic context.
    strong_back_squat_front_lock_conflict = (
        state.raw_label == "squat"
        and state.squat_label == "squat_front"
        and float(state.squat_conf or 0.0) >= 0.97
        and not bool(state.looks_clean)
        and float(state.olympic_conf or 0.0) >= 0.68
    )

    # Bench rescue must not override unanimous high-confidence
    # deadlift evidence from both the base and biomechanics classifiers.
    # A bodyweight pull-up winner must not override unanimous,
    # high-confidence press evidence unless bench has stronger Olympic conflict.
    # Strong shape-supported snatch evidence must outrank
    # conflicting bodyweight protection.
    # Explicit high-confidence snatch evidence should outrank a
    # conflicting clean-and-jerk final-recovery shape lock when the
    # clean signature itself is absent.
    # Strong clean-and-jerk specialist evidence can override a
    # trusted-base bench lock when both evidence systems strongly disagree.
    # V31: very strong C&J specialist evidence should not be blocked
    # by a conflicting protection state when the squat router sees OHS.
    strong_cj_overhead_squat_conflict = (
        state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.96
        and state.squat_label == "overhead_squat"
    )

    if strong_cj_overhead_squat_conflict:
        _add_lock(
            locks,
            label="clean_and_jerk",
            confidence=float(state.olympic_conf or 0.0),
            reason="strong_cj_overhead_squat_authority",
            priority=122,
        )

    strong_cj_trusted_bench_conflict = (
        state.protected_label == "bench_press"
        and state.protected_reason == "trusted_base_bench_press"
        and state.raw_label == "bench_press"
        and state.bio_label == "bench_press"
        and float(state.raw_conf or 0.0) >= 0.90
        and float(state.bio_conf or 0.0) >= 0.90
        and state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.87
    )

    strong_snatch_cj_recovery_conflict = (
        state.protected_label == "clean_and_jerk"
        and state.protected_reason
            == "clean_and_jerk_shape_final_recovery"
        and state.olympic_label == "snatch"
        and float(state.olympic_conf or 0.0) >= 0.80
        and not bool(state.looks_clean)
        and bool(state.looks_split)
        and bool(state.looks_thruster)
        and bool(state.truly_explosive)
    )

    strong_snatch_bodyweight_conflict = (
        state.protected_label in {
            "pull_up",
            "push_up",
            "handstand_push_up",
            "muscle_up",
            "burpee",
        }
        and state.olympic_label == "snatch"
        and float(state.olympic_conf or 0.0) >= 0.70
        and bool(state.looks_split)
        and bool(state.looks_thruster)
    )

    unanimous_press_pull_up_conflict = (
        state.protected_label == "pull_up"
        and state.protected_reason == "router_v6_bodyweight_winner"
        and state.raw_label == state.bio_label
        and float(state.raw_conf or 0.0) >= 0.95
        and float(state.bio_conf or 0.0) >= 0.95
        and (
            (
                state.raw_label == "bench_press"
                and float(state.olympic_conf or 0.0) < 0.65
            )
            or state.raw_label in {
                "push_press",
                "strict_press",
            }
        )
    )

    strong_deadlift_bench_conflict = (
        state.protected_label == "bench_press"
        and state.raw_label == "deadlift"
        and state.bio_label == "deadlift"
        and float(state.raw_conf or 0.0) >= 0.90
        and float(state.bio_conf or 0.0) >= 0.90
    )

    strong_squat_bench_conflict = (
        state.protected_label == "bench_press"
        and state.raw_label in {
            "squat",
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and state.squat_label == "squat_front"
        and float(state.squat_conf or 0.0) >= 0.83
        and float(state.wrist_overhead or 0.0) < 0.80
    )

    strong_snatch_bench_conflict = (
        state.protected_label == "bench_press"
        and state.olympic_label == "snatch"
        and bool(state.truly_explosive)
        and olympic_conf >= 0.50
        and not (
            state.raw_label == "deadlift"
            and state.bio_label == "deadlift"
        )
    )

    if strong_snatch_bench_conflict:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(0.82, olympic_conf),
            reason="explosive_snatch_over_bench_protection",
            priority=121,
        )

    # V32: resolve false front-squat subtype predictions when broader
    # evidence describes a non-explosive ordinary squat without clean/split
    # or overhead context.
    back_squat_front_ambiguity = (
        state.squat_label == "squat_front"
        and float(state.squat_conf or 0.0) >= 0.83
        and state.raw_label in {
            "squat",
            "squat_back",
            "squat_front",
        }
        and float(state.raw_conf or 0.0) >= 0.65
        and not bool(state.looks_clean)
        and not bool(state.looks_split)
        and not bool(state.truly_explosive)
        and float(state.wrist_overhead or 0.0) < 0.65
    )

    if back_squat_front_ambiguity:
        _add_lock(
            locks,
            label="squat_back",
            confidence=max(
                0.84,
                float(state.raw_conf or 0.0),
            ),
            reason="nonexplosive_back_squat_over_front_ambiguity",
            priority=123,
        )

    # V33: an extreme explosive push press should not be converted into
    # thruster when the Olympic router lacks strong C&J evidence and there
    # is no split-jerk or C&J shape.
    strong_push_press_thruster_conflict = (
        state.protected_label == "thruster"
        and state.protected_reason == "thruster_pattern_detected"
        and state.raw_label == "push_press"
        and float(state.raw_conf or 0.0) >= 0.98
        and float(state.olympic_conf or 0.0) < 0.65
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
        and float(state.explosive_score or 0.0) >= 150.0
    )

    if strong_push_press_thruster_conflict:
        _add_lock(
            locks,
            label="push_press",
            confidence=max(
                0.90,
                float(state.raw_conf or 0.0),
                float(state.bio_conf or 0.0)
                if state.bio_label == "push_press"
                else 0.0,
            ),
            reason="strong_push_press_over_false_thruster",
            priority=124,
        )

    # V34: distinguish C&J from false thruster scoring when the routers
    # show a split/thruster-shaped Olympic movement with moderate C&J
    # confidence and a clearly overhead catch/recovery pattern.
    split_thruster_shape_cj_conflict = (
        state.raw_label == "squat"
        and state.bio_label == "push_press"
        and state.olympic_label == "clean_and_jerk"
        and 0.60 <= float(state.olympic_conf or 0.0) < 0.90
        and state.squat_label in {
            "squat_front",
            "overhead_squat",
        }
        and bool(state.looks_split)
        and bool(state.looks_thruster)
        and not bool(state.looks_cj)
        and 0.60 <= float(state.wrist_overhead or 0.0) < 0.90
        and float(state.explosive_score or 0.0) >= 35.0
    )

    if split_thruster_shape_cj_conflict:
        _add_lock(
            locks,
            label="clean_and_jerk",
            confidence=max(
                0.82,
                float(state.olympic_conf or 0.0),
            ),
            reason="split_thruster_shape_cj_authority",
            priority=125,
        )

    # V35: release false back-squat scoring when the Olympic router
    # explicitly supports snatch and the motion carries a pull/thruster
    # conflict not present in ordinary back squats.
    moderate_snatch_back_squat_conflict = (
        state.olympic_label == "snatch"
        and 0.80 <= float(state.olympic_conf or 0.0) < 0.90
        and state.squat_label == "squat_back"
        and bool(state.looks_clean)
        and not bool(state.looks_split)
        and not bool(state.looks_cj)
        and (
            state.raw_label == "deadlift"
            or bool(state.looks_thruster)
        )
    )

    if moderate_snatch_back_squat_conflict:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(
                0.84,
                float(state.olympic_conf or 0.0),
            ),
            reason="moderate_snatch_over_false_back_squat",
            priority=126,
        )

    # V36: release weak trusted-bench locks when the surrounding router
    # evidence is inconsistent with a real bench press and favors an
    # Olympic movement pattern.
    weak_trusted_bench_snatch_conflict = (
        state.protected_label == "bench_press"
        and state.protected_reason == "trusted_base_bench_press"
        and state.raw_label == "bench_press"
        and float(state.raw_conf or 0.0) < 0.90
        and state.bio_label == "bench_press"
        and state.squat_label == "squat_front"
        and state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.85
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
        and not bool(state.looks_thruster)
    )

    if weak_trusted_bench_snatch_conflict:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(
                0.86,
                float(state.olympic_conf or 0.0),
            ),
            reason="snatch_over_weak_trusted_bench_lock",
            priority=127,
        )

    # V37: release a false handstand-push-up lock when both base and
    # biomechanics models support bench press and there is no overhead,
    # split, clean, C&J, or thruster geometry.
    false_handstand_bench_conflict = (
        state.protected_label == "handstand_push_up"
        and state.protected_reason
            == "handstand_push_up_bodyweight_pattern"
        and state.raw_label == "bench_press"
        and state.bio_label == "bench_press"
        and float(state.bio_conf or 0.0) >= 0.75
        and state.squat_label == "squat_front"
        and state.olympic_label == "snatch"
        and float(state.olympic_conf or 0.0) < 0.60
        and float(state.wrist_overhead or 0.0) < 0.10
        and not bool(state.looks_clean)
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
        and not bool(state.looks_thruster)
    )

    if false_handstand_bench_conflict:
        _add_lock(
            locks,
            label="bench_press",
            confidence=max(
                0.82,
                float(state.bio_conf or 0.0),
                float(state.raw_conf or 0.0),
            ),
            reason="bench_press_over_false_handstand_lock",
            priority=128,
        )

    # V38: release a false short-squat bench rescue when biomechanics
    # support push press and the Olympic router supports C&J with a clearly
    # overhead, moderately explosive movement.
    false_short_squat_bench_cj_conflict = (
        state.protected_label == "bench_press"
        and state.protected_reason
            == "bench_press_short_squat_rescue"
        and state.raw_label == "squat"
        and state.bio_label == "push_press"
        and float(state.bio_conf or 0.0) >= 0.75
        and state.squat_label == "squat_front"
        and state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.70
        and float(state.wrist_overhead or 0.0) >= 0.80
        and float(state.explosive_score or 0.0) >= 30.0
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
        and not bool(state.looks_thruster)
    )

    if false_short_squat_bench_cj_conflict:
        _add_lock(
            locks,
            label="clean_and_jerk",
            confidence=max(
                0.80,
                float(state.olympic_conf or 0.0),
            ),
            reason="cj_over_false_short_squat_bench_rescue",
            priority=129,
        )

    # V39: release a false bench-pattern lock when the Olympic router
    # strongly supports C&J and the base label is squat rather than bench.
    false_bench_pattern_cj_conflict = (
        state.protected_label == "bench_press"
        and state.protected_reason == "bench_pattern_detected"
        and state.raw_label == "squat"
        and state.bio_label == "bench_press"
        and float(state.bio_conf or 0.0) >= 0.75
        and state.squat_label == "squat_back"
        and state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.88
        and float(state.explosive_score or 0.0) < 30.0
        and float(state.wrist_overhead or 0.0) < 0.10
        and not bool(state.looks_clean)
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
        and not bool(state.looks_thruster)
    )

    if false_bench_pattern_cj_conflict:
        _add_lock(
            locks,
            label="clean_and_jerk",
            confidence=max(
                0.89,
                float(state.olympic_conf or 0.0),
            ),
            reason="cj_over_false_bench_pattern_lock",
            priority=130,
        )

    # V40: override false bench scoring under a pull-up protection when
    # the squat subtype, Olympic prediction, overhead position, and
    # explosive thruster-like geometry consistently support C&J.
    false_bodyweight_bench_cj_conflict = (
        state.protected_label == "pull_up"
        and state.protected_reason
            == "router_v6_bodyweight_winner"
        and state.raw_label == "bench_press"
        and state.bio_label == "bench_press"
        and float(state.raw_conf or 0.0) >= 0.95
        and float(state.bio_conf or 0.0) >= 0.95
        and state.squat_label == "squat_front"
        and float(state.squat_conf or 0.0) >= 0.90
        and state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.65
        and bool(state.looks_thruster)
        and bool(state.truly_explosive)
        and not bool(state.looks_split)
        and not bool(state.looks_cj)
        and float(state.wrist_overhead or 0.0) >= 0.65
    )

    if false_bodyweight_bench_cj_conflict:
        _add_lock(
            locks,
            label="clean_and_jerk",
            confidence=max(
                0.82,
                float(state.olympic_conf or 0.0),
            ),
            reason="cj_over_false_bodyweight_bench_scoring",
            priority=131,
        )

    # V41: release a false bench-pattern lock when the base and squat
    # routers agree on front squat and there is no Olympic or explosive
    # movement evidence.
    false_bench_pattern_front_squat_conflict = (
        state.protected_label == "bench_press"
        and state.protected_reason == "bench_pattern_detected"
        and state.raw_label == "squat_front"
        and float(state.raw_conf or 0.0) >= 0.95
        and state.squat_label == "squat_front"
        and float(state.squat_conf or 0.0) >= 0.70
        and state.olympic_label is None
        and float(state.olympic_conf or 0.0) == 0.0
        and float(state.explosive_score or 0.0) < 10.0
        and float(state.wrist_overhead or 0.0) < 0.10
        and not bool(state.looks_clean)
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
        and not bool(state.looks_thruster)
    )

    if false_bench_pattern_front_squat_conflict:
        _add_lock(
            locks,
            label="squat_front",
            confidence=max(
                0.90,
                float(state.raw_conf or 0.0),
                float(state.squat_conf or 0.0),
            ),
            reason="front_squat_over_false_bench_pattern_lock",
            priority=132,
        )

    # V42: release a false push-up bodyweight lock when the movement is
    # strongly explosive, has squat-back/Olympic evidence, and neither
    # base nor biomechanics routing supports push-up.
    false_explosive_push_up_snatch_conflict = (
        state.protected_label == "push_up"
        and state.protected_reason == "push_up_bodyweight_pattern"
        and state.squat_label == "squat_back"
        and state.olympic_label in {
            "snatch",
            "clean_and_jerk",
        }
        and float(state.olympic_conf or 0.0) >= 0.50
        and bool(state.truly_explosive)
        and float(state.explosive_score or 0.0) >= 100.0
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
        and state.raw_label != "push_up"
        and state.bio_label != "push_up"
    )

    if false_explosive_push_up_snatch_conflict:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(
                0.82,
                float(state.olympic_conf or 0.0),
            ),
            reason="snatch_over_false_explosive_push_up_lock",
            priority=133,
        )

    # V43: release specialized false bench rescues when a squat-labeled,
    # explosive Olympic movement has thruster geometry and no C&J shape.
    false_specialized_bench_snatch_conflict = (
        state.protected_label == "bench_press"
        and state.protected_reason in {
            "bench_press_short_overhead_rescue",
            "bench_press_fast_press_rescue",
        }
        and state.raw_label == "squat"
        and state.squat_label in {
            "overhead_squat",
            "squat_back",
        }
        and state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.55
        and bool(state.looks_thruster)
        and bool(state.truly_explosive)
        and float(state.explosive_score or 0.0) >= 90.0
        and float(state.wrist_overhead or 0.0) >= 0.25
        and not bool(state.looks_cj)
    )

    if false_specialized_bench_snatch_conflict:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(
                0.82,
                float(state.olympic_conf or 0.0),
            ),
            reason="snatch_over_false_specialized_bench_rescue",
            priority=134,
        )

    # V44: release false strong-bench agreement when the surrounding
    # evidence is inconsistent with a normal bench press and supports an
    # Olympic snatch pattern through either strong Olympic confidence or
    # extreme explosiveness.
    false_strong_bench_snatch_conflict = (
        state.protected_label == "bench_press"
        and state.protected_reason
            == "strong_bench_model_agreement"
        and state.raw_label == "bench_press"
        and state.bio_label == "bench_press"
        and float(state.raw_conf or 0.0) >= 0.998
        and float(state.bio_conf or 0.0) >= 0.998
        and state.squat_label == "squat_front"
        and state.olympic_label == "clean_and_jerk"
        and (
            float(state.olympic_conf or 0.0) >= 0.85
            or float(state.explosive_score or 0.0) >= 155.0
        )
        and float(state.wrist_overhead or 0.0) < 0.05
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
        and not bool(state.looks_thruster)
    )

    if false_strong_bench_snatch_conflict:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(
                0.86,
                float(state.olympic_conf or 0.0),
            ),
            reason="snatch_over_false_strong_bench_agreement",
            priority=135,
        )

    # V45: preserve an explicit overhead-squat subtype when press-family
    # scoring incorrectly promotes an explosive-looking overhead squat
    # into thruster. Sustained overhead position and the squat router
    # provide the stronger movement-specific evidence.
    false_thruster_overhead_squat_conflict = (
        state.raw_label == "squat"
        and state.bio_label == "push_press"
        and state.squat_label == "overhead_squat"
        and float(state.squat_conf or 0.0) >= 0.80
        and float(state.wrist_overhead or 0.0) >= 0.95
        and bool(state.looks_thruster)
        and not bool(state.looks_clean)
        and float(state.explosive_score or 0.0) >= 60.0
    )

    if false_thruster_overhead_squat_conflict:
        _add_lock(
            locks,
            label="overhead_squat",
            confidence=max(
                0.88,
                float(state.squat_conf or 0.0),
            ),
            reason="overhead_squat_over_false_thruster_family",
            priority=136,
        )

    # V46: preserve an explicit snatch router result when moderate
    # snatch confidence is incorrectly overridden by generic C&J,
    # split, and thruster shape flags.
    false_cj_shape_snatch_conflict = (
        state.raw_label == "squat"
        and state.squat_label in {
            "squat_back",
            "overhead_squat",
        }
        and state.olympic_label == "snatch"
        and 0.64 <= float(state.olympic_conf or 0.0) < 0.80
        and float(state.explosive_score or 0.0) >= 80.0
        and bool(state.truly_explosive)
        and not bool(state.looks_clean)
        and bool(state.looks_cj)
        and bool(state.looks_split)
        and bool(state.looks_thruster)
    )

    if false_cj_shape_snatch_conflict:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(
                0.84,
                float(state.olympic_conf or 0.0),
            ),
            reason="snatch_over_false_cj_shape_override",
            priority=137,
        )

    # V47: release a false back-squat family result when explosive
    # Olympic evidence appears in either a low-overhead pull pattern
    # or an overhead split/thruster pattern.
    low_overhead_snatch_pull = (
        bool(state.looks_clean)
        and not bool(state.looks_split)
        and not bool(state.looks_thruster)
        and float(state.wrist_overhead or 0.0) < 0.05
    )

    overhead_snatch_split = (
        not bool(state.looks_clean)
        and bool(state.looks_split)
        and bool(state.looks_thruster)
        and float(state.wrist_overhead or 0.0) >= 0.65
    )

    false_back_squat_snatch_conflict = (
        state.raw_label == "squat"
        and state.squat_label == "squat_back"
        and state.olympic_label == "clean_and_jerk"
        and 0.50 <= float(state.olympic_conf or 0.0) < 0.75
        and float(state.explosive_score or 0.0) >= 110.0
        and bool(state.truly_explosive)
        and not bool(state.looks_cj)
        and (
            low_overhead_snatch_pull
            or overhead_snatch_split
        )
    )

    if false_back_squat_snatch_conflict:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(
                0.84,
                float(state.olympic_conf or 0.0),
            ),
            reason="snatch_over_false_back_squat_family",
            priority=138,
        )

    # V48: release false front-squat subtype results in two low-explosive
    # ambiguity patterns: clean-rack-like upper-body evidence and a
    # push-press base disagreement with no overhead or Olympic shape.
    clean_rack_back_squat_ambiguity = (
        state.raw_label == "squat"
        and state.bio_label == "push_press"
        and state.olympic_label == "clean_and_jerk"
        and 0.75 <= float(state.olympic_conf or 0.0) < 0.82
        and bool(state.looks_clean)
        and float(state.wrist_overhead or 0.0) >= 0.55
    )

    press_base_back_squat_ambiguity = (
        state.raw_label == "push_press"
        and state.bio_label == "squat"
        and state.olympic_label == "snatch"
        and 0.55 <= float(state.olympic_conf or 0.0) < 0.65
        and not bool(state.looks_clean)
        and float(state.wrist_overhead or 0.0) < 0.05
    )

    false_front_squat_back_squat_conflict = (
        state.squat_label == "squat_front"
        and float(state.squat_conf or 0.0) >= 0.95
        and 30.0 <= float(state.explosive_score or 0.0) < 45.0
        and not bool(state.truly_explosive)
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
        and not bool(state.looks_thruster)
        and (
            clean_rack_back_squat_ambiguity
            or press_base_back_squat_ambiguity
        )
    )

    if false_front_squat_back_squat_conflict:
        _add_lock(
            locks,
            label="squat_back",
            confidence=max(
                0.86,
                float(state.squat_conf or 0.0),
            ),
            reason="squat_back_over_false_front_squat_subtype",
            priority=139,
        )

    # V49: release a false deadlift hinge lock when the base classifier
    # strongly supports squat and the motion is slow, non-explosive,
    # clean-like, and supported by a squat subtype.
    false_deadlift_hinge_squat_conflict = (
        state.protected_label == "deadlift"
        and state.protected_reason
            == "deadlift_hinge_pattern_detected"
        and state.raw_label == "squat"
        and float(state.raw_conf or 0.0) >= 0.99
        and state.bio_label == "deadlift"
        and float(state.bio_conf or 0.0) >= 0.99
        and state.squat_label in {
            "squat_back",
            "squat_front",
        }
        and float(state.squat_conf or 0.0) >= 0.70
        and state.olympic_label == "clean_and_jerk"
        and 0.75 <= float(state.olympic_conf or 0.0) < 0.80
        and bool(state.looks_clean)
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
        and not bool(state.truly_explosive)
        and float(state.explosive_score or 0.0) < 20.0
        and float(state.wrist_overhead or 0.0) < 0.20
    )

    if false_deadlift_hinge_squat_conflict:
        _add_lock(
            locks,
            label="squat_back",
            confidence=max(
                0.88,
                float(state.raw_conf or 0.0),
            ),
            reason="squat_back_over_false_deadlift_hinge_lock",
            priority=140,
        )

    # V50A: explicit snatch authority over a false deadlift rescue.
    snatch_over_deadlift_rescue = (
        state.olympic_label == "snatch"
        and float(state.olympic_conf or 0.0) >= 0.85
        and state.raw_label == "deadlift"
        and state.bio_label == "squat"
        and state.squat_label == "squat_back"
        and float(state.squat_conf or 0.0) >= 0.95
        and not bool(state.looks_cj)
    )

    if snatch_over_deadlift_rescue:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(
                0.90,
                float(state.olympic_conf or 0.0),
            ),
            reason="snatch_over_deadlift_setup_rescue_v50",
            priority=141,
        )

    # V50B: explicit snatch authority over a false thruster result.
    snatch_over_thruster_conflict = (
        state.olympic_label == "snatch"
        and float(state.olympic_conf or 0.0) >= 0.70
        and state.raw_label == "squat"
        and state.squat_label == "overhead_squat"
        and float(state.squat_conf or 0.0) >= 0.80
        and float(state.explosive_score or 0.0) < 60.0
        and bool(state.looks_split)
        and bool(state.looks_thruster)
        and not bool(state.looks_cj)
    )

    if snatch_over_thruster_conflict:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(
                0.86,
                float(state.olympic_conf or 0.0),
            ),
            reason="snatch_over_false_thruster_v50",
            priority=142,
        )

    # V50C: explicit snatch authority over a false burpee/bodyweight lock.
    snatch_over_burpee_conflict = (
        state.olympic_label == "snatch"
        and float(state.olympic_conf or 0.0) >= 0.60
        and state.raw_label == "squat"
        and state.squat_label == "overhead_squat"
        and float(state.explosive_score or 0.0) >= 100.0
        and bool(state.looks_split)
        and bool(state.looks_clean)
        and not bool(state.looks_cj)
    )

    if snatch_over_burpee_conflict:
        _add_lock(
            locks,
            label="snatch",
            confidence=max(
                0.84,
                float(state.olympic_conf or 0.0),
            ),
            reason="snatch_over_false_burpee_v50",
            priority=143,
        )

    # V51: preserve a strong explicit front-squat prediction when
    # lower-confidence subtype or Olympic-family evidence tries to steal it.
    strong_raw_front_squat = (
        state.raw_label == "squat_front"
        and float(state.raw_conf or 0.0) >= 0.93
        and state.bio_label in {"squat", "push_press"}
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
    )

    if strong_raw_front_squat:
        _add_lock(
            locks,
            label="squat_front",
            confidence=max(
                0.94,
                float(state.raw_conf or 0.0),
            ),
            reason="strong_raw_front_squat_authority_v51",
            priority=144,
        )

    # V52A: a very strong clean-and-jerk router prediction outranks
    # a lower-confidence push-press protection.
    strong_cj_over_push_press = (
        state.olympic_label == "clean_and_jerk"
        and float(state.olympic_conf or 0.0) >= 0.95
        and state.raw_label == "push_press"
        and state.bio_label == "push_press"
        and state.protected_label == "push_press"
        and state.protected_reason == "push_press_pattern_detected"
    )

    if strong_cj_over_push_press:
        _add_lock(
            locks,
            label="clean_and_jerk",
            confidence=max(
                0.97,
                float(state.olympic_conf or 0.0),
            ),
            reason="strong_cj_over_push_press_v52",
            priority=145,
        )

    # V52B: explosive overhead/split evidence supports clean-and-jerk
    # over an overhead-squat subtype when the Olympic router agrees.
    explosive_cj_over_overhead_squat = (
        state.olympic_label == "clean_and_jerk"
        and 0.60 <= float(state.olympic_conf or 0.0) < 0.70
        and state.raw_label == "squat"
        and state.bio_label == "squat"
        and state.squat_label == "overhead_squat"
        and float(state.squat_conf or 0.0) >= 0.80
        and float(state.explosive_score or 0.0) >= 80.0
        and bool(state.truly_explosive)
        and bool(state.looks_split)
        and bool(state.looks_thruster)
        and not bool(state.looks_cj)
    )

    if explosive_cj_over_overhead_squat:
        _add_lock(
            locks,
            label="clean_and_jerk",
            confidence=max(
                0.84,
                float(state.olympic_conf or 0.0),
            ),
            reason="explosive_cj_over_overhead_squat_v52",
            priority=146,
        )

    # V53A: strong deadlift evidence outranks a false burpee protection.
    deadlift_over_false_burpee = (
        state.raw_label == "deadlift"
        and float(state.raw_conf or 0.0) >= 0.995
        and state.bio_label == "squat"
        and state.squat_label == "squat_back"
        and state.protected_label == "burpee"
        and state.protected_reason == "burpee_bodyweight_pattern"
        and float(state.wrist_overhead or 0.0) < 0.05
        and bool(state.looks_clean)
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
    )

    if deadlift_over_false_burpee:
        _add_lock(
            locks,
            label="deadlift",
            confidence=max(
                0.96,
                float(state.raw_conf or 0.0),
            ),
            reason="deadlift_over_false_burpee_v53",
            priority=147,
        )

    # V53B: strong deadlift evidence outranks a false clean/Olympic subtype.
    deadlift_over_false_clean = (
        state.raw_label == "deadlift"
        and float(state.raw_conf or 0.0) >= 0.999
        and state.bio_label == "squat"
        and state.squat_label == "squat_front"
        and float(state.squat_conf or 0.0) >= 0.90
        and state.olympic_label == "snatch"
        and 0.70 <= float(state.olympic_conf or 0.0) < 0.80
        and float(state.wrist_overhead or 0.0) < 0.05
        and bool(state.looks_clean)
        and not bool(state.looks_cj)
        and not bool(state.looks_split)
    )

    if deadlift_over_false_clean:
        _add_lock(
            locks,
            label="deadlift",
            confidence=max(
                0.97,
                float(state.raw_conf or 0.0),
            ),
            reason="deadlift_over_false_clean_v53",
            priority=148,
        )

    # This does not use state.final_label. It uses the explicit protection
    # detector and its reason as another Router V8 evidence source.
    if (
        state.protected_label in SUPPORTED_LABELS
        and state.protected_reason
        and not strong_olympic_pull_up_conflict
        and not strong_cj_pull_up_conflict
        and not strong_snatch_bench_conflict
        and not strong_squat_bench_conflict
        and not strong_deadlift_bench_conflict
        and not unanimous_press_pull_up_conflict
        and not strong_snatch_bodyweight_conflict
        and not strong_snatch_cj_recovery_conflict
        and not strong_cj_trusted_bench_conflict
        and not strong_cj_overhead_squat_conflict
    ):
        protected_conf = max(
            float(state.final_conf or 0.0),
            raw_conf if state.raw_label == state.protected_label else 0.0,
            bio_conf if state.bio_label == state.protected_label else 0.0,
            bodyweight_conf
            if state.bodyweight_label == state.protected_label
            else 0.0,
            0.80,
        )

        _add_lock(
            locks,
            label=state.protected_label,
            confidence=protected_conf,
            reason=f"protected_evidence:{state.protected_reason}",
            priority=120,
        )

    # ==========================================================
    # 2. Clean and jerk shape
    # ==========================================================
    if (
        state.looks_cj
        and explosive >= 80.0
        and state.bodyweight_label not in {
            "push_up",
            "pull_up",
            "handstand_push_up",
        }
    ):
        _add_lock(
            locks,
            label="clean_and_jerk",
            confidence=max(0.86, olympic_conf),
            reason="clean_and_jerk_shape",
            priority=108,
        )

    # ==========================================================
    # 3. Split jerk shape
    # ==========================================================
    # A generic looks_split signal is noisy. Require:
    # - overhead-press context,
    # - strong Olympic confidence,
    # - explosive but not extreme movement,
    # - no clean-and-jerk shape.
    if (
        state.looks_split
        and not state.looks_cj
        and not state.looks_thruster
        and (
            state.raw_label == "push_press"
            or state.bio_label == "push_press"
        )
        and state.olympic_label in {
            "clean_and_jerk",
            "split_jerk",
        }
        and olympic_conf >= 0.85
        and 25.0 <= explosive <= 90.0
    ):
        _add_lock(
            locks,
            label="split_jerk",
            confidence=max(0.80, olympic_conf),
            reason="context_supported_split_jerk",
            priority=106,
        )

    # ==========================================================
    # 4. Snatch
    # ==========================================================
    if (
        state.olympic_label == "snatch"
        and olympic_conf >= 0.82
        and state.truly_explosive
        and wrist_overhead >= 0.45
        and not state.looks_cj
        and not (
            state.looks_split
            and (
                state.raw_label == "push_press"
                or state.bio_label == "push_press"
            )
        )
    ):
        _add_lock(
            locks,
            label="snatch",
            confidence=olympic_conf,
            reason="snatch_router_with_overhead_explosion",
            priority=104,
        )

    # ==========================================================
    # 5. Squat variant
    # ==========================================================
    squat_context = (
        state.raw_label in SQUAT_LABELS
        or state.bio_label in SQUAT_LABELS
    )

    if (
        squat_context
        and state.squat_label in {
            "squat_back",
            "squat_front",
            "overhead_squat",
        }
        and squat_conf >= 0.90
        and not olympic_shape_present
        and not state.truly_explosive
    

        and not strong_back_squat_front_lock_conflict):
        _add_lock(
            locks,
            label=state.squat_label,
            confidence=squat_conf,
            reason="context_supported_squat_variant",
            priority=100,
        )

    if (
        state.squat_label == "overhead_squat"
        and squat_conf >= 0.75
        and wrist_overhead >= 0.60
        and not olympic_shape_present
        and explosive < 40.0
    ):
        _add_lock(
            locks,
            label="overhead_squat",
            confidence=squat_conf,
            reason="overhead_squat_geometry",
            priority=102,
        )

    # ==========================================================
    # 6. Narrow push-press agreement
    # ==========================================================
    # This is the only direct base/biomechanics agreement lock retained.
    # Generic agreement caused false locks for HSPU, push-up, OHS and thruster.
    if (
        state.raw_label == "push_press"
        and state.bio_label == "push_press"
        and raw_conf >= 0.65
        and bio_conf >= 0.75
        and explosive < 30.0
        and not state.looks_split
        and not state.looks_cj
    ):
        _add_lock(
            locks,
            label="push_press",
            confidence=max(raw_conf, bio_conf),
            reason="context_supported_push_press_agreement",
            priority=101,
        )

    # Remove duplicate label locks while keeping the highest-priority version.
    ordered = sorted(
        locks,
        key=lambda item: (
            int(item.get("priority", 0)),
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )

    unique: list[dict] = []
    seen_labels: set[str] = set()

    for lock in ordered:
        label = str(lock.get("label"))
        if label in seen_labels:
            continue

        seen_labels.add(label)
        unique.append(lock)

    return unique
