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
