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
