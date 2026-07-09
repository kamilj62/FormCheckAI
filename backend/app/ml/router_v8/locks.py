from .state import RouterState


def get_locks(state: RouterState) -> list[dict]:
    locks = []

    if (
        state.raw_label in {"squat", "squat_back", "squat_front", "overhead_squat"}
        and state.squat_label in {"squat_back", "squat_front", "overhead_squat"}
        and float(state.squat_conf or 0.0) >= 0.90
    ):
        locks.append({
            "label": state.squat_label,
            "confidence": float(state.squat_conf or 0.0),
            "reason": "clear_squat_variant",
        })

    if (
        state.bodyweight_label in {"push_up", "pull_up", "handstand_push_up"}
        and float(state.bodyweight_conf or 0.0) >= 0.97
        and state.raw_label in {"push_up", "pull_up", "handstand_push_up", "deadlift", "bench_press"}
    ):
        locks.append({
            "label": state.bodyweight_label,
            "confidence": float(state.bodyweight_conf or 0.0),
            "reason": "clear_bodyweight_router",
        })

    if (
        state.olympic_label in {"clean", "clean_and_jerk", "snatch", "split_jerk"}
        and float(state.olympic_conf or 0.0) >= 0.90
        and float(state.explosive_score or 0.0) >= 20.0
    ):
        locks.append({
            "label": state.olympic_label,
            "confidence": float(state.olympic_conf or 0.0),
            "reason": "clear_olympic_router",
        })

    return locks
