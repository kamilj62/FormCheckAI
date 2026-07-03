"""
Olympic Router V5.

Goal:
Centralize Olympic routing decisions outside app/main.py.
This starts as a safe wrapper and can gradually absorb routing logic.
"""

from app.movement.event_detector import detect_movement_events
from app.ml.oly_router_features import build_oly_router_features


def route_olympic_lift(
    biomechanics,
    raw_label,
    raw_confidence,
    olympic_label,
    olympic_confidence,
):
    """
    Return:
        final_label, final_confidence, debug
    """

    debug = {
        "raw_label": raw_label,
        "raw_confidence": float(raw_confidence or 0.0),
        "olympic_label": olympic_label,
        "olympic_confidence": float(olympic_confidence or 0.0),
        "router": "v5_scaffold",
    }

    if olympic_label in {"snatch", "clean", "clean_and_jerk", "split_jerk"}:
        events = detect_movement_events(biomechanics, olympic_label)
        features = build_oly_router_features(biomechanics, events)

        debug["events"] = events
        debug["features"] = features

    # Stage 1: unanimous agreement.
    if (
        raw_label == olympic_label
        and olympic_label in {"snatch", "clean", "clean_and_jerk", "split_jerk"}
    ):
        debug["decision"] = "agreement"
        return olympic_label, max(float(raw_confidence or 0.0), float(olympic_confidence or 0.0)), debug

    # ------------------------------------------------------------------
    # Stage 2: Clean & Jerk rescue from Push Press
    # ------------------------------------------------------------------
    if raw_label == "push_press":

        # Evaluate the clip as a clean & jerk regardless of what the
        # Olympic classifier predicted.
        cj_events = detect_movement_events(biomechanics, "clean_and_jerk")
        cj_features = build_oly_router_features(biomechanics, cj_events)

        debug["cj_events"] = cj_events
        debug["cj_features"] = cj_features

        if (
            cj_features.get("extension_to_catch", 0) >= 8
            and (
                cj_features.get("catch_depth", 0) >= 40
                or cj_features.get("catch_to_finish", 0) >= 40
                or cj_features.get("lockout_duration", 0) >= 30
            )
        ):
            debug["decision"] = "cj_rescue_from_push_press"
            return (
                "clean_and_jerk",
                max(float(raw_confidence or 0.0), float(olympic_confidence or 0.0)),
                debug,
            )

    # ------------------------------------------------------------------
    # Stage 3: Snatch rescue from Squat
    # ------------------------------------------------------------------
    if (
        raw_label in {"squat", "squat_back", "squat_front", "overhead_squat"}
        and olympic_label == "snatch"
        and float(olympic_confidence or 0.0) >= 0.50
    ):
        features = debug.get("features", {})

        if (
            features.get("has_overhead", 0) >= 1
            and features.get("catch_overhead", 0) >= 1
            and features.get("extension_to_catch", 0) >= 8
            and features.get("lockout_duration", 0) >= 8
        ):
            debug["decision"] = "snatch_rescue_from_squat"
            return "snatch", max(0.70, float(olympic_confidence or 0.0)), debug

    debug["decision"] = "fallback"
    return raw_label, float(raw_confidence or 0.0), debug
