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

    # -------- Split Jerk diagnostics --------
    split_events = detect_movement_events(biomechanics, "split_jerk")
    split_features = build_oly_router_features(biomechanics, split_events)

    debug["split_events"] = split_events
    debug["split_features"] = split_features

    # Stage 0: Clean rescue from weak Snatch.
    # Clean clips can be weakly routed as snatch when wrists briefly appear overhead/noisy.
    if (
        olympic_label == "snatch"
        and float(olympic_confidence or 0.0) < 0.65
    ):
        clean_events = detect_movement_events(biomechanics, "clean")
        clean_features = build_oly_router_features(biomechanics, clean_events)

        debug["clean_events"] = clean_events
        debug["clean_features"] = clean_features

        if (
            clean_features.get("catch_depth", 0) > 100
            and clean_features.get("extension_to_catch", 0) >= 8
            and clean_features.get("catch_to_finish", 0) >= 8
        ):
            debug["decision"] = "clean_rescue_from_weak_snatch"
            return "clean", 0.75, debug

    # Stage 0b: Split Jerk rescue from C&J.
    # Standalone split jerks can be misread as C&J because the rack/dip/drive
    # looks like the second half of a clean and jerk.
    if (
        raw_label == "clean_and_jerk"
        and olympic_label == "clean_and_jerk"
        and features.get("catch_to_finish", 0) >= 400
        and features.get("lockout_duration", 0) >= 300
    ):
        debug["decision"] = "split_jerk_rescue_from_cj"
        return "split_jerk", max(0.80, float(olympic_confidence or 0.0)), debug

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
