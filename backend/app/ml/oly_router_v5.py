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

    # Safe initial behavior:
    # Do not override anything yet.
    if olympic_label in {"snatch", "clean", "clean_and_jerk", "split_jerk"}:
        events = detect_movement_events(biomechanics, olympic_label)
        features = build_oly_router_features(biomechanics, events)

        debug["events"] = events
        debug["features"] = features

    return raw_label, float(raw_confidence or 0.0), debug
