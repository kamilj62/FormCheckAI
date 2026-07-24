from __future__ import annotations

from typing import Any


PRESS_LABELS = {
    "bench_press",
    "strict_press",
    "push_press",
    "thruster",
}


def _number(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def classify_press_variant_shadow(
    *,
    family: str | None,
    biomechanics_summary: dict[str, Any] | None,
    bodyweight_summary: dict[str, Any] | None,
    routing_candidates: dict[str, Any] | None,
    explosive_score: float,
    looks_strict: bool,
    looks_thruster: bool,
    strong_overhead: bool,
) -> dict[str, Any]:
    """
    Shadow-only press variant classifier.

    Runs only when the family router selects the press family.
    It does not modify production routing.
    """

    if family != "press":
        return {
            "version": "press_variant_shadow_v2",
            "eligible": False,
            "label": None,
            "reason": "family_not_press",
        }

    bio = biomechanics_summary or {}
    body = bodyweight_summary or {}
    candidates = routing_candidates or {}

    avg_torso = _number(
        bio,
        "avg_torso_angle",
        _number(body, "avg_torso_angle"),
    )
    min_knee = _number(bio, "min_knee_angle", 180.0)
    max_knee = _number(bio, "max_knee_angle", 180.0)
    min_hip = _number(bio, "min_hip_angle", 180.0)
    max_hip = _number(bio, "max_hip_angle", 180.0)

    knee_range = max(0.0, max_knee - min_knee)
    hip_range = max(0.0, max_hip - min_hip)

    shoulder_y_range = _number(body, "shoulder_y_range")
    hip_y_range = _number(body, "hip_y_range")
    wrist_overhead_ratio = _number(
        body,
        "wrist_above_shoulder_ratio",
    )

    squat_candidate = candidates.get("squat_router") or {}
    squat_label = squat_candidate.get("label")

    try:
        squat_conf = float(
            squat_candidate.get("confidence") or 0.0
        )
    except (TypeError, ValueError):
        squat_conf = 0.0

    base_candidate = candidates.get("base") or {}
    bio_candidate = candidates.get("biomechanics") or {}
    v5_candidate = candidates.get("router_v5") or {}

    source_labels = {
        base_candidate.get("label"),
        bio_candidate.get("label"),
        v5_candidate.get("label"),
    }

    horizontal = avg_torso >= 55.0
    upright = avg_torso < 25.0
    non_horizontal = avg_torso < 55.0

    scores = {
        "bench_press": 0.0,
        "strict_press": 0.0,
        "push_press": 0.0,
        "thruster": 0.0,
    }

    evidence: dict[str, list[str]] = {
        label: []
        for label in scores
    }

    def add(label: str, score: float, reason: str) -> None:
        scores[label] += float(score)
        evidence[label].append(reason)

    # ---------------------------------------------------------
    # Bench press: horizontal torso orientation.
    # ---------------------------------------------------------
    if horizontal:
        add(
            "bench_press",
            3.0,
            f"horizontal torso angle {avg_torso:.1f}",
        )

    if "bench_press" in source_labels:
        add(
            "bench_press",
            0.35,
            "existing classifier supports bench press",
        )

    # ---------------------------------------------------------
    # Thruster: deep squat followed by a press.
    # ---------------------------------------------------------
    deep_squat = (
        non_horizontal
        and min_knee <= 105.0
        and min_hip <= 115.0
        and knee_range >= 45.0
        and hip_range >= 40.0
    )

    front_squat_support = (
        non_horizontal
        and squat_label == "squat_front"
        and squat_conf >= 0.80
    )

    large_vertical_travel = (
        non_horizontal
        and hip_y_range >= 0.20
        and shoulder_y_range >= 0.18
    )

    if deep_squat:
        add(
            "thruster",
            1.5,
            (
                f"deep squat: min knee {min_knee:.1f}, "
                f"min hip {min_hip:.1f}"
            ),
        )

    if front_squat_support:
        add(
            "thruster",
            0.75,
            f"front squat router support {squat_conf:.3f}",
        )

    if large_vertical_travel:
        add(
            "thruster",
            0.45,
            "large hip and shoulder vertical travel",
        )

    if looks_thruster:
        add(
            "thruster",
            0.50,
            "existing thruster geometry signal",
        )

    # ---------------------------------------------------------
    # Strict press versus push press.
    # ---------------------------------------------------------
    small_leg_motion = (
        knee_range <= 13.0
        and hip_range <= 10.0
    )

    moderate_leg_drive = (
        knee_range >= 14.0
        and hip_range >= 11.0
    )

    if upright and small_leg_motion:
        add(
            "strict_press",
            1.25,
            (
                f"minimal leg motion: knee {knee_range:.1f}, "
                f"hip {hip_range:.1f}"
            ),
        )

    if looks_strict and explosive_score < 8.0:
        add(
            "strict_press",
            0.75,
            "strict pattern with low explosive score",
        )

    if upright and moderate_leg_drive:
        add(
            "push_press",
            1.25,
            (
                f"leg drive: knee {knee_range:.1f}, "
                f"hip {hip_range:.1f}"
            ),
        )

    if upright and explosive_score >= 8.0:
        add(
            "push_press",
            min(0.75, explosive_score / 30.0),
            f"explosive score {explosive_score:.1f}",
        )

    if (
        strong_overhead
        and "push_press" in source_labels
    ):
        add(
            "push_press",
            0.45,
            "overhead press classifier agreement",
        )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    winner = ranked[0]
    runner_up = ranked[1]

    return {
        "version": "press_variant_shadow_v1",
        "eligible": True,
        "label": winner[0],
        "score": round(winner[1], 3),
        "margin": round(winner[1] - runner_up[1], 3),
        "runner_up": {
            "label": runner_up[0],
            "score": round(runner_up[1], 3),
        },
        "scores": {
            label: round(score, 3)
            for label, score in ranked
        },
        "evidence": evidence,
        "features": {
            "avg_torso_angle": round(avg_torso, 3),
            "min_knee_angle": round(min_knee, 3),
            "knee_range": round(knee_range, 3),
            "min_hip_angle": round(min_hip, 3),
            "hip_range": round(hip_range, 3),
            "shoulder_y_range": round(shoulder_y_range, 3),
            "hip_y_range": round(hip_y_range, 3),
            "wrist_overhead_ratio": round(
                wrist_overhead_ratio,
                3,
            ),
            "explosive_score": round(
                float(explosive_score),
                3,
            ),
            "squat_label": squat_label,
            "squat_confidence": round(squat_conf, 3),
            "horizontal": horizontal,
            "non_horizontal": non_horizontal,
            "deep_squat": deep_squat,
            "front_squat_support": front_squat_support,
            "large_vertical_travel": large_vertical_travel,
            "upright": upright,
            "small_leg_motion": small_leg_motion,
            "moderate_leg_drive": moderate_leg_drive,
        },
    }
