from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ml.movement_signatures import (
    BODYWEIGHT_LABELS,
    OLYMPIC_LABELS,
    PRESS_LABELS,
    SQUAT_LABELS,
)


@dataclass
class FinalClassifierDecision:
    label: str | None
    confidence: float
    mode: str
    reason: str | None = None
    changed: bool = False


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _signals(data: dict[str, Any] | None) -> dict[str, Any]:
    value = (data or {}).get("signals") or {}
    return value if isinstance(value, dict) else {}


def simplify_final_classification(
    *,
    current_label: str | None,
    current_confidence: float,
    current_mode: str | None,
    forced_label: str | None,
    family_shadow: dict[str, Any] | None,
    press_variant_shadow: dict[str, Any] | None,
    hierarchical_shadow: dict[str, Any] | None,
    specialist_router_stack: dict[str, Any] | None = None,
) -> FinalClassifierDecision:
    """
    Conservative production bridge for the simplified hierarchy.

    The existing classifier still produces the first answer. This arbiter only
    changes it when the family-first shadow route has specific, trusted support.
    """

    current_conf = _float(current_confidence)

    base = FinalClassifierDecision(
        label=current_label,
        confidence=current_conf,
        mode=current_mode or "model_prediction",
        reason=None,
        changed=False,
    )

    if forced_label:
        return base

    hierarchical = hierarchical_shadow or {}
    if not hierarchical.get("eligible"):
        return base

    label = hierarchical.get("label")
    family = hierarchical.get("family")
    source = hierarchical.get("source")

    if not label or label == current_label:
        return base

    family_result = family_shadow or {}
    press_result = press_variant_shadow or {}
    specialist_stack = specialist_router_stack or {}
    family_margin = _float(family_result.get("margin"))
    family_score = _float(family_result.get("score"))
    exact_conf = _float(hierarchical.get("confidence"))
    family_signals = _signals(family_result)
    specialist_winner = specialist_stack.get("winner") or {}
    specialist_margin = _float(specialist_stack.get("margin"))

    def promote(confidence: float, reason: str) -> FinalClassifierDecision:
        return FinalClassifierDecision(
            label=str(label),
            confidence=min(
                1.0,
                max(0.0, max(current_conf, confidence)),
            ),
            mode="simplified_classifier",
            reason=reason,
            changed=True,
        )

    # Missing/unknown production labels are the lowest-risk place to accept a
    # clear hierarchy result.
    if current_label in {None, "", "unknown", "Unknown"}:
        if family_margin >= 0.25 and exact_conf > 0.0:
            return promote(exact_conf, "hierarchy_over_unknown")

    protected_bodyweight = bool(
        family_signals.get("protected_bodyweight_supported")
    )

    if (
        family == "bodyweight"
        and label in BODYWEIGHT_LABELS
        and protected_bodyweight
        and current_label not in OLYMPIC_LABELS
    ):
        return promote(
            max(exact_conf, 0.86),
            "protected_bodyweight_hierarchy",
        )

    if family == "press" and label in PRESS_LABELS:
        press_margin = _float(press_result.get("margin"))
        press_score = _float(press_result.get("score"))
        press_features = press_result.get("features") or {}

        strong_push_press = (
            label == "push_press"
            and (
                press_features.get("push_press_consensus")
                or press_features.get("protected_push_press")
            )
        )

        strong_strict_press = (
            label == "strict_press"
            and (
                press_features.get("strict_press_geometry")
                or press_features.get("strict_press_consensus")
                or press_features.get("protected_strict_press")
            )
        )

        protected_current_olympic = (
            current_label in OLYMPIC_LABELS
            and current_conf >= 0.80
            and str(current_mode or "") in {
                "router_v5",
                "olympic_locked",
                "router_v8_context_lock",
            }
        )

        specialist_press_winner = (
            specialist_winner.get("router") == "press"
            and specialist_winner.get("label") == label
            and specialist_margin >= 0.20
        )

        if (
            not protected_current_olympic
            and family_margin >= 0.25
            and press_margin >= 0.15
            and press_score >= 1.0
            and (
                strong_push_press
                or strong_strict_press
                or specialist_press_winner
                or current_label in PRESS_LABELS
            )
        ):
            return promote(
                max(exact_conf, press_score),
                "press_hierarchy_variant",
            )

    if (
        family == "squat"
        and label in SQUAT_LABELS
        and source == "squat_router"
        and (
            family_margin >= 0.75
            or (
                specialist_winner.get("router") == "squat"
                and specialist_winner.get("label") == label
                and specialist_margin >= 0.30
            )
        )
        and exact_conf >= 0.80
        and current_label not in BODYWEIGHT_LABELS
        and not (
            current_label in OLYMPIC_LABELS
            and current_conf >= 0.80
        )
    ):
        return promote(exact_conf, "squat_hierarchy_variant")

    if (
        family == "olympic"
        and label in OLYMPIC_LABELS
        and source in {"router_v5", "olympic_router"}
        and (
            family_margin >= 0.30
            or (
                specialist_winner.get("router") == "olympic"
                and specialist_winner.get("label") == label
                and specialist_margin >= 0.25
            )
        )
        and exact_conf >= 0.70
        and family_score >= 1.0
    ):
        return promote(exact_conf, "olympic_hierarchy_variant")

    return base
