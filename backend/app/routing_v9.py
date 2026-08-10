from dataclasses import dataclass
from typing import Optional


@dataclass
class RoutingCandidate:
    label: Optional[str]
    confidence: float = 0.0


@dataclass
class RoutingContext:
    base: RoutingCandidate
    biomechanics: RoutingCandidate
    squat: RoutingCandidate
    olympic: RoutingCandidate
    bodyweight: RoutingCandidate

    explosive_score: float = 0.0
    wrist_overhead_ratio: float = 0.0

    looks_clean: bool = False
    looks_clean_and_jerk: bool = False
    looks_split_jerk: bool = False
    looks_strict_press: bool = False
    looks_thruster: bool = False
    looks_push_up: bool = False
    looks_pull_up: bool = False
    looks_handstand_push_up: bool = False


@dataclass
class RoutingDecision:
    label: str
    confidence: float
    reason: str


def choose_exercise(ctx: RoutingContext) -> RoutingDecision:
    """
    Experimental simplified router.

    IMPORTANT:
    This is not wired into production yet.
    It exists so we can develop and benchmark a simpler arbitration system
    without changing the current analyzer.
    """

    return RoutingDecision(
        label=ctx.base.label or "unknown",
        confidence=float(ctx.base.confidence or 0.0),
        reason="v9_placeholder_base",
    )
