from dataclasses import dataclass, field
from typing import Any


@dataclass
class RouterState:
    """
    Complete routing state for Router V8.

    This object is immutable in spirit:
    main.py gathers evidence,
    Router V8 makes decisions.
    """

    # ----------------------------------------------------------
    # Base classifier
    # ----------------------------------------------------------

    raw_label: str | None = None
    raw_conf: float = 0.0

    # ----------------------------------------------------------
    # Biomechanics
    # ----------------------------------------------------------

    bio_label: str | None = None
    bio_conf: float = 0.0

    # ----------------------------------------------------------
    # Specialized routers
    # ----------------------------------------------------------

    squat_label: str | None = None
    squat_conf: float = 0.0

    olympic_label: str | None = None
    olympic_conf: float = 0.0

    bodyweight_label: str | None = None
    bodyweight_conf: float = 0.0

    # ----------------------------------------------------------
    # Existing V7 decision
    # ----------------------------------------------------------

    final_label: str | None = None
    final_conf: float = 0.0
    analysis_mode: str | None = None

    # ----------------------------------------------------------
    # Protection system
    # ----------------------------------------------------------

    protected_label: str | None = None
    protected_reason: str | None = None

    # ----------------------------------------------------------
    # Motion descriptors
    # ----------------------------------------------------------

    explosive_score: float = 0.0
    wrist_overhead: float = 0.0

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------

    routing_trace: list = field(default_factory=list)
    router_scores: dict = field(default_factory=dict)

    # Preserve anything else without schema changes
    extras: dict[str, Any] = field(default_factory=dict)

    def as_prediction_inputs(self):
        """
        Compatibility layer for the existing collector.
        """
        return dict(
            raw_label=self.raw_label,
            raw_conf=self.raw_conf,
            bio_label=self.bio_label,
            bio_conf=self.bio_conf,
            squat_label=self.squat_label,
            squat_conf=self.squat_conf,
            olympic_label=self.olympic_label,
            olympic_conf=self.olympic_conf,
            bodyweight_label=self.bodyweight_label,
            bodyweight_conf=self.bodyweight_conf,
        )