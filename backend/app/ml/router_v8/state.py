from dataclasses import dataclass, field
from typing import Any


@dataclass
class RouterState:
    """
    Complete evidence state for Router V8.

    Router V8 remains shadow-only. The state contains classifier opinions plus
    movement-shape evidence, but V8 does not modify production routing.
    """

    # Base classifier
    raw_label: str | None = None
    raw_conf: float = 0.0

    # Biomechanics classifier
    bio_label: str | None = None
    bio_conf: float = 0.0
    bio_reason: str | None = None

    # Specialist routers
    squat_label: str | None = None
    squat_conf: float = 0.0

    olympic_label: str | None = None
    olympic_conf: float = 0.0

    bodyweight_label: str | None = None
    bodyweight_conf: float = 0.0

    # Existing production result: diagnostic only.
    # V8 fusion must not use these fields to choose its winner.
    final_label: str | None = None
    final_conf: float = 0.0
    analysis_mode: str | None = None

    # Existing protection diagnostics
    protected_label: str | None = None
    protected_reason: str | None = None

    # Motion descriptors
    explosive_score: float = 0.0
    wrist_overhead: float = 0.0

    looks_clean: bool = False
    looks_cj: bool = False
    looks_split: bool = False
    looks_strict: bool = False
    looks_thruster: bool = False
    truly_explosive: bool = False
    bar_pos_valid: bool = False

    # Diagnostics
    routing_trace: list = field(default_factory=list)
    router_scores: dict = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def as_prediction_inputs(self):
        return {
            "raw_label": self.raw_label,
            "raw_conf": self.raw_conf,
            "bio_label": self.bio_label,
            "bio_conf": self.bio_conf,
            "squat_label": self.squat_label,
            "squat_conf": self.squat_conf,
            "olympic_label": self.olympic_label,
            "olympic_conf": self.olympic_conf,
            "bodyweight_label": self.bodyweight_label,
            "bodyweight_conf": self.bodyweight_conf,
        }