from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MovementSignature:
    family: str
    internal_label: str
    positive_signals: tuple[str, ...]
    negative_signals: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


MOVEMENT_SIGNATURES: dict[str, MovementSignature] = {
    "back_squat": MovementSignature(
        family="squat",
        internal_label="squat_back",
        positive_signals=("deep squat", "bar behind shoulders", "not overhead"),
        negative_signals=("front rack", "explosive pull", "overhead lockout"),
        aliases=("squat_back",),
    ),
    "front_squat": MovementSignature(
        family="squat",
        internal_label="squat_front",
        positive_signals=("deep squat", "front rack", "elbows forward"),
        negative_signals=("overhead lockout", "clean pull", "back rack"),
        aliases=("squat_front",),
    ),
    "overhead_squat": MovementSignature(
        family="squat",
        internal_label="overhead_squat",
        positive_signals=("deep squat", "sustained overhead wrists", "locked elbows"),
        negative_signals=("jerk catch", "clean catch", "press dip"),
    ),
    "deadlift": MovementSignature(
        family="hinge",
        internal_label="deadlift",
        positive_signals=("hip hinge", "straight arms", "bar path from floor"),
        negative_signals=("front rack", "overhead lockout", "deep squat catch"),
    ),
    "bench_press": MovementSignature(
        family="press",
        internal_label="bench_press",
        positive_signals=("horizontal torso", "press from chest", "elbow extension"),
        negative_signals=("standing overhead", "deep squat", "pull-up hang"),
    ),
    "strict_press": MovementSignature(
        family="press",
        internal_label="strict_press",
        positive_signals=("upright torso", "overhead press", "minimal knee motion"),
        negative_signals=("dip drive", "deep squat", "clean catch"),
    ),
    "push_press": MovementSignature(
        family="press",
        internal_label="push_press",
        positive_signals=("upright torso", "knee dip", "overhead drive"),
        negative_signals=("deep squat", "split catch", "clean pull"),
    ),
    "thruster": MovementSignature(
        family="press",
        internal_label="thruster",
        positive_signals=("deep squat", "continuous squat to overhead press"),
        negative_signals=("clean pull", "front-rack catch before press", "split catch"),
    ),
    "pull_up": MovementSignature(
        family="bodyweight",
        internal_label="pull_up",
        positive_signals=("vertical hang", "body travels toward hands"),
        negative_signals=("barbell rack", "bench torso", "deep squat"),
    ),
    "push_up": MovementSignature(
        family="bodyweight",
        internal_label="push_up",
        positive_signals=("horizontal plank", "elbow bend and press"),
        negative_signals=("bench setup", "overhead motion", "vertical hang"),
    ),
    "handstand_push_up": MovementSignature(
        family="bodyweight",
        internal_label="handstand_push_up",
        positive_signals=("inverted body", "overhead bodyweight press"),
        negative_signals=("barbell overhead press", "bench torso", "deep squat"),
    ),
    "burpee": MovementSignature(
        family="bodyweight",
        internal_label="burpee",
        positive_signals=("stand to plank", "push-up floor phase", "jump/stand recovery"),
        negative_signals=("barbell rack", "pull-up hang", "bench setup"),
    ),
    "bar_muscle_up": MovementSignature(
        family="bodyweight",
        internal_label="muscle_up",
        positive_signals=("vertical pull", "transition over bar", "top support"),
        negative_signals=("ring support", "barbell rack", "bench torso"),
        aliases=("muscle_up",),
    ),
    "ring_muscle_up": MovementSignature(
        family="bodyweight",
        internal_label="muscle_up",
        positive_signals=("vertical pull", "ring transition", "ring dip support"),
        negative_signals=("bar support", "barbell rack", "bench torso"),
    ),
    "clean": MovementSignature(
        family="olympic",
        internal_label="clean",
        positive_signals=("explosive pull", "front-rack catch", "stand recovery"),
        negative_signals=("jerk after catch", "sustained overhead squat", "bench torso"),
    ),
    "clean_and_jerk": MovementSignature(
        family="olympic",
        internal_label="clean_and_jerk",
        positive_signals=("clean pull", "front-rack catch", "jerk dip drive", "overhead catch"),
        negative_signals=("single press from rack", "strict no-dip press", "bodyweight hang"),
    ),
    "split_jerk": MovementSignature(
        family="olympic",
        internal_label="split_jerk",
        positive_signals=("dip drive", "split catch", "overhead lockout"),
        negative_signals=("clean pull", "deep squat", "strict press"),
    ),
    "snatch": MovementSignature(
        family="olympic",
        internal_label="snatch",
        positive_signals=("explosive pull", "single overhead catch", "overhead recovery"),
        negative_signals=("front-rack catch", "bench torso", "pull-up hang"),
    ),
}


def _label_entries() -> dict[str, MovementSignature]:
    entries: dict[str, MovementSignature] = {}
    for public_label, signature in MOVEMENT_SIGNATURES.items():
        entries[public_label] = signature
        entries[signature.internal_label] = signature
        for alias in signature.aliases:
            entries[alias] = signature
    return entries


LABEL_SIGNATURES = _label_entries()
LABEL_TO_FAMILY = {
    label: signature.family
    for label, signature in LABEL_SIGNATURES.items()
}

PRESS_LABELS = {
    label
    for label, signature in LABEL_SIGNATURES.items()
    if signature.family == "press"
}
BODYWEIGHT_LABELS = {
    label
    for label, signature in LABEL_SIGNATURES.items()
    if signature.family == "bodyweight"
}
OLYMPIC_LABELS = {
    label
    for label, signature in LABEL_SIGNATURES.items()
    if signature.family == "olympic"
}
SQUAT_LABELS = {
    "squat",
    *{
        label
        for label, signature in LABEL_SIGNATURES.items()
        if signature.family == "squat"
    },
}
HINGE_LABELS = {
    label
    for label, signature in LABEL_SIGNATURES.items()
    if signature.family == "hinge"
}

SUPPORTED_FORCED_LABELS = {
    signature.internal_label
    for signature in MOVEMENT_SIGNATURES.values()
}


def normalize_forced_exercise_label(label: str | None) -> str | None:
    """Normalize user-supplied exercise labels to production labels."""
    if not label:
        return None

    requested_label = (
        str(label)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    signature = LABEL_SIGNATURES.get(requested_label)

    if signature is None:
        raise ValueError(f"Unsupported forced exercise label: {label}")

    normalized_label = signature.internal_label

    if normalized_label not in SUPPORTED_FORCED_LABELS:
        raise ValueError(f"Unsupported forced exercise label: {label}")

    return normalized_label
