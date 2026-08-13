from app.ml.movement_signatures import (
    BODYWEIGHT_LABELS,
    LABEL_TO_FAMILY,
    MOVEMENT_SIGNATURES,
    OLYMPIC_LABELS,
    PRESS_LABELS,
    SQUAT_LABELS,
    normalize_forced_exercise_label,
)


def test_movement_signatures_cover_18_public_exercises():
    assert len(MOVEMENT_SIGNATURES) == 18


def test_key_router_family_assignments_are_canonical():
    assert LABEL_TO_FAMILY["thruster"] == "press"
    assert LABEL_TO_FAMILY["handstand_push_up"] == "bodyweight"
    assert LABEL_TO_FAMILY["front_squat"] == "squat"
    assert LABEL_TO_FAMILY["squat_front"] == "squat"
    assert LABEL_TO_FAMILY["clean_and_jerk"] == "olympic"
    assert LABEL_TO_FAMILY["deadlift"] == "hinge"


def test_family_label_sets_come_from_signatures():
    assert "thruster" in PRESS_LABELS
    assert "thruster" not in OLYMPIC_LABELS
    assert "handstand_push_up" in BODYWEIGHT_LABELS
    assert "bar_muscle_up" in BODYWEIGHT_LABELS
    assert "ring_muscle_up" in BODYWEIGHT_LABELS
    assert "squat_front" in SQUAT_LABELS


def test_forced_label_normalization_uses_canonical_signatures():
    assert normalize_forced_exercise_label("back squat") == "squat_back"
    assert normalize_forced_exercise_label("front-squat") == "squat_front"
    assert normalize_forced_exercise_label("bar_muscle_up") == "muscle_up"
    assert normalize_forced_exercise_label("ring-muscle-up") == "muscle_up"
    assert normalize_forced_exercise_label("thruster") == "thruster"


def test_forced_label_normalization_rejects_unknown_labels():
    try:
        normalize_forced_exercise_label("not a lift")
    except ValueError as exc:
        assert "Unsupported forced exercise label" in str(exc)
    else:
        raise AssertionError("Expected unsupported forced label to raise")
