from app.phase_engine.squat_visuals import choose_stable_squat_setup_frame


def row(frame, knee, hip, torso_lean=0):
    return {
        "frame": frame,
        "knee": knee,
        "hip": hip,
        "torso_lean": torso_lean,
    }


def test_chooses_strongest_stable_upright_setup():
    records = [
        row(20, 171, 160),
        row(22, 170, 159),
        row(24, 169, 158),
        row(26, 168, 157),
        row(28, 150, 130),
    ]

    assert choose_stable_squat_setup_frame(records, start=28, fallback=28) == 22


def test_recency_only_breaks_ties_between_equal_upright_frames():
    records = [
        row(20, 170, 160, 5),
        row(22, 170, 160, 5),
        row(24, 170, 160, 5),
        row(26, 150, 130, 20),
    ]

    assert choose_stable_squat_setup_frame(records, start=26, fallback=26) == 22


def test_prefers_upright_plateau_over_stable_early_hinge():
    records = [
        row(138, 175, 174, 4),
        row(140, 175, 174, 4),
        row(142, 175, 174, 4),
        row(160, 174, 172, 5),
        row(162, 175, 170, 7),
        row(164, 176, 168, 10),
        row(166, 172, 164, 12),
    ]

    assert choose_stable_squat_setup_frame(records, start=166, fallback=166) == 140


def test_rejects_isolated_upright_pose_spike():
    records = [
        row(20, 145, 125),
        row(22, 171, 160),
        row(24, 146, 126),
    ]

    assert choose_stable_squat_setup_frame(records, start=24, fallback=24) == 24


def test_rejects_center_frame_below_upright_threshold():
    records = [
        row(20, 164, 150),
        row(22, 150, 132),
        row(24, 164, 150),
    ]

    assert choose_stable_squat_setup_frame(records, start=24, fallback=24) == 24


def test_does_not_bridge_missing_pose_detections():
    records = [
        row(10, 170, 160),
        row(12, 169, 159),
        row(30, 168, 158),
    ]

    assert choose_stable_squat_setup_frame(records, start=30, fallback=30) == 30


def test_rejects_stable_forward_hinge_with_extended_legs():
    records = [
        row(20, 170, 155, 48),
        row(22, 169, 154, 50),
        row(24, 168, 153, 52),
    ]

    assert choose_stable_squat_setup_frame(records, start=24, fallback=24) == 24


def test_accepts_stable_upright_geometry_from_oblique_camera():
    records = [
        row(40, 159, 141, 28),
        row(42, 160, 142, 29),
        row(44, 158, 140, 30),
    ]

    assert choose_stable_squat_setup_frame(records, start=44, fallback=44) == 42


def test_falls_back_when_setup_hold_is_too_short():
    records = [row(10, 170, 160), row(12, 169, 159)]

    assert choose_stable_squat_setup_frame(records, start=12, fallback=9) == 9


def test_can_signal_that_no_pre_start_setup_exists():
    records = [
        row(60, 100, 90, 40),
        row(62, 95, 85, 42),
        row(64, 90, 80, 45),
    ]

    assert choose_stable_squat_setup_frame(
        records,
        start=64,
        fallback=None,
    ) is None
