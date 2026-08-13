import numpy as np

from app.ml.rep_detector import (
    dedupe_thruster_full_cycles,
    detect_reps_for_label,
    filter_short_squat_fragments,
    fill_push_press_gap_reps,
    normalize_rep_detector_label,
    rep_detector_spec,
    recover_long_overhead_squat_clip,
    recover_push_press_cycles,
    recover_squat_rhythm_clip,
    recover_single_clean_rep,
    recover_split_jerk_cycles,
    split_long_single_rep,
    validate_rep_phases,
)


def old_thruster_fragment_result():
    return [
        {
            "rep": 1,
            "start_frame": 121,
            "dip_frame": 184,
            "drive_frame": 206,
            "catch_frame": 262,
            "lockout_frame": 287,
            "end_frame": 287,
            "score": 9.2,
        },
        {
            "rep": 2,
            "start_frame": 298,
            "dip_frame": 304,
            "drive_frame": 323,
            "catch_frame": 343,
            "lockout_frame": 363,
            "end_frame": 363,
            "score": 9.2,
        },
        {
            "rep": 3,
            "start_frame": 308,
            "dip_frame": 341,
            "drive_frame": 359,
            "catch_frame": 423,
            "lockout_frame": 444,
            "end_frame": 444,
            "score": 9.2,
        },
        {
            "rep": 4,
            "start_frame": 454,
            "dip_frame": 533,
            "drive_frame": 560,
            "catch_frame": 582,
            "lockout_frame": 602,
            "end_frame": 602,
            "score": 9.2,
        },
        {
            "rep": 5,
            "start_frame": 616,
            "dip_frame": 680,
            "drive_frame": 710,
            "catch_frame": 732,
            "lockout_frame": 757,
            "end_frame": 757,
            "score": 9.2,
        },
        {
            "rep": 6,
            "start_frame": 697,
            "dip_frame": 707,
            "drive_frame": 725,
            "catch_frame": 757,
            "lockout_frame": 763,
            "end_frame": 763,
            "score": 9.2,
        },
    ]


def clean_shaped_biomechanics():
    records = []
    for index in range(80):
        if index < 28:
            wrist_drop = 0.18
            knee = 170
            hip = 165
            hip_y = 0.62
        elif index < 38:
            wrist_drop = 0.12
            knee = 176
            hip = 174
            hip_y = 0.58
        elif index < 48:
            wrist_drop = 0.02
            knee = 118
            hip = 125
            hip_y = 0.76
        else:
            wrist_drop = 0.04
            knee = 160
            hip = 158
            hip_y = 0.61

        records.append({
            "frame_number": index * 3,
            "knee_angle": knee,
            "hip_angle": hip,
            "hip_y": hip_y,
            "wrist_y": 0.50 + wrist_drop,
            "shoulder_y": 0.50,
            "wrist_x": 0.51,
            "shoulder_x": 0.50,
        })

    return records


def push_press_cycle_biomechanics(reps=5):
    records = []
    frame = 0

    for _ in range(reps):
        cycle = [
            (172, 0.56, 0.50, 120),
            (166, 0.58, 0.50, 125),
            (160, 0.57, 0.50, 130),
            (168, 0.48, 0.50, 145),
            (176, 0.42, 0.50, 158),
            (178, 0.40, 0.50, 165),
            (176, 0.42, 0.50, 164),
            (174, 0.52, 0.50, 145),
        ]

        for knee, wrist_y, shoulder_y, elbow in cycle:
            records.append({
                "frame_number": frame,
                "knee_angle": knee,
                "hip_angle": 170,
                "wrist_y": wrist_y,
                "shoulder_y": shoulder_y,
                "elbow_angle": elbow,
            })
            frame += 5

    return records


def split_jerk_cycle_biomechanics(reps=4):
    records = []
    frame = 0

    for _ in range(reps):
        cycle = [
            (175, 172, 0.54, 0.50, 125),
            (164, 166, 0.55, 0.50, 130),
            (158, 160, 0.54, 0.50, 135),
            (172, 174, 0.47, 0.50, 145),
            (176, 176, 0.41, 0.50, 158),
            (174, 174, 0.39, 0.50, 165),
            (174, 174, 0.40, 0.50, 164),
            (170, 170, 0.52, 0.50, 145),
        ]

        for knee, hip, wrist_y, shoulder_y, elbow in cycle:
            records.append({
                "frame_number": frame,
                "knee_angle": knee,
                "hip_angle": hip,
                "wrist_y": wrist_y,
                "shoulder_y": shoulder_y,
                "elbow_angle": elbow,
            })
            frame += 5

    return records


def squat_rhythm_biomechanics(frames=96):
    records = []

    for index in range(frames):
        phase = index / max(1, frames - 1)
        first_dip = np.exp(-((phase - 0.32) ** 2) / 0.0025)
        second_dip = np.exp(-((phase - 0.68) ** 2) / 0.0025)
        dip = max(first_dip, second_dip)

        records.append({
            "frame_number": index,
            "knee_angle": 172 - 52 * dip,
            "hip_angle": 166 - 42 * dip,
            "hip_y": 0.56 + 0.14 * dip,
        })

    return records


def test_rep_detector_normalizes_aliases_to_runtime_labels():
    assert normalize_rep_detector_label("Back Squat") == "squat_back"
    assert normalize_rep_detector_label("handstand-push-up") == (
        "handstand_push_up"
    )


def test_rep_detector_routes_thruster_through_press_detector():
    calls = []

    def press_detector(biomechanics, exercise_label):
        calls.append((biomechanics, exercise_label))
        return (
            [
                {
                    "rep": 1,
                    "start_frame": 1,
                    "dip_frame": 2,
                    "drive_frame": 3,
                    "catch_frame": 4,
                    "lockout_frame": 5,
                    "end_frame": 6,
                }
            ],
            {"detected_reps": 1},
        )

    result = detect_reps_for_label(
        label="thruster",
        biomechanics=[{"frame_number": 1}],
        detectors={"push_press": press_detector},
    )

    assert calls == [([{"frame_number": 1}], "thruster")]
    assert result.error is None
    assert len(result.reps) == 1
    assert result.phase_complete is True
    assert result.phase_ordered is True


def test_rep_detector_dedupes_overlapping_thruster_fragments():
    cleaned = dedupe_thruster_full_cycles(old_thruster_fragment_result())

    assert [rep["rep"] for rep in cleaned] == [1, 2, 3]
    assert [(rep["start_frame"], rep["end_frame"]) for rep in cleaned] == [
        (121, 444),
        (454, 602),
        (616, 763),
    ]
    assert cleaned[0]["breakdown"]["merged_fragments"] == 3
    assert cleaned[1]["breakdown"]["merged_fragments"] == 1
    assert cleaned[2]["breakdown"]["merged_fragments"] == 2


def test_rep_detector_dedupes_thrusters_on_route():
    def press_detector(biomechanics, exercise_label):
        return old_thruster_fragment_result(), {"detected_reps": 6}

    result = detect_reps_for_label(
        label="thruster",
        biomechanics=[{"frame_number": 1}],
        detectors={"push_press": press_detector},
    )

    assert len(result.reps) == 3
    assert result.summary["detected_reps"] == 3
    assert result.phase_complete is True
    assert result.phase_ordered is True


def test_rep_detector_dedupes_start_end_thruster_fragments():
    fragments = [
        {"rep": 1, "start_frame": 220, "end_frame": 278, "score": 9.0},
        {"rep": 2, "start_frame": 246, "end_frame": 309, "score": 9.0},
        {"rep": 3, "start_frame": 272, "end_frame": 330, "score": 9.6},
        {"rep": 4, "start_frame": 299, "end_frame": 357, "score": 7.8},
        {"rep": 5, "start_frame": 317, "end_frame": 375, "score": 9.0},
        {"rep": 6, "start_frame": 359, "end_frame": 430, "score": 10.0},
        {"rep": 7, "start_frame": 398, "end_frame": 456, "score": 10.0},
        {"rep": 8, "start_frame": 575, "end_frame": 642, "score": 9.0},
    ]

    cleaned = dedupe_thruster_full_cycles(fragments)

    assert len(cleaned) == 3
    assert [(rep["start_frame"], rep["end_frame"]) for rep in cleaned] == [
        (220, 375),
        (359, 456),
        (575, 642),
    ]


def test_rep_detector_filters_short_front_squat_fragments():
    fragments = [
        {"rep": 1, "start_frame": 66, "end_frame": 78, "score": 6.0},
        {"rep": 2, "start_frame": 157, "end_frame": 185, "score": 8.0},
        {"rep": 3, "start_frame": 317, "end_frame": 348, "score": 8.0},
        {"rep": 4, "start_frame": 374, "end_frame": 386, "score": 6.0},
    ]

    filtered = filter_short_squat_fragments(fragments)

    assert len(filtered) == 2
    assert [(rep["start_frame"], rep["end_frame"]) for rep in filtered] == [
        (157, 185),
        (317, 348),
    ]
    assert [rep["rep"] for rep in filtered] == [1, 2]


def test_rep_detector_front_squat_falls_back_to_broad_squat_cycles():
    def squat_detector(biomechanics, exercise_label):
        if exercise_label == "squat_front":
            return [], {"detected_reps": 0}
        return (
            [
                {"rep": 1, "start_frame": 66, "descent_frame": 69, "bottom_frame": 73, "ascent_frame": 75, "end_frame": 78},
                {"rep": 2, "start_frame": 157, "descent_frame": 161, "bottom_frame": 166, "ascent_frame": 174, "end_frame": 185},
                {"rep": 3, "start_frame": 317, "descent_frame": 319, "bottom_frame": 321, "ascent_frame": 333, "end_frame": 348},
                {"rep": 4, "start_frame": 374, "descent_frame": 378, "bottom_frame": 382, "ascent_frame": 383, "end_frame": 386},
            ],
            {"detected_reps": 4},
        )

    result = detect_reps_for_label(
        label="squat_front",
        biomechanics=[{"frame_number": index} for index in range(450)],
        detectors={"squat": squat_detector},
    )

    assert len(result.reps) == 2
    assert result.summary["detected_reps"] == 2
    assert result.phase_complete is True


def test_rep_detector_recovers_front_squat_from_squat_rhythm():
    def empty_squat_detector(biomechanics, exercise_label):
        return [], {"detected_reps": 0}

    result = detect_reps_for_label(
        label="squat_front",
        biomechanics=squat_rhythm_biomechanics(),
        detectors={"squat": empty_squat_detector},
    )

    assert len(result.reps) == 2
    assert result.summary["detected_reps"] == 2
    assert result.phase_complete is True
    assert result.phase_ordered is True


def test_rep_detector_recovers_long_overhead_squat_from_tiny_fragment():
    recovered = recover_long_overhead_squat_clip(
        [
            {
                "rep": 1,
                "start_frame": 85,
                "descent_frame": 85,
                "bottom_frame": 86,
                "ascent_frame": 87,
                "end_frame": 89,
            }
        ],
        total_frames=255,
    )

    assert len(recovered) == 2
    assert [rep["rep"] for rep in recovered] == [1, 2]
    assert all(rep["end_frame"] > rep["start_frame"] for rep in recovered)


def test_rep_detector_recovers_short_overhead_squat_clip():
    recovered = recover_long_overhead_squat_clip(
        [
            {
                "rep": 1,
                "start_frame": 40,
                "descent_frame": 41,
                "bottom_frame": 43,
                "ascent_frame": 51,
                "end_frame": 61,
            }
        ],
        total_frames=96,
    )

    assert len(recovered) == 2


def test_squat_rhythm_recovery_requires_motion():
    static_records = [
        {
            "frame_number": index,
            "knee_angle": 175,
            "hip_angle": 170,
            "hip_y": 0.55,
        }
        for index in range(96)
    ]

    assert recover_squat_rhythm_clip(
        static_records,
        [],
        label="front_squat",
    ) == []


def test_rep_detector_splits_long_single_pull_cycle():
    reps = [
        {
            "rep": 1,
            "start_frame": 73,
            "pull_frame": 124,
            "top_frame": 134,
            "descent_frame": 148,
            "end_frame": 165,
            "score": 9.0,
        }
    ]

    split = split_long_single_rep(
        reps,
        total_frames=176,
        phase_fields=(
            "start_frame",
            "pull_frame",
            "top_frame",
            "descent_frame",
            "end_frame",
        ),
        min_span=80,
    )

    assert len(split) == 2
    assert split[0]["end_frame"] < split[1]["start_frame"]
    assert [rep["rep"] for rep in split] == [1, 2]


def test_rep_detector_recovers_single_clean_when_primary_detector_misses():
    def empty_clean_detector(biomechanics):
        return [], {"detected_reps": 0}

    result = detect_reps_for_label(
        label="clean",
        biomechanics=clean_shaped_biomechanics(),
        detectors={"clean": empty_clean_detector},
    )

    assert len(result.reps) == 1
    assert result.summary["detected_reps"] == 1
    assert result.phase_complete is True
    assert result.phase_ordered is True
    assert result.reps[0]["breakdown"]["catch"] == "recovered"


def test_rep_detector_recovers_clean_from_broad_pull_shape():
    records = clean_shaped_biomechanics()
    for record in records:
        record["wrist_x"] = 0.95

    assert recover_single_clean_rep(records) is not None


def test_clean_recovery_does_not_create_rep_without_front_rack():
    records = clean_shaped_biomechanics()
    for record in records:
        record["wrist_y"] = 0.95

    assert recover_single_clean_rep(records) is None


def test_rep_detector_recovers_multi_rep_push_press_cycles():
    recovered = recover_push_press_cycles(
        push_press_cycle_biomechanics(5),
        existing_reps=[{"rep": 1, "start_frame": 0, "end_frame": 30}],
    )

    assert len(recovered) == 5
    assert all("lockout_frame" in rep for rep in recovered)


def test_rep_detector_fills_large_push_press_gap():
    existing = [
        {"rep": 1, "start_frame": 2, "dip_frame": 12, "drive_frame": 22, "lockout_frame": 42, "end_frame": 68},
        {"rep": 2, "start_frame": 90, "dip_frame": 118, "drive_frame": 121, "lockout_frame": 127, "end_frame": 150},
        {"rep": 3, "start_frame": 172, "dip_frame": 207, "drive_frame": 208, "lockout_frame": 208, "end_frame": 225},
        {"rep": 4, "start_frame": 252, "dip_frame": 284, "drive_frame": 285, "lockout_frame": 285, "end_frame": 296},
    ]

    recovered = fill_push_press_gap_reps(
        [{"frame_number": index} for index in range(301)],
        existing,
    )

    assert len(recovered) == 5


def test_rep_detector_recovers_split_jerk_cycles():
    recovered = recover_split_jerk_cycles(
        split_jerk_cycle_biomechanics(4),
        existing_reps=[
            {"rep": 1, "start_frame": 0, "end_frame": 30},
            {"rep": 2, "start_frame": 80, "end_frame": 110},
        ],
    )

    assert len(recovered) == 4
    assert all("catch_frame" in rep for rep in recovered)


def test_rep_detector_keeps_primary_split_jerk_when_recovery_overcounts():
    existing = [
        {"rep": 1, "start_frame": 1, "dip_frame": 10, "drive_frame": 20, "catch_frame": 35, "lockout_frame": 40, "end_frame": 90},
        {"rep": 2, "start_frame": 120, "dip_frame": 130, "drive_frame": 140, "catch_frame": 160, "lockout_frame": 170, "end_frame": 220},
        {"rep": 3, "start_frame": 250, "dip_frame": 260, "drive_frame": 270, "catch_frame": 300, "lockout_frame": 310, "end_frame": 360},
        {"rep": 4, "start_frame": 390, "dip_frame": 400, "drive_frame": 410, "catch_frame": 450, "lockout_frame": 460, "end_frame": 520},
    ]

    recovered = recover_split_jerk_cycles(
        split_jerk_cycle_biomechanics(7),
        existing_reps=existing,
    )

    assert len(recovered) == 4
    assert recovered == existing


def test_rep_detector_recovers_clipped_opening_split_jerk():
    existing = [
        {"rep": 1, "start_frame": 170, "dip_frame": 171, "drive_frame": 172, "catch_frame": 173, "lockout_frame": 176, "end_frame": 228},
        {"rep": 2, "start_frame": 287, "dip_frame": 295, "drive_frame": 296, "catch_frame": 297, "lockout_frame": 300, "end_frame": 349},
    ]

    recovered = recover_split_jerk_cycles(
        split_jerk_cycle_biomechanics(3),
        existing_reps=existing,
    )

    assert len(recovered) >= 3
    assert recovered[0]["start_frame"] < 100


def test_rep_detector_reports_missing_detector():
    result = detect_reps_for_label(
        label="push_up",
        biomechanics=[],
        detectors={},
    )

    assert result.error == "missing_detector:push_up"
    assert result.reps == []
    assert result.phase_complete is False


def test_rep_phase_validation_finds_missing_and_unordered_fields():
    spec = rep_detector_spec("strict_press")
    validations = validate_rep_phases(
        [
            {
                "rep": 1,
                "start_frame": 10,
                "press_frame": 20,
                "lockout_frame": 15,
            }
        ],
        spec.required_phase_fields,
    )

    assert validations[0].complete is False
    assert validations[0].ordered is False
    assert validations[0].missing_fields == ("end_frame",)
