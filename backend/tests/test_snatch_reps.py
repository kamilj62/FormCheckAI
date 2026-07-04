from app.main import (
    analyze_clean_reps,
    analyze_snatch_reps,
    analyze_split_jerk_reps,
    looks_like_clean_only,
    looks_like_split_jerk,
)


def test_snatch_reps_detects_multiple_overhead_catches():
    biomechanics = []
    catch_centers = [45, 120, 195, 270]

    for frame in range(320):
        overhead = any(center <= frame <= center + 8 for center in catch_centers)
        in_pull = any(center - 25 <= frame < center for center in catch_centers)
        catch_bottom = overhead and any(frame == center + 2 for center in catch_centers)

        biomechanics.append({
            "frame_number": frame,
            "hip_y": 0.72 if catch_bottom else (0.42 if overhead else 0.50),
            "wrist_y": 0.30 if overhead else 0.82,
            "shoulder_y": 0.55,
            "hip_angle": 178.0 if in_pull else (95.0 if overhead else 160.0),
            "knee_angle": 176.0 if in_pull else (105.0 if overhead else 165.0),
        })

    reps, summary = analyze_snatch_reps(biomechanics)

    assert len(reps) == 4
    assert summary["detected_reps"] == 4
    assert [rep["catch_frame"] for rep in reps] == [51, 126, 201, 276]


def test_clean_reps_detects_multiple_front_rack_catches():
    biomechanics = []
    catch_centers = [110, 270, 430]

    for frame in range(520):
        front_rack = any(center <= frame <= center + 7 for center in catch_centers)
        in_pull = any(center - 35 <= frame < center for center in catch_centers)
        catch_bottom = front_rack and any(frame == center + 2 for center in catch_centers)

        biomechanics.append({
            "frame_number": frame,
            "hip_y": 0.72 if catch_bottom else (0.42 if front_rack else 0.50),
            "wrist_y": 0.53 if front_rack else 0.86,
            "shoulder_y": 0.55,
            "wrist_x": 0.50,
            "shoulder_x": 0.50,
            "hip_angle": 178.0 if in_pull else (105.0 if front_rack else 160.0),
            "knee_angle": 176.0 if in_pull else (118.0 if front_rack else 165.0),
        })

    reps, summary = analyze_clean_reps(biomechanics)

    assert len(reps) == 3
    assert summary["detected_reps"] == 3
    assert [rep["catch_frame"] for rep in reps] == [115, 275, 435]
    assert looks_like_clean_only(biomechanics) is True


def test_split_jerk_reps_detects_multiple_overhead_catches():
    biomechanics = []
    catch_centers = [60, 140, 220, 300]

    for frame in range(360):
        overhead = any(center <= frame <= center + 8 for center in catch_centers)
        dip = any(center - 18 <= frame <= center - 12 for center in catch_centers)
        drive = any(center - 11 <= frame < center for center in catch_centers)

        biomechanics.append({
            "frame_number": frame,
            "knee_angle": 145.0 if dip else (178.0 if drive else 168.0),
            "hip_angle": 150.0 if dip else (178.0 if drive else 166.0),
            "torso_angle": 5.0,
            "elbow_angle": 170.0 if overhead else 120.0,
            "wrist_y": 0.30 if overhead else 0.78,
            "shoulder_y": 0.55,
            "wrist_x": 0.50,
            "shoulder_x": 0.50,
            "valgus_ratio": 1.0,
        })

    reps, summary = analyze_split_jerk_reps(biomechanics)

    assert len(reps) == 4
    assert summary["detected_reps"] == 4
    assert [rep["catch_frame"] for rep in reps] == [60, 140, 220, 300]
    assert looks_like_split_jerk(biomechanics) is True
