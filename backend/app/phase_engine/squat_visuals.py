from statistics import median


def choose_stable_squat_setup_frame(records, start, fallback):
    """Choose the strongest reliably upright frame before squat descent."""
    rows = sorted(
        (row for row in records if int(row["frame"]) <= int(start)),
        key=lambda row: int(row["frame"]),
    )

    if len(rows) < 3:
        return None if fallback is None else int(fallback)

    frame_steps = [
        int(right["frame"]) - int(left["frame"])
        for left, right in zip(rows, rows[1:])
        if int(right["frame"]) > int(left["frame"])
    ]
    # The video scan uses a fixed cadence; larger differences indicate missed
    # pose detections and must not be treated as part of the same stable hold.
    expected_step = min(frame_steps) if frame_steps else 1
    max_step = max(1, expected_step * 2)

    def candidates(knee_min, hip_min, torso_lean_max):
        stable = []

        for index in range(1, len(rows) - 1):
            window = rows[index - 1:index + 2]
            center = rows[index]
            steps = [
                int(window[offset + 1]["frame"])
                - int(window[offset]["frame"])
                for offset in range(2)
            ]

            if any(step <= 0 or step > max_step for step in steps):
                continue

            knees = [float(row["knee"]) for row in window]
            hips = [float(row["hip"]) for row in window]
            torso_leans = [float(row.get("torso_lean", 0.0)) for row in window]

            if (
                float(center["knee"]) >= knee_min
                and float(center["hip"]) >= hip_min
                and float(center.get("torso_lean", 0.0)) <= torso_lean_max
                and median(knees) >= knee_min
                and median(hips) >= hip_min
                and median(torso_leans) <= torso_lean_max
                and max(knees) - min(knees) <= 14
                and max(hips) - min(hips) <= 18
                and max(torso_leans) - min(torso_leans) <= 12
            ):
                stable.append(center)

        return stable

    setup_candidates = candidates(
        knee_min=165,
        hip_min=150,
        torso_lean_max=25,
    )
    if not setup_candidates:
        setup_candidates = candidates(
            knee_min=155,
            hip_min=135,
            torso_lean_max=35,
        )

    if not setup_candidates:
        return None if fallback is None else int(fallback)

    best = max(
        setup_candidates,
        key=lambda row: (
            float(row["knee"])
            + float(row["hip"])
            - float(row.get("torso_lean", 0.0)),
            int(row["frame"]),
        ),
    )
    return int(best["frame"])
