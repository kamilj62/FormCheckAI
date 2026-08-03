import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

MAX_GAP_FRAMES = 30
MIN_CHUNK_ROWS = 9
MIN_PHASE_ROWS = 4
MIN_SOURCE_SPAN = 20
MIN_KNEE_RANGE = 20.0
UPRIGHT_KNEE_MIN = 150.0
BOTTOM_KNEE_MAX = 145.0
MIN_HIP_DROP = 0.035


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
    )

    return parser.parse_args()


def split_on_frame_gaps(rows):
    rows = sorted(
        rows,
        key=lambda row: int(row["frame_number"]),
    )

    chunks = []
    current = []

    for row in rows:
        if not current:
            current = [row]
            continue

        gap = (
            int(row["frame_number"])
            - int(current[-1]["frame_number"])
        )

        if gap <= MAX_GAP_FRAMES:
            current.append(row)
        else:
            if len(current) >= MIN_CHUNK_ROWS:
                chunks.append(current)

            current = [row]

    if len(current) >= MIN_CHUNK_ROWS:
        chunks.append(current)

    return chunks


def smooth(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) < 3:
        return values.copy()

    output = values.copy()

    for index in range(1, len(values) - 1):
        output[index] = np.median(
            values[index - 1:index + 2]
        )

    return output


def find_local_minima(values):
    minima = []

    for index in range(1, len(values) - 1):
        if (
            values[index] <= values[index - 1]
            and values[index] < values[index + 1]
        ):
            minima.append(index)

    return minima


def state_name(forward, inward):
    if not forward and not inward:
        return "neither"
    if forward and not inward:
        return "forward_only"
    if not forward and inward:
        return "inward_only"
    return "both"


def detect_chunk_reps(
    chunk,
    name_to_index,
):
    frames = np.asarray(
        [int(row["frame_number"]) for row in chunk],
        dtype=np.int64,
    )

    knee = np.asarray(
        [
            float(
                row["biomechanics"].get(
                    "knee_angle",
                    180.0,
                )
                or 180.0
            )
            for row in chunk
        ],
        dtype=np.float64,
    )

    hip_y = np.asarray(
        [
            float(
                row["biomechanics"].get(
                    "hip_y",
                    0.5,
                )
                or 0.5
            )
            for row in chunk
        ],
        dtype=np.float64,
    )

    knee_smoothed = smooth(knee)

    if (
        float(np.max(knee_smoothed) - np.min(knee_smoothed))
        < MIN_KNEE_RANGE
    ):
        return [], Counter({"knee_range_too_small": 1})

    minima = find_local_minima(knee_smoothed)

    candidates = []
    rejections = Counter()
    last_end = -1

    for bottom in minima:
        if bottom <= last_end:
            continue

        bottom_knee = float(knee_smoothed[bottom])

        if bottom_knee > BOTTOM_KNEE_MAX:
            rejections["bottom_not_deep_enough"] += 1
            continue

        setup_candidates = [
            index
            for index in range(0, bottom)
            if float(knee_smoothed[index]) >= UPRIGHT_KNEE_MIN
        ]

        finish_candidates = [
            index
            for index in range(bottom + 1, len(chunk))
            if float(knee_smoothed[index]) >= UPRIGHT_KNEE_MIN
        ]

        if not setup_candidates:
            rejections["missing_setup"] += 1
            continue

        if not finish_candidates:
            rejections["missing_finish"] += 1
            continue

        start = setup_candidates[-1]
        end = finish_candidates[0]

        descent_rows = bottom - start
        ascent_rows = end - bottom
        source_span = int(frames[end] - frames[start])

        if descent_rows < MIN_PHASE_ROWS:
            rejections["descent_too_short"] += 1
            continue

        if ascent_rows < MIN_PHASE_ROWS:
            rejections["ascent_too_short"] += 1
            continue

        if source_span < MIN_SOURCE_SPAN:
            rejections["source_span_too_short"] += 1
            continue

        setup_hip = float(hip_y[start])
        bottom_hip = float(hip_y[bottom])
        finish_hip = float(hip_y[end])

        hip_drop_from_setup = bottom_hip - setup_hip
        hip_drop_from_finish = bottom_hip - finish_hip

        if (
            hip_drop_from_setup < MIN_HIP_DROP
            or hip_drop_from_finish < MIN_HIP_DROP
        ):
            rejections["insufficient_hip_drop"] += 1
            continue

        rep_rows = chunk[start:end + 1]

        forward_labels = np.asarray(
            [
                int(row["labels"]["knees_forward"])
                for row in rep_rows
            ],
            dtype=np.float64,
        )

        inward_labels = np.asarray(
            [
                int(row["labels"]["knees_inward"])
                for row in rep_rows
            ],
            dtype=np.float64,
        )

        forward_fraction = float(
            np.mean(forward_labels)
        )
        inward_fraction = float(
            np.mean(inward_labels)
        )

        candidates.append({
            "start_index": start,
            "bottom_index": bottom,
            "end_index": end,
            "start_frame": int(frames[start]),
            "bottom_frame": int(frames[bottom]),
            "end_frame": int(frames[end]),
            "sampled_rows": int(end - start + 1),
            "descent_rows": int(descent_rows),
            "ascent_rows": int(ascent_rows),
            "source_span": source_span,
            "setup_knee": float(knee[start]),
            "bottom_knee": float(knee[bottom]),
            "finish_knee": float(knee[end]),
            "hip_drop_from_setup": hip_drop_from_setup,
            "hip_drop_from_finish": hip_drop_from_finish,
            "forward_fraction": forward_fraction,
            "inward_fraction": inward_fraction,
            "forward_majority": int(
                forward_fraction >= 0.5
            ),
            "inward_majority": int(
                inward_fraction >= 0.5
            ),
        })

        last_end = end

    return candidates, rejections


def summarize(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return {}

    return {
        "count": int(len(values)),
        "min": round(float(np.min(values)), 3),
        "p10": round(float(np.percentile(values, 10)), 3),
        "median": round(float(np.median(values)), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
        "max": round(float(np.max(values)), 3),
    }


def main():
    args = parse_args()

    data_path = BASE / f"knee_pose_{args.split}.jsonl"

    # Raw pose rows retain upright setup and lockout frames.
    # Geometry is read directly from each row's biomechanics dictionary.
    name_to_index = None

    rows_by_video = defaultdict(list)

    with data_path.open() as file:
        for line in file:
            row = json.loads(line)
            rows_by_video[
                str(row["video_id"])
            ].append(row)

    all_reps = []
    rejection_counts = Counter()
    videos_with_reps = 0
    chunks_examined = 0
    chunks_with_reps = 0

    for video_id, rows in rows_by_video.items():
        video_rep_count = 0

        for chunk_index, chunk in enumerate(
            split_on_frame_gaps(rows)
        ):
            chunks_examined += 1

            reps, rejections = detect_chunk_reps(
                chunk,
                name_to_index,
            )

            rejection_counts.update(rejections)

            if reps:
                chunks_with_reps += 1

            for rep_index, rep in enumerate(reps):
                all_reps.append({
                    "video_id": video_id,
                    "chunk_index": chunk_index,
                    "rep_index": rep_index,
                    **rep,
                })

                video_rep_count += 1

        if video_rep_count > 0:
            videos_with_reps += 1

    state_counts = Counter()

    for rep in all_reps:
        state_counts[
            state_name(
                bool(rep["forward_majority"]),
                bool(rep["inward_majority"]),
            )
        ] += 1

    print("=" * 76)
    print(args.split.upper())
    print("=" * 76)

    print("videos:", len(rows_by_video))
    print("chunks examined:", chunks_examined)
    print("chunks with reps:", chunks_with_reps)
    print("videos with reps:", videos_with_reps)
    print("detected reps:", len(all_reps))
    print("rep states:", dict(state_counts))

    print(
        "sampled rows per rep:",
        summarize([
            rep["sampled_rows"]
            for rep in all_reps
        ]),
    )

    print(
        "source span per rep:",
        summarize([
            rep["source_span"]
            for rep in all_reps
        ]),
    )

    print(
        "descent rows:",
        summarize([
            rep["descent_rows"]
            for rep in all_reps
        ]),
    )

    print(
        "ascent rows:",
        summarize([
            rep["ascent_rows"]
            for rep in all_reps
        ]),
    )

    print(
        "bottom knee angle:",
        summarize([
            rep["bottom_knee"]
            for rep in all_reps
        ]),
    )

    print(
        "forward fractions:",
        summarize([
            rep["forward_fraction"]
            for rep in all_reps
        ]),
    )

    print(
        "inward fractions:",
        summarize([
            rep["inward_fraction"]
            for rep in all_reps
        ]),
    )

    print("\nRejections:")

    for reason, count in rejection_counts.most_common():
        print(reason, count)

    output_path = (
        BASE
        / f"knee_v9_raw_pose_rep_audit_{args.split}.json"
    )

    output_path.write_text(
        json.dumps(
            {
                "version": "v9_raw_pose_rep_detection_audit",
                "split": args.split,
                "settings": {
                    "max_gap_frames": MAX_GAP_FRAMES,
                    "minimum_chunk_rows": MIN_CHUNK_ROWS,
                    "minimum_phase_rows": MIN_PHASE_ROWS,
                    "minimum_source_span": MIN_SOURCE_SPAN,
                    "minimum_knee_range": MIN_KNEE_RANGE,
                    "upright_knee_minimum": UPRIGHT_KNEE_MIN,
                    "bottom_knee_maximum": BOTTOM_KNEE_MAX,
                    "minimum_hip_drop": MIN_HIP_DROP,
                },
                "videos": len(rows_by_video),
                "chunks_examined": chunks_examined,
                "chunks_with_reps": chunks_with_reps,
                "videos_with_reps": videos_with_reps,
                "detected_reps": len(all_reps),
                "state_counts": dict(state_counts),
                "rejection_counts": dict(
                    rejection_counts
                ),
                "reps": all_reps,
            },
            indent=2,
        )
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
