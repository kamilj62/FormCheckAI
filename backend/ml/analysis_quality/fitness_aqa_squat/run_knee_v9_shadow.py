import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from ml.analysis_quality.fitness_aqa_squat import (
    audit_knee_v9_raw_pose_reps as rep_detector,
)

detect_chunk_reps = rep_detector.detect_chunk_reps
split_on_frame_gaps = rep_detector.split_on_frame_gaps
from ml.analysis_quality.fitness_aqa_squat.build_knee_v9_rep_features import (
    build_feature_names,
    build_vector,
    geometry,
    phase_windows,
)


DEFAULT_MODEL = (
    Path("ml/analysis_quality/fitness_aqa_squat/models")
    / "knee_complete_rep_v9.joblib"
)

CONSERVATIVE_FORWARD_THRESHOLD = 0.52


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run V9 knee-fault models in shadow mode "
            "on an extracted raw-pose JSONL file."
        )
    )

    parser.add_argument(
        "--pose-jsonl",
        required=True,
        help="Raw pose JSONL created by extract_knee_pose.py",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )

    parser.add_argument(
        "--shadow-min-phase-rows",
        type=int,
        default=4,
        help=(
            "Shadow-only detector override. "
            "Training used 4; use 2 only for fast real clips."
        ),
    )

    return parser.parse_args()


def load_pose_rows(path):
    rows = []

    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            rows.append(row)

    rows.sort(
        key=lambda row: int(row["frame_number"])
    )

    return rows


def main():
    args = parse_args()

    pose_path = Path(args.pose_jsonl)
    model_path = Path(args.model)

    if not pose_path.exists():
        raise SystemExit(
            f"Pose JSONL not found: {pose_path}"
        )

    if not model_path.exists():
        raise SystemExit(
            f"V9 model not found: {model_path}"
        )

    rows = load_pose_rows(pose_path)

    # This changes only the imported detector module in this
    # standalone shadow process. It does not modify training,
    # saved datasets, production analysis, or source constants.
    rep_detector.MIN_PHASE_ROWS = int(
        args.shadow_min_phase_rows
    )

    if not rows:
        raise SystemExit(
            f"No pose rows found in: {pose_path}"
        )

    video_ids = {
        str(row["video_id"])
        for row in rows
    }

    if len(video_ids) != 1:
        raise SystemExit(
            "Shadow runner currently expects one video. "
            f"Found video IDs: {sorted(video_ids)}"
        )

    video_id = next(iter(video_ids))
    video_path = str(
        rows[0].get("video_path", "")
    )

    bundle = joblib.load(model_path)

    required_keys = {
        "forward_model",
        "inward_model",
        "feature_names",
    }

    missing_keys = required_keys - set(bundle)

    if missing_keys:
        raise RuntimeError(
            "Model bundle is missing keys: "
            + ", ".join(sorted(missing_keys))
        )

    generated_feature_names = build_feature_names()
    model_feature_names = list(
        bundle["feature_names"]
    )

    if generated_feature_names != model_feature_names:
        raise RuntimeError(
            "V9 feature ordering does not match "
            "the saved model bundle"
        )

    chunks = split_on_frame_gaps(rows)

    results = []
    rejection_totals = {}

    global_rep_index = 0

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):
        detected_reps, rejections = detect_chunk_reps(
            chunk,
            None,
        )

        for reason, count in rejections.items():
            rejection_totals[reason] = (
                rejection_totals.get(reason, 0)
                + int(count)
            )

        for detected in detected_reps:
            start_index = int(
                detected["start_index"]
            )
            bottom_index = int(
                detected["bottom_index"]
            )
            end_index = int(
                detected["end_index"]
            )

            rep_rows = chunk[
                start_index:end_index + 1
            ]

            bottom_position = (
                bottom_index - start_index
            )

            geometry_rows = [
                geometry(row)
                for row in rep_rows
            ]

            windows = phase_windows(
                rep_rows,
                bottom_position,
            )

            vector = build_vector(
                rep_rows,
                windows,
                geometry_rows,
                bottom_position,
            )

            if vector.shape != (
                len(model_feature_names),
            ):
                raise RuntimeError(
                    "Unexpected V9 feature-vector shape: "
                    f"{vector.shape}"
                )

            if not np.all(np.isfinite(vector)):
                raise RuntimeError(
                    "Non-finite V9 feature value detected"
                )

            X = vector.reshape(1, -1)

            forward_score = float(
                np.clip(
                    bundle["forward_model"].predict(X)[0],
                    0.0,
                    1.0,
                )
            )

            inward_score = float(
                np.clip(
                    bundle["inward_model"].predict(X)[0],
                    0.0,
                    1.0,
                )
            )

            global_rep_index += 1

            results.append({
                "rep": global_rep_index,
                "chunk": chunk_index,
                "start_frame": int(
                    detected["start_frame"]
                ),
                "bottom_frame": int(
                    detected["bottom_frame"]
                ),
                "end_frame": int(
                    detected["end_frame"]
                ),
                "sampled_rows": int(
                    detected["sampled_rows"]
                ),
                "descent_rows": int(
                    detected["descent_rows"]
                ),
                "ascent_rows": int(
                    detected["ascent_rows"]
                ),
                "bottom_knee": float(
                    detected["bottom_knee"]
                ),
                "forward_score": forward_score,
                "forward_shadow_warning": bool(
                    forward_score
                    >= CONSERVATIVE_FORWARD_THRESHOLD
                ),
                # Diagnostic only. The inward model was rejected.
                "inward_score_diagnostic": inward_score,
            })

    print("=" * 76)
    print("V9 SHADOW RESULT")
    print("=" * 76)
    print("video_id:", video_id)
    print("video_path:", video_path)
    print("pose rows:", len(rows))
    print("chunks:", len(chunks))
    print("detected reps:", len(results))
    print(
        "forward shadow threshold:",
        CONSERVATIVE_FORWARD_THRESHOLD,
    )
    print(
        "shadow minimum phase rows:",
        rep_detector.MIN_PHASE_ROWS,
    )

    if not results:
        print("\nNo complete repetitions detected.")

        if rejection_totals:
            print("\nDetector rejections:")

            for reason, count in sorted(
                rejection_totals.items(),
                key=lambda item: (-item[1], item[0]),
            ):
                print(f"  {reason}: {count}")
    else:
        for result in results:
            warning = (
                "WARNING"
                if result["forward_shadow_warning"]
                else "clear"
            )

            print("\n" + "-" * 76)
            print("rep:", result["rep"])
            print(
                "frames:",
                (
                    f'{result["start_frame"]} -> '
                    f'{result["bottom_frame"]} -> '
                    f'{result["end_frame"]}'
                ),
            )
            print(
                "rows:",
                (
                    f'{result["sampled_rows"]} '
                    f'(descent={result["descent_rows"]}, '
                    f'ascent={result["ascent_rows"]})'
                ),
            )
            print(
                "bottom knee:",
                round(result["bottom_knee"], 2),
            )
            print(
                "forward score:",
                round(result["forward_score"], 4),
            )
            print(
                "forward shadow decision:",
                warning,
            )
            print(
                "inward score, diagnostic only:",
                round(
                    result[
                        "inward_score_diagnostic"
                    ],
                    4,
                ),
            )

    summary = {
        "version": "v9_shadow",
        "video_id": video_id,
        "video_path": video_path,
        "pose_jsonl": str(pose_path),
        "model": str(model_path),
        "pose_rows": len(rows),
        "chunks": len(chunks),
        "detected_reps": len(results),
        "forward_shadow_threshold": (
            CONSERVATIVE_FORWARD_THRESHOLD
        ),
        "rejection_counts": rejection_totals,
        "reps": results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(summary, indent=2)
        )

        print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
