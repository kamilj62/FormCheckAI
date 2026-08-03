import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

INTERVAL_TYPES = [
    "complete",
    "descent_only",
    "ascent_only",
    "short_or_ambiguous",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
    )

    return parser.parse_args()


def classify_interval(row):
    bottom_fraction = float(
        row["bottom_frame_fraction"]
    )
    frame_count = int(row["frame_count"])

    if (
        frame_count >= 5
        and 0.15 < bottom_fraction < 0.85
    ):
        return "complete"

    if bottom_fraction >= 0.85:
        return "descent_only"

    if bottom_fraction <= 0.15:
        return "ascent_only"

    return "short_or_ambiguous"


def feature_is_available(
    feature_name,
    interval_type,
):
    # Base interval fields are always available.
    if "__" not in feature_name:
        return True

    statistic = feature_name.split("__", 1)[1]

    if interval_type == "complete":
        return True

    if interval_type == "descent_only":
        unavailable = (
            statistic.startswith("ascent_")
            or statistic.startswith("finish_")
            or statistic == "bottom_to_finish"
            or statistic == "setup_to_finish"
            or statistic == "ascent_range"
        )

        return not unavailable

    if interval_type == "ascent_only":
        unavailable = (
            statistic.startswith("setup_")
            or statistic.startswith("descent_")
            or statistic == "setup_to_bottom"
            or statistic == "setup_to_finish"
            or statistic == "descent_range"
            or statistic == "peak_positive_from_setup"
            or statistic == "peak_negative_from_setup"
        )

        return not unavailable

    # Short/ambiguous intervals retain only phase-independent
    # bottom and reliability summaries.
    allowed = (
        statistic.startswith("bottom_")
        or statistic.endswith("_range")
    )

    return allowed


def main():
    args = parse_args()

    source_path = (
        BASE / f"knee_v8_phase_{args.split}.jsonl"
    )

    source_metadata_path = (
        BASE
        / f"knee_v8_phase_{args.split}_metadata.json"
    )

    output_path = (
        BASE / f"knee_v8_masked_{args.split}.jsonl"
    )

    output_metadata_path = (
        BASE
        / f"knee_v8_masked_{args.split}_metadata.json"
    )

    source_metadata = json.loads(
        source_metadata_path.read_text()
    )

    source_names = source_metadata["feature_names"]

    output_names = (
        list(source_names)
        + [
            f"interval_type__{name}"
            for name in INTERVAL_TYPES
        ]
        + [
            "phase_has_setup",
            "phase_has_descent",
            "phase_has_bottom",
            "phase_has_ascent",
            "phase_has_finish",
        ]
    )

    interval_counts = Counter()
    state_counts = Counter()
    written = 0

    with source_path.open() as source, output_path.open("w") as output:
        for line in source:
            row = json.loads(line)

            interval_type = classify_interval(row)
            interval_counts[interval_type] += 1

            source_vector = np.asarray(
                row["features"],
                dtype=np.float32,
            )

            if len(source_vector) != len(source_names):
                raise RuntimeError(
                    "Source feature vector mismatch"
                )

            masked_vector = source_vector.copy()

            for index, feature_name in enumerate(
                source_names
            ):
                if not feature_is_available(
                    feature_name,
                    interval_type,
                ):
                    masked_vector[index] = 0.0

            type_features = np.asarray(
                [
                    float(interval_type == name)
                    for name in INTERVAL_TYPES
                ],
                dtype=np.float32,
            )

            phase_flags = {
                "complete": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                "descent_only": [
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                "ascent_only": [
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                "short_or_ambiguous": [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
            }

            availability_features = np.asarray(
                phase_flags[interval_type],
                dtype=np.float32,
            )

            final_vector = np.concatenate([
                masked_vector,
                type_features,
                availability_features,
            ])

            if not np.all(np.isfinite(final_vector)):
                raise RuntimeError(
                    "Non-finite masked V8 feature"
                )

            forward = bool(
                row["targets"]["forward_majority"]
            )
            inward = bool(
                row["targets"]["inward_majority"]
            )

            if not forward and not inward:
                state = "neither"
            elif forward and not inward:
                state = "forward_only"
            elif not forward and inward:
                state = "inward_only"
            else:
                state = "both"

            state_counts[
                f"{interval_type}:{state}"
            ] += 1

            output_row = {
                **row,
                "interval_type": interval_type,
                "features": final_vector.tolist(),
            }

            output.write(
                json.dumps(output_row) + "\n"
            )

            written += 1

    metadata = {
        "version": "v8_phase_masked",
        "split": args.split,
        "source": str(source_path),
        "output": str(output_path),
        "source_feature_count": len(source_names),
        "feature_count": len(output_names),
        "feature_names": output_names,
        "intervals_written": written,
        "interval_type_counts": dict(
            interval_counts
        ),
        "interval_type_state_counts": dict(
            sorted(state_counts.items())
        ),
    }

    output_metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    print("Split:", args.split)
    print("Rows:", written)
    print(
        "Feature count:",
        len(output_names),
    )
    print(
        "Interval types:",
        dict(interval_counts),
    )

    print("\nStates by interval type:")

    for key, count in sorted(
        state_counts.items()
    ):
        print(f"{key}: {count}")

    print("\nOutput:", output_path)
    print("Metadata:", output_metadata_path)


if __name__ == "__main__":
    main()
