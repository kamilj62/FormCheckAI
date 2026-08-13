import argparse
import csv
import gc
import signal
from pathlib import Path

from app.main import extract_video_biomechanics
from app.feature_engine.movement_video_features_v2 import (
    build_movement_video_features,
)
from app.feature_engine.feature_names_v2 import FEATURE_NAMES


META_FIELDS = [
    "family",
    "exercise_label",
    "video",
    "name",
    "source_group",
]

OUTPUT_FIELDS = META_FIELDS + FEATURE_NAMES


class VideoTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise VideoTimeout("video_timeout")


def load_completed(output_path, failure_path):
    completed = set()

    for path in (output_path, failure_path):
        if not path.exists():
            continue

        try:
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("video"):
                        completed.add(row["video"])
        except Exception:
            pass

    return completed


def append_row(path, fieldnames, row):
    exists = path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)
        f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    failure_path = output_path.with_name(
        output_path.stem + "_failures.csv"
    )

    with manifest_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    completed = load_completed(output_path, failure_path)

    pending = [
        row for row in rows
        if row["video"] not in completed
    ]

    print("Manifest:", manifest_path)
    print("Total:", len(rows))
    print("Already completed:", len(completed))
    print("Pending:", len(pending))
    print("Timeout per video:", args.timeout, "seconds")
    print()

    success_count = 0
    failure_count = 0

    signal.signal(signal.SIGALRM, timeout_handler)

    for i, row in enumerate(pending, 1):
        print(
            f"[{i}/{len(pending)}] "
            f"{row['family']} :: {row['name']}",
            flush=True,
        )

        sequence = None
        biomechanics = None

        try:
            signal.alarm(args.timeout)

            sequence, biomechanics, debug = extract_video_biomechanics(
                row["video"],
                sample_every=1,
            )

            signal.alarm(0)

            if len(sequence) < 10 or len(biomechanics) < 10:
                append_row(
                    failure_path,
                    META_FIELDS + [
                        "reason",
                        "frames_processed",
                        "pose_frames",
                    ],
                    {
                        **row,
                        "reason": "insufficient_data",
                        "frames_processed": debug.get("frames_processed"),
                        "pose_frames": debug.get("pose_frames"),
                    },
                )

                failure_count += 1
                print("  -> insufficient_data", flush=True)

            else:
                features = build_movement_video_features(
                    biomechanics
                )

                result = {
                    key: row[key]
                    for key in META_FIELDS
                }

                for name, value in zip(
                    FEATURE_NAMES,
                    features,
                ):
                    result[name] = float(value)

                append_row(
                    output_path,
                    OUTPUT_FIELDS,
                    result,
                )

                success_count += 1
                print("  -> saved", flush=True)

        except VideoTimeout:
            signal.alarm(0)

            append_row(
                failure_path,
                META_FIELDS + ["reason"],
                {
                    **row,
                    "reason": f"timeout_{args.timeout}s",
                },
            )

            failure_count += 1
            print(
                f"  -> TIMEOUT after {args.timeout}s; skipped",
                flush=True,
            )

        except Exception as exc:
            signal.alarm(0)

            append_row(
                failure_path,
                META_FIELDS + ["reason"],
                {
                    **row,
                    "reason": str(exc),
                },
            )

            failure_count += 1
            print(f"  -> ERROR: {exc}", flush=True)

        finally:
            signal.alarm(0)

            if sequence is not None:
                del sequence

            if biomechanics is not None:
                del biomechanics

            gc.collect()

    final_completed = load_completed(
        output_path,
        failure_path,
    )

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print("Manifest rows:", len(rows))
    print("Completed:", len(final_completed))
    print("Successful this run:", success_count)
    print("Failures this run:", failure_count)
    print("Features:", output_path)
    print("Failures:", failure_path)


if __name__ == "__main__":
    main()
