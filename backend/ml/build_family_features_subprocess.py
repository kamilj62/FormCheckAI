import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from app.feature_engine.feature_names_v2 import FEATURE_NAMES


META_FIELDS = [
    "family",
    "exercise_label",
    "video",
    "name",
    "source_group",
]

OUTPUT_FIELDS = META_FIELDS + FEATURE_NAMES


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


WORKER = r'''
import json
import sys

from app.main import extract_video_biomechanics
from app.feature_engine.movement_video_features_v2 import (
    build_movement_video_features,
)
from app.feature_engine.feature_names_v2 import FEATURE_NAMES

video = sys.argv[1]
out_path = sys.argv[2]

sequence, biomechanics, debug = extract_video_biomechanics(
    video,
    sample_every=1,
)

result = {
    "ok": False,
    "reason": None,
    "features": None,
    "frames_processed": debug.get("frames_processed"),
    "pose_frames": debug.get("pose_frames"),
}

if len(sequence) < 10 or len(biomechanics) < 10:
    result["reason"] = "insufficient_data"
else:
    features = build_movement_video_features(biomechanics)

    result["ok"] = True
    result["features"] = {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, features)
    }

with open(out_path, "w") as f:
    json.dump(result, f)
'''


def process_video(row, timeout_seconds):
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False,
    )
    tmp_path = tmp.name
    tmp.close()

    env = os.environ.copy()

    backend_root = str(Path.cwd())
    existing_pythonpath = env.get("PYTHONPATH", "")

    env["PYTHONPATH"] = (
        backend_root
        if not existing_pythonpath
        else backend_root + os.pathsep + existing_pythonpath
    )

    cmd = [
        sys.executable,
        "-c",
        WORKER,
        row["video"],
        tmp_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=timeout_seconds,
        )

        if proc.returncode != 0:
            return {
                "ok": False,
                "reason": f"worker_exit_{proc.returncode}",
            }

        if not Path(tmp_path).exists():
            return {
                "ok": False,
                "reason": "worker_no_result",
            }

        with open(tmp_path) as f:
            return json.load(f)

    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "reason": f"timeout_{timeout_seconds}s",
        }

    except Exception as exc:
        return {
            "ok": False,
            "reason": str(exc),
        }

    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


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

    completed = load_completed(
        output_path,
        failure_path,
    )

    pending = [
        row for row in rows
        if row["video"] not in completed
    ]

    print("Manifest:", manifest_path)
    print("Total:", len(rows))
    print("Already completed:", len(completed))
    print("Pending:", len(pending))
    print("Hard timeout:", args.timeout, "seconds")
    print()

    success_count = 0
    failure_count = 0

    for i, row in enumerate(pending, 1):
        print(
            f"[{i}/{len(pending)}] "
            f"{row['family']} :: {row['name']}",
            flush=True,
        )

        result = process_video(
            row,
            args.timeout,
        )

        if result.get("ok"):
            out_row = {
                key: row[key]
                for key in META_FIELDS
            }

            out_row.update(result["features"])

            append_row(
                output_path,
                OUTPUT_FIELDS,
                out_row,
            )

            success_count += 1
            print("  -> saved", flush=True)

        else:
            failure_row = {
                **row,
                "reason": result.get(
                    "reason",
                    "unknown_failure",
                ),
                "frames_processed": result.get(
                    "frames_processed",
                    "",
                ),
                "pose_frames": result.get(
                    "pose_frames",
                    "",
                ),
            }

            append_row(
                failure_path,
                META_FIELDS + [
                    "reason",
                    "frames_processed",
                    "pose_frames",
                ],
                failure_row,
            )

            failure_count += 1

            print(
                f"  -> skipped: {failure_row['reason']}",
                flush=True,
            )

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
