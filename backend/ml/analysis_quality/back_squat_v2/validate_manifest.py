#!/usr/bin/env python3

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"

ALLOWED_FAULTS = {
    "shallow_depth",
    "torso_collapse",
    "knee_valgus",
    "heel_lift",
    "neck_position",
}

ALLOWED_SEVERITIES = {
    "none",
    "mild",
    "moderate",
    "severe",
}

ALLOWED_CAMERA_ANGLES = {
    "front",
    "front_45",
    "side",
    "rear_45",
    "rear",
    "unknown",
}


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def main() -> None:
    if not MANIFEST_PATH.exists():
        fail(f"Manifest not found: {MANIFEST_PATH}")

    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON: {exc}")

    if data.get("schema_version") != 1:
        fail("schema_version must be 1")

    if data.get("exercise") != "squat_back":
        fail("exercise must be squat_back")

    videos = data.get("videos")
    if not isinstance(videos, list):
        fail("videos must be a list")

    seen_ids: set[str] = set()

    for index, video in enumerate(videos):
        prefix = f"videos[{index}]"

        if not isinstance(video, dict):
            fail(f"{prefix} must be an object")

        video_id = video.get("id")
        if not isinstance(video_id, str) or not video_id.strip():
            fail(f"{prefix}.id must be a nonempty string")

        if video_id in seen_ids:
            fail(f"duplicate video id: {video_id}")
        seen_ids.add(video_id)

        path = video.get("path")
        if not isinstance(path, str) or not path.strip():
            fail(f"{prefix}.path must be a nonempty string")

        camera_angle = video.get("camera_angle", "unknown")
        if camera_angle not in ALLOWED_CAMERA_ANGLES:
            fail(
                f"{prefix}.camera_angle must be one of "
                f"{sorted(ALLOWED_CAMERA_ANGLES)}"
            )

        expected_reps = video.get("expected_reps")
        if not isinstance(expected_reps, int) or expected_reps < 0:
            fail(f"{prefix}.expected_reps must be a nonnegative integer")

        reps = video.get("reps")
        if not isinstance(reps, list):
            fail(f"{prefix}.reps must be a list")

        if len(reps) != expected_reps:
            fail(
                f"{prefix}.reps contains {len(reps)} entries, "
                f"but expected_reps is {expected_reps}"
            )

        for rep_index, rep in enumerate(reps, start=1):
            rep_prefix = f"{prefix}.reps[{rep_index - 1}]"

            if rep.get("rep") != rep_index:
                fail(f"{rep_prefix}.rep must equal {rep_index}")

            faults = rep.get("faults")
            if not isinstance(faults, dict):
                fail(f"{rep_prefix}.faults must be an object")

            unknown_faults = set(faults) - ALLOWED_FAULTS
            if unknown_faults:
                fail(
                    f"{rep_prefix} contains unsupported faults: "
                    f"{sorted(unknown_faults)}"
                )

            for fault_name in ALLOWED_FAULTS:
                severity = faults.get(fault_name, "none")
                if severity not in ALLOWED_SEVERITIES:
                    fail(
                        f"{rep_prefix}.faults.{fault_name} must be one of "
                        f"{sorted(ALLOWED_SEVERITIES)}"
                    )

    print("Manifest validation passed")
    print(f"Videos: {len(videos)}")
    print(f"Annotated reps: {sum(len(v['reps']) for v in videos)}")


if __name__ == "__main__":
    main()
