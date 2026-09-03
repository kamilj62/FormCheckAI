"""Generate a Build 5 versus candidate squat Setup contact sheet."""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app import main as app_main  # noqa: E402


def build5_setup_frame(records, start, fallback):
    before = [row for row in records if int(row["frame"]) <= int(start)]
    candidates = [
        row
        for row in before
        if float(row["knee"]) >= 165 and float(row["hip"]) >= 150
    ]

    if not candidates:
        candidates = [
            row
            for row in before
            if float(row["knee"]) >= 155 and float(row["hip"]) >= 130
        ]

    if not candidates:
        return int(start if fallback is None else fallback)

    return int(
        max(
            candidates,
            key=lambda row: (
                float(row["knee"]) + float(row["hip"]),
                int(start) - int(row["frame"]),
            ),
        )["frame"]
    )


def read_frame(video_path, frame_number):
    capture = cv2.VideoCapture(str(video_path))
    frame = None
    ok = False
    for index in range(int(frame_number) + 1):
        ok, decoded = capture.read()
        if not ok:
            break
        if index == int(frame_number):
            frame = decoded
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {frame_number}")
    return frame


def labeled_tile(frame, label, frame_number):
    tile = cv2.resize(frame, (480, 270))
    cv2.rectangle(tile, (0, 0), (480, 45), (0, 0, 0), -1)
    cv2.putText(
        tile,
        f"{label} - frame {frame_number}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, nargs="+")
    parser.add_argument(
        "--output",
        type=Path,
        default=BACKEND_DIR / "work" / "squat_setup_comparison.jpg",
    )
    args = parser.parse_args()

    if len(args.video) > 1:
        for video_path in args.video:
            output_path = (
                BACKEND_DIR
                / "work"
                / f"{video_path.stem}_squat_setup_comparison.jpg"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(video_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
            )
        return

    video_path = args.video[0]
    exercise_label = (
        "squat_front" if "front" in video_path.stem.lower() else "squat_back"
    )

    result = app_main.analyze_video(
        str(video_path),
        make_visuals=False,
        make_overlay=False,
        forced_exercise_label=exercise_label,
    )
    reps = result.get("rep_feedback") or []
    rep = app_main.choose_phase_rep(reps)
    if not rep:
        raise RuntimeError(
            f"Analyzer did not return a usable squat rep: {result.get('analysis_mode')}"
        )

    start = int(rep.get("start_frame", rep.get("start", 0)))
    bottom = int(rep.get("bottom_frame", rep.get("bottom", start + 1)))
    end = int(rep.get("end_frame", rep.get("end", bottom + 1)))

    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    stable_selector = app_main.choose_stable_squat_setup_frame
    candidate_records = []

    def diagnostic_selector(records, start, fallback):
        candidate_records.extend(records)
        return stable_selector(records, start=start, fallback=fallback)

    try:
        app_main.choose_stable_squat_setup_frame = diagnostic_selector
        candidate = app_main.pick_squat_visual_frames_from_video(
            str(video_path), start, bottom, end, total_frames
        )

        app_main.choose_stable_squat_setup_frame = build5_setup_frame
        build5 = app_main.pick_squat_visual_frames_from_video(
            str(video_path), start, bottom, end, total_frames
        )
    finally:
        app_main.choose_stable_squat_setup_frame = stable_selector

    lockout_matches = candidate["lockout"] == build5["lockout"]
    tiles = [
        labeled_tile(
            read_frame(video_path, build5["setup"]),
            "BUILD 5 SETUP",
            build5["setup"],
        ),
        labeled_tile(
            read_frame(video_path, candidate["setup"]),
            "CANDIDATE SETUP",
            candidate["setup"],
        ),
        labeled_tile(
            read_frame(video_path, candidate["lockout"]),
            "UNCHANGED LOCKOUT",
            candidate["lockout"],
        ),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.hconcat(tiles)):
        raise RuntimeError(f"Could not write {args.output}")

    print(f"Rep anchors: start={start}, bottom={bottom}, end={end}")
    print(f"Build 5 Setup: {build5['setup']}")
    print(f"Candidate Setup: {candidate['setup']}")
    nearby = [
        row
        for row in candidate_records
        if candidate["setup"] - 6 <= int(row["frame"]) <= candidate["setup"] + 6
    ]
    print("Candidate pose measurements:")
    for row in nearby:
        print(
            "  frame={frame} knee={knee:.1f} hip={hip:.1f} "
            "torso_lean={torso_lean:.1f}".format(**row)
        )
    print(f"Lockout: {candidate['lockout']} (unchanged={lockout_matches})")
    print(f"Contact sheet: {args.output.resolve()}")


if __name__ == "__main__":
    main()
