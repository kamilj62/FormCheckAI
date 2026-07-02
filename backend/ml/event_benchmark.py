from pathlib import Path
from app.main import extract_video_biomechanics
from app.movement.event_detector import detect_movement_events

BENCHMARKS = [
    {
        "label": "snatch",
        "folder": "/Users/josephkamil/Desktop/Capstone/Oly_Data/raw/snatch_mp4",
        "pattern": "*.mp4",
        "limit": 20,
        "order": ["setup", "first_pull", "extension", "catch", "finish"],
    },
    {
        "label": "clean_and_jerk",
        "folder": "/Users/josephkamil/Desktop/Capstone/Oly_Data/raw/clean_and_jerk",
        "pattern": "*.avi",
        "limit": 20,
        "order": ["setup", "clean_extension", "clean_catch", "clean_recovery", "jerk_dip", "jerk_drive", "jerk_catch", "finish"],
    },
]

def check_order(events, keys):
    missing = [k for k in keys if k not in events]
    if missing:
        return False, f"missing {missing}"

    vals = [events[k] for k in keys]
    for a, b, ka, kb in zip(vals, vals[1:], keys, keys[1:]):
        if not a < b:
            return False, f"{ka}={a} not < {kb}={b}"
    return True, "ok"

def main():
    total = 0
    passed = 0

    for bench in BENCHMARKS:
        files = sorted(Path(bench["folder"]).glob(bench["pattern"]))[:bench["limit"]]

        print(f"\n=== {bench['label']} ===")

        for f in files:
            _, biomechanics, debug = extract_video_biomechanics(str(f))
            events = detect_movement_events(biomechanics, bench["label"])
            ok, reason = check_order(events, bench["order"])

            total += 1
            passed += int(ok)

            status = "PASS" if ok else "FAIL"
            print(f"{status:4s} {f.name} {reason} {events}")

    print(f"\nOverall event ordering: {passed}/{total} ({100*passed/total:.1f}%)")

if __name__ == "__main__":
    main()
