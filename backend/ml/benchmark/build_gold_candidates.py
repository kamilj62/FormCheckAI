import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "Desktop" / "Capstone"
OUT = Path("ml/benchmark/config/gold_candidates.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

LABEL_HINTS = {
    "bench_press": ["benchpress", "bench_press", "bench press", "bench"],
    "squat_back": ["squat_back", "backsquat", "back squat", "back squats"],
    "squat_front": ["squat_front", "frontsquat", "front squat", "front squats"],
    "overhead_squat": ["overhead_squat", "overheadsquat", "overhead squat"],
    "deadlift": ["deadlift"],
    "push_press": ["push_press", "pushpress", "push press"],
    "strict_press": ["strict_press", "strictpress", "strict press"],
    "clean": ["real_clean", "/clean/", "clean-correct"],
    "clean_and_jerk": ["clean_and_jerk", "cleanandjerk", "clean and jerk"],
    "split_jerk": ["split_jerk", "splitjerk", "split jerk"],
    "snatch": ["snatch"],
    "push_up": ["push_up", "pushup", "pushups", "/push-up/", "/pushups/"],
    "pull_up": ["pull_up", "pullup", "pull up", "pullups"],
    "handstand_push_up": ["handstand_push_up", "handstandpushup", "handstand"],
    "thruster": ["thruster"],
}

rows = []

for label, hints in LABEL_HINTS.items():
    found = []
    for p in ROOT.rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        if "_Removed_Datasets" in p.parts or "_Duplicate_Backups" in p.parts:
            continue
        if "FormCheck_Phase_Audit" in p.parts:
            continue

        text = str(p).lower()
        compact = text.replace("_", "").replace("-", "").replace(" ", "")

        matched = False
        for h in hints:
            h2 = h.lower()
            if h2 in text or h2.replace("_", "").replace("-", "").replace(" ", "") in compact:
                matched = True
                break

        # Avoid cross-label contamination.
        if label == "push_up" and "handstand" in text:
            matched = False

        if matched:
            found.append(p)

    # Dedupe by filename + size so copied files do not appear twice.
    seen = set()
    unique = []
    for p in sorted(found, key=lambda x: (len(str(x)), str(x))):
        key = (p.name.lower(), p.stat().st_size)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    found = unique[:30]

    for p in found:
        rows.append({
            "label": label,
            "video": str(p),
            "name": p.name,
            "size": p.stat().st_size,
            "selected": "",
            "expected_reps": "",
            "notes": "",
        })

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["label", "video", "name", "size", "selected", "expected_reps", "notes"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUT}")
print(f"Rows: {len(rows)}")
