from pathlib import Path
from collections import defaultdict
import argparse
import csv
import shutil

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

ROOT = Path.home() / "Desktop" / "Capstone"
REPORT_DIR = Path("ml/reports/dataset_manager")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IGNORE_PARTS = {
    "_Duplicate_Backups",
    "_Removed_Datasets",
    "FormCheck_Phase_Audit",
    "__pycache__",
}

PROTECTED_ROOTS = [
    "data/dataset_v2/raw",
    "Oly_Data/raw",
    "Oly_Data/segmented",
    "PushUps",
    "PullUps",
    "HandstandPushups",
    "BenchPress",
    "deadlift",
    "bench press",
]

SCRATCH_ROOTS = [
    "squat_keep",
    "Capstone",
]

PROTECTED_NAMES = {
    "Push press- diptiming.mov",
    "Back squats- forward lean.mov",
    "Bench press- arch.mov",
    "Front squat- elbow drop.mov",
    "Front squats- bar drift.mov",
    "Front squats- butt wink .mov",
    "Bench press- elbow flare.mov",
    "OverheadSquat- correct4.mov",
    "Bench press- head lift.mov",
    "Deadlift- bar drift.mov",
    "Push press- elbows drop.mov",
    "Bench press- bar path.mov",
    "Push press- knee cave.mov",
    "Push Press- Head Clearance .mov",
    "clean-correct.mov",
    "snatch- correct.mov",
    "OverheadSquat- correct5.mov",
    "OverheadSquats- correct3.mov",
    "Back Squat- knee valgus.mov",
    "Front Squat- Depth.mov",
    "Deadlift - bad back.mov",
    "backsquat_short.mov",
    "pushpress_short.mov",
    "thruster_one_rep_720p.mp4",
}

LABEL_RULES = [
    ("handstandpushups", "handstand_push_up"),
    ("handstand", "handstand_push_up"),
    ("cleanandjerk", "clean_and_jerk"),
    ("clean_and_jerk", "clean_and_jerk"),
    ("benchpress", "bench_press"),
    ("bench_press", "bench_press"),
    ("bench press", "bench_press"),
    ("pullups", "pull_up"),
    ("pullup", "pull_up"),
    ("pull up", "pull_up"),
    ("pushups", "push_up"),
    ("pushup", "push_up"),
    ("push-up", "push_up"),
    ("split_jerk", "split_jerk"),
    ("splitjerk", "split_jerk"),
    ("strict_press", "strict_press"),
    ("push_press", "push_press"),
    ("squat_back", "squat_back"),
    ("squat_front", "squat_front"),
    ("overhead_squat", "overhead_squat"),
    ("deadlift", "deadlift"),
    ("snatch", "snatch"),
    ("clean", "clean"),
    ("jerk", "split_jerk"),
    ("not_oly", "not_oly"),
]

def norm(s):
    return s.lower().replace("_", "").replace("-", "").replace(" ", "")

def label_for(path):
    text = " ".join(part.lower() for part in path.parts)
    compact = norm(text)
    for key, label in LABEL_RULES:
        if key.lower() in text or norm(key) in compact:
            return label
    return "unknown"

def is_ignored(path):
    return any(part in IGNORE_PARTS for part in path.parts)

def is_trusted(path):
    sp = str(path)
    return any(root in sp for root in PROTECTED_ROOTS)


def is_protected(row):
    path = row["path"]
    name = Path(path).name

    if name in PROTECTED_NAMES:
        return True

    return any(root in path for root in PROTECTED_ROOTS)

def scan():
    videos = []
    for p in ROOT.rglob("*"):
        if is_ignored(p):
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        try:
            st = p.stat()
        except Exception:
            continue
        videos.append({
            "path": str(p),
            "name": p.name,
            "size": st.st_size,
            "parent": p.parent.name,
            "label": label_for(p),
            "trusted": is_trusted(p),
        })
    return videos

def choose_canonical(rows):
    trusted = [r for r in rows if r["trusted"]]
    candidates = trusted if trusted else rows

    def rank(r):
        path = r["path"]
        for i, root in enumerate(PROTECTED_ROOTS):
            if root in path:
                return (i, len(path), path)
        return (999, len(path), path)

    return sorted(candidates, key=rank)[0]

def audit(args):
    videos = scan()
    counts = defaultdict(int)
    for r in videos:
        counts[r["label"]] += 1

    print("\nDataset Audit")
    print("=" * 60)
    print(f"Active videos: {len(videos)}")
    print("=" * 60)

    for label, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{n:5d}  {label}")

def duplicates(args):
    videos = scan()
    groups = defaultdict(list)

    for r in videos:
        groups[(r["name"].lower(), r["size"])].append(r)

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}

    print("\nDuplicate Report")
    print("=" * 60)
    print(f"Duplicate groups: {len(dup_groups)}")
    print(f"Duplicate files: {sum(len(v) for v in dup_groups.values())}")
    print("=" * 60)

    for _, rows in sorted(dup_groups.items(), key=lambda x: len(x[1]), reverse=True)[:args.limit]:
        keep = choose_canonical(rows)
        labels = sorted({r["label"] for r in rows})
        print("\n---")
        print(f"{rows[0]['name']} | copies={len(rows)} | labels={labels}")
        print("KEEP:")
        print(" ", keep["path"])
        print("REMOVE:")
        for r in rows:
            if r["path"] != keep["path"]:
                print(" ", r["path"])

def canonicalize(args):
    videos = scan()
    groups = defaultdict(list)

    for r in videos:
        groups[(r["name"].lower(), r["size"])].append(r)

    rows_out = []

    for _, rows in groups.items():
        if len(rows) <= 1:
            continue

        keep = choose_canonical(rows)

        for r in rows:
            rows_out.append({
                "keep": "YES" if r["path"] == keep["path"] else "",
                "label": r["label"],
                "canonical": keep["path"],
                "duplicate": r["path"],
                "trusted": r["trusted"],
            })

    out = REPORT_DIR / "canonicalization_plan.csv"

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["keep", "label", "canonical", "duplicate", "trusted"])
        w.writeheader()
        w.writerows(rows_out)

    print("\nCanonicalization plan written:")
    print(out)
    print(f"Rows: {len(rows_out)}")

def conflicts(args):
    videos = scan()
    groups = defaultdict(list)

    for r in videos:
        groups[(r["name"].lower(), r["size"])].append(r)

    conflict_rows = []

    for _, rows in groups.items():
        labels = sorted({r["label"] for r in rows})
        if len(labels) <= 1:
            continue

        for r in rows:
            out = dict(r)
            out["labels_in_group"] = ",".join(labels)
            conflict_rows.append(out)

    out = REPORT_DIR / "conflicting_label_duplicates.csv"

    with out.open("w", newline="") as f:
        fields = ["label", "labels_in_group", "name", "size", "parent", "trusted", "path"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in conflict_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print("\nConflicts written:")
    print(out)
    print(f"Rows: {len(conflict_rows)}")

def unknown(args):
    rows = [r for r in scan() if r["label"] == "unknown"]

    out = REPORT_DIR / "unknown_videos.csv"

    with out.open("w", newline="") as f:
        fields = ["label", "name", "size", "parent", "trusted", "path"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print("\nUnknown videos written:")
    print(out)
    print(f"Rows: {len(rows)}")

def move_duplicates(args):
    videos = scan()
    groups = defaultdict(list)

    for r in videos:
        groups[(r["name"].lower(), r["size"])].append(r)

    dest_root = ROOT / "_Duplicate_Backups" / "canonicalized_duplicates"
    planned = []

    for _, rows in groups.items():
        if len(rows) <= 1:
            continue

        keep = choose_canonical(rows)

        for r in rows:
            src = Path(r["path"])
            if r["path"] == keep["path"]:
                continue

            # Never move files from protected/canonical dataset folders
            # or known benchmark/regression clip names.
            if is_protected(r):
                continue

            rel = src.relative_to(ROOT)
            dest = dest_root / rel
            planned.append((src, dest))

    print("\nMove duplicate plan")
    print("=" * 60)
    print(f"Files to move: {len(planned)}")
    print(f"Destination: {dest_root}")
    print("=" * 60)

    for src, dest in planned[:args.limit]:
        print(f"{src} -> {dest}")

    if not args.apply:
        print("\nDry run only. Add --apply to move files.")
        return

    for src, dest in planned:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dest))

    print("\nMoved duplicates.")

def verify(args):
    videos = scan()
    groups = defaultdict(list)
    unknown_count = 0

    for r in videos:
        groups[(r["name"].lower(), r["size"])].append(r)
        if r["label"] == "unknown":
            unknown_count += 1

    dup_groups = [v for v in groups.values() if len(v) > 1]
    conflicts_count = 0

    for rows in dup_groups:
        if len({r["label"] for r in rows}) > 1:
            conflicts_count += 1

    print("\nDataset Verify")
    print("=" * 60)
    print(f"Active videos: {len(videos)}")
    print(f"Duplicate groups: {len(dup_groups)}")
    print(f"Conflicting duplicate groups: {conflicts_count}")
    print(f"Unknown videos: {unknown_count}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit")
    p = sub.add_parser("duplicates")
    p.add_argument("--limit", type=int, default=50)

    sub.add_parser("canonicalize")
    sub.add_parser("conflicts")
    sub.add_parser("unknown")
    sub.add_parser("verify")

    p = sub.add_parser("move-duplicates")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--limit", type=int, default=30)

    args = parser.parse_args()

    {
        "audit": audit,
        "duplicates": duplicates,
        "canonicalize": canonicalize,
        "conflicts": conflicts,
        "unknown": unknown,
        "verify": verify,
        "move-duplicates": move_duplicates,
    }[args.cmd](args)

if __name__ == "__main__":
    main()
