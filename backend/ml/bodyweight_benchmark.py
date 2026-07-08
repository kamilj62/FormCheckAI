from pathlib import Path
from datetime import datetime
import subprocess
import json
import shutil
import os
import sys
import pandas as pd

script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)

API_URL = os.getenv("API_URL", "http://localhost:8000")
FAILURES_DIR = Path("ml/reports/bodyweight_failures")

BENCHMARKS = [
    {
        "name": "handstand_push_up",
        "folders": [
            "/Users/josephkamil/Desktop/Capstone/HandstandPushups",
        ],
        "patterns": ["*.mp4", "*.mov", "*.avi"],
    },
    {
        "name": "pull_up",
        "folders": [
            "/Users/josephkamil/Desktop/Capstone/pull Up",
            "/Users/josephkamil/Desktop/Capstone/PullUps",
        ],
        "patterns": ["*.mp4", "*.mov", "*.avi"],
    },
    {
        "name": "push_up",
        "folders": [
            "/Users/josephkamil/Desktop/Capstone/push-up",
            "/Users/josephkamil/Desktop/Capstone/PushUps",
        ],
        "patterns": ["*.mp4", "*.mov", "*.avi"],
    },
]


def analyze_video(path):
    cmd = [
        "curl", "-s", "--max-time", "180",
        "-X", "POST",
        "-F", f"file=@{path}",
        f"{API_URL.rstrip('/')}/analyze",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=200,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": "request_timeout", "analysis_mode": "error"}

    if result.returncode != 0:
        return {
            "error": f"curl_failed_{result.returncode}",
            "stderr": result.stderr[-500:],
            "analysis_mode": "error",
        }

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "error": "invalid_json",
            "response": result.stdout[-500:],
            "analysis_mode": "error",
        }



def _safe_name(x):
    return str(x or "unknown").replace("/", "_").replace(" ", "_").replace(":", "_")


def save_failure_artifacts(row):
    true_label = _safe_name(row.get("true_label"))
    predicted = _safe_name(row.get("predicted_label"))
    video = row.get("video")

    stem = Path(str(video)).stem if video else "unknown_video"
    out_dir = FAILURES_DIR / true_label / f"{stem}__pred_{predicted}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(row)
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))


rows = []

for bench in BENCHMARKS:
    files = []

    for folder in bench["folders"]:
        folder = Path(folder)

        if not folder.exists():
            print("Missing folder:", folder)
            continue

        for pattern in bench["patterns"]:
            files.extend(folder.glob(pattern))

    files = sorted(set(files))

    print(f"\nFound {len(files)} files for {bench['name']}")

    for f in files:
        print("Testing", bench["name"], f.name, flush=True)
        print("FULL_PATH=", str(f), flush=True)
        data = analyze_video(f)

        debug = data.get("debug") or {}

        rows.append({
            "true_label": bench["name"],
            "predicted_label": data.get("exercise_label"),
            "confidence": data.get("confidence"),
            "analysis_mode": data.get("analysis_mode"),
            "error": data.get("error"),
            "valid_extraction": (
                data.get("error") is None
                and data.get("analysis_mode") not in {"insufficient_data", "error"}
                and data.get("exercise_label") not in {None, "Unknown"}
            ),
            "video": str(f),
            "rep_count": len(data.get("rep_feedback", [])),
            "raw_label": debug.get("raw_label"),
            "bio_label": debug.get("bio_label"),
            "squat_label": debug.get("squat_label"),
            "olympic_pred": debug.get("olympic_pred"),
            "bodyweight_router_label": debug.get("bodyweight_router_label"),
            "bodyweight_router_conf": debug.get("bodyweight_router_conf"),
            "protected_reason": debug.get("protected_reason"),
            "routing_trace": json.dumps(debug.get("routing_trace", []), default=str),
            "router_v6_label": debug.get("router_v6_label"),
            "router_v6_conf": debug.get("router_v6_conf"),
            "router_v6_decision": debug.get("router_v6_decision"),
        })


df = pd.DataFrame(rows)

Path("ml/reports").mkdir(parents=True, exist_ok=True)

csv_path = Path("ml/reports/bodyweight_benchmark_latest.csv")
df.to_csv(csv_path, index=False)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
history_dir = Path("ml/reports/history")
history_dir.mkdir(parents=True, exist_ok=True)

history_path = history_dir / f"bodyweight_benchmark_{timestamp}.csv"
df.to_csv(history_path, index=False)

print("\nSaved:", csv_path)
print("History:", history_path)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(pd.crosstab(df.true_label, df.predicted_label))

print("\n==============================")
print("PER-CLASS ACCURACY")
print("==============================")

for label in sorted(df.true_label.unique()):
    subset = df[df.true_label == label]
    acc = (subset.true_label == subset.predicted_label).mean()
    print(f"{label:20s} {100*acc:5.1f}%")

overall = (df.true_label == df.predicted_label).mean()
valid = df[df.valid_extraction == True]
valid_overall = (valid.true_label == valid.predicted_label).mean() if len(valid) else 0
invalid = df[df.valid_extraction != True]
v6_valid = df[df.router_v6_label.notna()] if "router_v6_label" in df.columns else pd.DataFrame()
v6_overall = (v6_valid.true_label == v6_valid.router_v6_label).mean() if len(v6_valid) else 0


print("\n==============================")
print(f"OVERALL RAW: {100*overall:.1f}%")
print(f"VALID ONLY: {100*valid_overall:.1f}% ({len(valid)}/{len(df)} valid)")
print(f"ROUTER V6 AUDIT: {100*v6_overall:.1f}% ({len(v6_valid)}/{len(df)} with v6 label)")
print(f"INVALID / INSUFFICIENT: {len(invalid)}")
print("==============================")

if len(invalid):
    print("\n==============================")
    print("INVALID / INSUFFICIENT VIDEOS")
    print("==============================")
    print(invalid[[
        "true_label",
        "predicted_label",
        "analysis_mode",
        "error",
        "video"
    ]].to_string(index=False))

mistakes = df[df.true_label != df.predicted_label]

print("\n==============================")
print("MISTAKES")
print("==============================")

if len(mistakes):
    print(mistakes.to_string(index=False))

    if FAILURES_DIR.exists():
        shutil.rmtree(FAILURES_DIR)
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in mistakes.iterrows():
        save_failure_artifacts(row.to_dict())

    print(f"\nSaved failure artifacts: {FAILURES_DIR}")
else:
    print("None 🎉")
