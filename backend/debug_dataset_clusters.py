import numpy as np

def analyze_sample(biomech):
    hip_y = np.array([b["hip_y"] for b in biomech])
    knee = np.array([b["knee_angle"] for b in biomech])
    wrist = np.array([b["wrist_y"] for b in biomech])
    shoulder = np.array([b["shoulder_y"] for b in biomech])

    hip_v = np.diff(hip_y)
    hip_a = np.diff(hip_v)

    explosion = np.max(np.abs(hip_a)) if len(hip_a) > 0 else 0
    depth = np.min(knee)
    overhead = np.mean(wrist < shoulder)

    return {
        "explosion": explosion,
        "depth": depth,
        "overhead": overhead
    }


def classify_cluster(f):
    if f["explosion"] > 8 and f["overhead"] > 0.4:
        return "OLYMPIC_LIKE"

    if f["depth"] < 110:
        return "SQUAT_LIKE"

    return "HINGE_OR_OTHER"


# ====== SIMULATE YOUR DATASET HERE ======
# Replace this with your real biomech dataset loader

dataset = {
    "snatch": [],
    "clean": [],
    "squat_back": [],
    "deadlift": []
}

results = {}

for label, samples in dataset.items():
    counts = {"OLYMPIC_LIKE": 0, "SQUAT_LIKE": 0, "HINGE_OR_OTHER": 0}

    for biomech in samples:
        f = analyze_sample(biomech)
        cluster = classify_cluster(f)
        counts[cluster] += 1

    total = max(len(samples), 1)

    results[label] = {
        "OLYMPIC_%": counts["OLYMPIC_LIKE"] / total,
        "SQUAT_%": counts["SQUAT_LIKE"] / total,
        "HINGE_%": counts["HINGE_OR_OTHER"] / total,
    }

print("\n=== DATASET COLLAPSE REPORT ===\n")
for k, v in results.items():
    print(k, v)