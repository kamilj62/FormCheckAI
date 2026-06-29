import cv2
import numpy as np
from pathlib import Path
from app.phase_engine.squat_v3 import pick_squat_visual_phases

OUT = Path("outputs/squat_v1_vs_v3_audit.jpg")
OUT.parent.mkdir(parents=True, exist_ok=True)

TESTS = [
    ("knee_valgus", "/Users/josephkamil/Desktop/Capstone/Back Squat- knee valgus.mov", {
        "setup": 257, "descent": 275, "bottom": 298, "ascent": 329, "lockout": 360
    }, 257, 298, 360),

    ("heel_rise", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/Back squats- heel rise.mov", {
        "setup": 160, "descent": 175, "bottom": 189, "ascent": 197, "lockout": 207
    }, 160, 189, 207),

    ("bar_drift", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/Back squats- bar drift.mov", {
        "setup": 152, "descent": 160, "bottom": 167, "ascent": 168, "lockout": 171
    }, 152, 167, 171),

    ("yt_001", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/yt_001.mp4", {
        "setup": 1, "descent": 29, "bottom": 52, "ascent": 58, "lockout": 67
    }, 1, 52, 67),
]

PHASES = ["pre_roll", "setup", "descent", "bottom", "ascent", "lockout"]
CELL_W, CELL_H = 230, 160
LABEL_W = 220
HEADER_H = 90
ROW_H = CELL_H * 2 + 80

def read_cell(video, label, frame_idx, tag):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(int(frame_idx), total - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    else:
        frame = cv2.resize(frame, (CELL_W, CELL_H))

    cv2.rectangle(frame, (0, 0), (CELL_W, 34), (0, 0, 0), -1)
    cv2.putText(frame, f"{tag} {label.upper()} {frame_idx}", (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255,255,255), 1)
    return frame

rows = []
width = LABEL_W + len(PHASES) * CELL_W

header = np.ones((HEADER_H, width, 3), dtype=np.uint8) * 245
cv2.putText(header, "SQUAT PHASE AUDIT: PRE-ROLL + V1 vs V3", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

for i, phase in enumerate(PHASES):
    x = LABEL_W + i * CELL_W + 20
    cv2.putText(header, phase.upper(), (x, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

rows.append(header)

for name, video, v1, start, bottom, end in TESTS:
    v3 = pick_squat_visual_phases(video, start, bottom, end, sample_step=2)

    row = np.ones((ROW_H, width, 3), dtype=np.uint8) * 235

    cv2.putText(row, name, (12, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 2)
    cv2.putText(row, f"start={start}", (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
    cv2.putText(row, f"bottom={bottom}", (12, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
    cv2.putText(row, f"end={end}", (12, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

    if not v3 or "error" in v3:
        cv2.putText(row, f"V3 ERROR", (12, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
        v3 = v1

    pre = max(0, start - 25)

    v1_with_pre = {"pre_roll": pre, **v1}
    v3_with_pre = {"pre_roll": pre, **v3}

    for i, phase in enumerate(PHASES):
        x = LABEL_W + i * CELL_W

        row[20:20+CELL_H, x:x+CELL_W] = read_cell(
            video, phase, v1_with_pre[phase], "V1"
        )

        row[45+CELL_H:45+CELL_H*2, x:x+CELL_W] = read_cell(
            video, phase, v3_with_pre[phase], "V3"
        )

    rows.append(row)

sheet = cv2.vconcat(rows)
cv2.imwrite(str(OUT), sheet)
print(OUT)
