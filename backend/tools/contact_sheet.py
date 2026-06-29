import argparse
from pathlib import Path
import cv2
import numpy as np

PHASES = ["SETUP", "DESCENT", "BOTTOM", "ASCENT", "LOCKOUT"]

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--frames", nargs="+", type=int, required=True)
parser.add_argument("--out", default=None)
args = parser.parse_args()

video = args.video
frames = args.frames

if len(frames) != 5:
    raise SystemExit("Please provide exactly 5 frames: setup descent bottom ascent lockout")

out = args.out
if out is None:
    stem = Path(video).stem.replace(" ", "_")
    out = f"outputs/{stem}_contact_sheet.jpg"

Path(out).parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video)
imgs = []

for label, frame_idx in zip(PHASES, frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()

    if not ok:
        img = np.zeros((280, 420, 3), dtype=np.uint8)
        cv2.putText(img, "MISSING FRAME", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    else:
        img = cv2.resize(img, (420, 280))

    cv2.rectangle(img, (0, 0), (420, 42), (0, 0, 0), -1)
    cv2.putText(img, f"{label} {frame_idx}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    imgs.append(img)

cap.release()

sheet = cv2.hconcat(imgs)
cv2.imwrite(out, sheet)
print(out)
