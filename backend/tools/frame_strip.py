import argparse
from pathlib import Path
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--frames", nargs="+", type=int, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(args.video)
imgs = []

for f in args.frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    if not ok:
        continue

    img = cv2.resize(img, (260, 180))
    cv2.rectangle(img, (0, 0), (260, 30), (0, 0, 0), -1)
    cv2.putText(
        img,
        str(f),
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    imgs.append(img)

cap.release()

if not imgs:
    raise SystemExit("No frames extracted.")

sheet = cv2.hconcat(imgs)
cv2.imwrite(args.out, sheet)
print(args.out)
