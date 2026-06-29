import math
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


def _angle(a, b, c):
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    mag1 = math.hypot(bax, bay)
    mag2 = math.hypot(bcx, bcy)
    if mag1 == 0 or mag2 == 0:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))


def _norm(vals):
    a = np.array(vals, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _nearest(records, target, lo, hi):
    pool = [r for r in records if lo <= r["frame"] <= hi]
    if not pool:
        return None
    return min(pool, key=lambda r: abs(r["frame"] - target))


def extract_pose_records(video_path, start, end, sample_step=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, min(int(start), total - 1))
    end = max(start + 1, min(int(end), total - 1))

    records = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        for frame_idx in range(start, end + 1, sample_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            P = mp_pose.PoseLandmark

            def pt(x):
                return (lm[x.value].x, lm[x.value].y)

            l_sh, r_sh = pt(P.LEFT_SHOULDER), pt(P.RIGHT_SHOULDER)
            l_hip, r_hip = pt(P.LEFT_HIP), pt(P.RIGHT_HIP)
            l_knee, r_knee = pt(P.LEFT_KNEE), pt(P.RIGHT_KNEE)
            l_ankle, r_ankle = pt(P.LEFT_ANKLE), pt(P.RIGHT_ANKLE)

            hip_y = (l_hip[1] + r_hip[1]) / 2.0
            knee = (_angle(l_hip, l_knee, l_ankle) + _angle(r_hip, r_knee, r_ankle)) / 2.0
            hip = (_angle(l_sh, l_hip, l_knee) + _angle(r_sh, r_hip, r_knee)) / 2.0

            records.append({
                "frame": frame_idx,
                "hip_y": hip_y,
                "knee": knee,
                "hip": hip,
            })

    cap.release()

    if len(records) < 5:
        return records

    hip_y_n = _norm([r["hip_y"] for r in records])
    knee_flex_n = _norm([180.0 - r["knee"] for r in records])
    hip_flex_n = _norm([180.0 - r["hip"] for r in records])
    upright_n = _norm([r["knee"] + r["hip"] - (r["hip_y"] * 120.0) for r in records])

    for i, r in enumerate(records):
        r["bottom_score"] = 0.45 * hip_y_n[i] + 0.35 * knee_flex_n[i] + 0.20 * hip_flex_n[i]
        r["upright_score"] = float(upright_n[i])

    return records


def find_bottom(records, approx_bottom, radius=16):
    pool = [
        r for r in records
        if abs(r["frame"] - int(approx_bottom)) <= radius
    ]

    if not pool:
        pool = records

    max_score = max(r["bottom_score"] for r in pool)
    plateau = [r for r in pool if r["bottom_score"] >= max_score * 0.98]
    return plateau[0] if plateau else max(pool, key=lambda r: r["bottom_score"])


def find_setup_before_bottom(records, bottom):
    before = [r for r in records if r["frame"] < bottom["frame"] - 6]
    upright = [r for r in before if r["knee"] >= 135 and r["hip"] >= 105]

    pool = upright if upright else before
    if not pool:
        return records[0]

    best = max(r["upright_score"] for r in pool)
    candidates = [r for r in pool if r["upright_score"] >= best * 0.95]

    # Setup should be the last stable upright frame before descent.
    return min(candidates, key=lambda r: r["frame"])


def find_lockout_after_bottom(records, bottom):
    after = [r for r in records if r["frame"] > bottom["frame"] + 10]
    upright = [r for r in after if r["knee"] >= 135 and r["hip"] >= 105]

    pool = upright if upright else after
    if not pool:
        return records[-1]

    best = max(r["upright_score"] for r in pool)
    candidates = [r for r in pool if r["upright_score"] >= best * 0.95]

    # Lockout should be the first stable upright frame after ascent.
    return min(candidates, key=lambda r: r["frame"])


def pick_squat_visual_phases(
    video_path,
    start_frame,
    bottom_frame,
    end_frame,
    sample_step=2,
    pad_before=35,
    pad_after=45,
):
    visual_start = max(0, int(start_frame) - pad_before)
    visual_end = int(end_frame) + pad_after

    records = extract_pose_records(video_path, visual_start, visual_end, sample_step)

    if len(records) < 8:
        return {"error": "not_enough_pose_frames", "records": len(records)}

    bottom = find_bottom(records, bottom_frame)
    setup = find_setup_before_bottom(records, bottom)
    lockout = find_lockout_after_bottom(records, bottom)

    setup_f = setup["frame"]
    bottom_f = bottom["frame"]
    lockout_f = lockout["frame"]

    if not (setup_f < bottom_f < lockout_f):
        return {
            "error": "anchor_order_invalid",
            "setup": setup_f,
            "bottom": bottom_f,
            "lockout": lockout_f,
        }

    descent = _nearest(records, setup_f + (bottom_f - setup_f) * 0.50, setup_f + 1, bottom_f - 1)
    ascent = _nearest(records, bottom_f + (lockout_f - bottom_f) * 0.50, bottom_f + 1, lockout_f - 1)

    if not descent or not ascent:
        return {
            "error": "missing_midpoint",
            "setup": setup_f,
            "bottom": bottom_f,
            "lockout": lockout_f,
        }

    return {
        "setup": setup_f,
        "descent": descent["frame"],
        "bottom": bottom_f,
        "ascent": ascent["frame"],
        "lockout": lockout_f,
    }
