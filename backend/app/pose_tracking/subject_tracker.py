from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class Candidate:
    landmarks: Any
    center: tuple[float, float]
    area: float
    box: tuple[float, float, float, float]
    score: float


class SubjectTracker:
    def __init__(self, model_path: str, num_poses: int = 4):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Pose model not found: {model_path}")

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = PoseLandmarker.create_from_options(options)

        self.subject_center: Optional[tuple[float, float]] = None
        self.subject_area: Optional[float] = None
        self.last_box: Optional[tuple[float, float, float, float]] = None
        self.last_debug: dict = {}

    def close(self):
        self.landmarker.close()

    def _candidate(self, landmarks) -> Candidate:
        visible = [p for p in landmarks if getattr(p, "visibility", 1.0) >= 0.30]
        if len(visible) < 8:
            visible = landmarks

        xs = [p.x for p in visible]
        ys = [p.y for p in visible]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        box = (x1, y1, x2, y2)
        area = max(1e-6, (x2 - x1) * (y2 - y1))

        torso = [landmarks[i] for i in [11, 12, 23, 24]]
        cx = sum(p.x for p in torso) / len(torso)
        cy = sum(p.y for p in torso) / len(torso)
        center = (cx, cy)

        center_dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
        lower_body_bonus = y2

        if self.subject_center is None:
            track_dist = 0.0

            # Initial lock: choose the closest/front athlete.
            # Front athlete usually has the biggest visible body box and lowest feet/body.
            # Initial lock: prioritize the front person.
            # The front lifter usually has the lowest visible feet/body in the frame.
            score = (4.0 * area) + (12.0 * lower_body_bonus) - (0.2 * center_dist)
        else:
            track_dist = (
                (cx - self.subject_center[0]) ** 2
                + (cy - self.subject_center[1]) ** 2
            ) ** 0.5

            # After lock: strongly prefer staying on the same nearby foreground athlete.
            # After lock: do not switch easily.
            score = (3.0 * area) + (6.0 * lower_body_bonus) - (18.0 * track_dist) - (0.2 * center_dist)

        return Candidate(landmarks, center, area, box, score)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            self.last_debug = {"status": "no_pose"}
            return None

        candidates = [self._candidate(lms) for lms in result.pose_landmarks]

        if self.subject_center is None:
            best = max(candidates, key=lambda c: c.score)
        else:
            valid = []
            for c in candidates:
                dx = c.center[0] - self.subject_center[0]
                dy = c.center[1] - self.subject_center[1]
                jump = (dx * dx + dy * dy) ** 0.5
                area_ratio = c.area / max(self.subject_area or c.area, 1e-6)

                if jump <= 0.18 and 0.50 <= area_ratio <= 1.90:
                    valid.append(c)

            if not valid:
                self.last_debug = {
                    "status": "rejected_all",
                    "num_candidates": len(candidates),
                }
                return None

            best = max(valid, key=lambda c: c.score)

        if self.subject_center is None:
            self.subject_center = best.center
            self.subject_area = best.area
        else:
            self.subject_center = (
                self.subject_center[0] * 0.90 + best.center[0] * 0.10,
                self.subject_center[1] * 0.90 + best.center[1] * 0.10,
            )
            self.subject_area = (self.subject_area or best.area) * 0.90 + best.area * 0.10

        self.last_box = best.box
        self.last_debug = {
            "status": "tracked",
            "num_candidates": len(candidates),
            "area": best.area,
            "center": best.center,
            "score": best.score,
        }

        return best.landmarks
