from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class TrackedPose:
    landmarks: Any
    center: tuple[float, float]
    area: float
    score: float


class SubjectTracker:
    def __init__(self, model_path: str, num_poses: int = 4):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Pose model not found: {model_path}")

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

    def close(self):
        self.landmarker.close()

    def _pose_stats(self, landmarks):
        idxs = [
            11, 12, 23, 24,  # shoulders, hips
        ]

        pts = [landmarks[i] for i in idxs]
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]

        center = (sum(xs) / len(xs), sum(ys) / len(ys))
        area = max(1e-6, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        return center, area

    def _score_pose(self, center, area):
        image_center = (0.5, 0.5)
        center_dist = ((center[0] - image_center[0]) ** 2 + (center[1] - image_center[1]) ** 2) ** 0.5

        if self.subject_center is None:
            tracking_dist = 0.0
        else:
            tracking_dist = (
                (center[0] - self.subject_center[0]) ** 2
                + (center[1] - self.subject_center[1]) ** 2
            ) ** 0.5

        return (
            2.0 * area
            - 0.8 * center_dist
            - 1.8 * tracking_dist
        )

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        candidates: list[TrackedPose] = []

        for landmarks in result.pose_landmarks:
            center, area = self._pose_stats(landmarks)
            score = self._score_pose(center, area)
            candidates.append(TrackedPose(landmarks, center, area, score))

        best = max(candidates, key=lambda p: p.score)

        if self.subject_center is not None:
            dx = best.center[0] - self.subject_center[0]
            dy = best.center[1] - self.subject_center[1]
            jump = (dx * dx + dy * dy) ** 0.5

            area_ratio = best.area / max(self.subject_area or best.area, 1e-6)

            if jump > 0.25 or area_ratio < 0.35 or area_ratio > 2.8:
                return None

        if self.subject_center is None:
            self.subject_center = best.center
            self.subject_area = best.area
        else:
            self.subject_center = (
                self.subject_center[0] * 0.85 + best.center[0] * 0.15,
                self.subject_center[1] * 0.85 + best.center[1] * 0.15,
            )
            self.subject_area = (self.subject_area or best.area) * 0.85 + best.area * 0.15

        return best.landmarks