"""
Olympic Router V7 feature wrapper.

Single source of truth for the feature vector used by the
64-feature Olympic router candidate.
"""

import numpy as np

from app.feature_engine.feature_names_v2 import FEATURE_NAMES
from app.feature_engine.movement_video_features_v2 import build_movement_video_features


def build_router_v7_features(biomechanics):
    """
    Returns:
        X: shape (1, n_features) float32 numpy array
    """
    feats = build_movement_video_features(biomechanics)

    if len(feats) != len(FEATURE_NAMES):
        raise ValueError(
            f"Router V7 feature mismatch: got {len(feats)}, expected {len(FEATURE_NAMES)}"
        )

    return np.asarray(feats, dtype=np.float32).reshape(1, -1)
