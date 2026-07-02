FEATURE_NAMES = [
    "knee_mean","knee_std","knee_min","knee_max","knee_delta",
    "hip_mean","hip_std","hip_min","hip_max","hip_delta",
    "elbow_mean","elbow_std","elbow_min","elbow_max","elbow_delta",
    "shoulder_mean","shoulder_std","shoulder_min","shoulder_max","shoulder_delta",

    "wrist_y_mean","wrist_y_std","wrist_y_min","wrist_y_max","wrist_y_delta",
    "hip_y_mean","hip_y_std","hip_y_min","hip_y_max","hip_y_delta",

    "wrist_shoulder_distance_mean",
    "wrist_shoulder_distance_std",
    "wrist_shoulder_distance_min",
    "wrist_shoulder_distance_max",
    "wrist_shoulder_distance_delta",

    "overhead_ratio",
    "has_overhead",
    "first_overhead_pct",
    "last_overhead_pct",
    "overhead_span_pct",

    "min_knee_angle",
    "min_hip_angle",
    "max_elbow_angle",
    "max_shoulder_angle",

    "min_knee_time_pct",
    "min_hip_time_pct",

    "max_wrist_motion",
    "mean_wrist_motion",
    "max_hip_motion",
    "mean_hip_motion",

    "overhead_near_bottom",
    "late_overhead_flag",
    "early_min_knee",
    "late_min_knee",
]

while len(FEATURE_NAMES) < 80:
    FEATURE_NAMES.append(f"reserved_{len(FEATURE_NAMES)}")
