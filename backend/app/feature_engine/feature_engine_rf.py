import numpy as np

def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0


def build_rf_features(biomechanics):
    features = []

    for b in biomechanics:

        features.extend([
            safe_float(b.get("knee_angle", 0)),
            safe_float(b.get("hip_angle", 0)),
            safe_float(b.get("ankle_angle", 0)),
            safe_float(b.get("shoulder_angle", 0)),
            safe_float(b.get("elbow_angle", 0)),

            safe_float(b.get("hip_y", 0)),
            safe_float(b.get("knee_y", 0)),
            safe_float(b.get("shoulder_y", 0)),

            safe_float(b.get("center_of_mass_x", 0)),
            safe_float(b.get("center_of_mass_y", 0)),
        ])

    knee_vals = [safe_float(b.get("knee_angle", 0)) for b in biomechanics]
    hip_vals = [safe_float(b.get("hip_angle", 0)) for b in biomechanics]

    if knee_vals:
        features.extend([
            np.mean(knee_vals),
            np.std(knee_vals),
            np.min(knee_vals),
            np.max(knee_vals),
        ])

    if hip_vals:
        features.extend([
            np.mean(hip_vals),
            np.std(hip_vals),
            np.min(hip_vals),
            np.max(hip_vals),
        ])

    features = np.array(features, dtype=np.float32)

    # safety pad to RF size
    if len(features) < 528:
        features = np.pad(features, (0, 528 - len(features)))
    else:
        features = features[:528]

    return features