import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_features_and_biomechanics(results):
    def safe_float(x):
        try:
            if x is None:
                return 0.0
            return float(x)
        except:
            return 0.0

    if not results.pose_landmarks:
        return None, None

    landmarks = results.pose_landmarks.landmark

    # -------------------------------
    # 1. 68-feature base vector
    # -------------------------------
    FEATURE_LANDMARKS = [
        "NOSE",
        "LEFT_EAR", "RIGHT_EAR",
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST",
        "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_HEEL", "RIGHT_HEEL",
    ]

    features = []

    for name in FEATURE_LANDMARKS:
        idx = mp_pose.PoseLandmark[name].value
        lm = landmarks[idx]

        features.extend([
            safe_float(lm.x),
            safe_float(lm.y),
            safe_float(lm.z),
            safe_float(lm.visibility),
        ])

    features = np.array(features, dtype=np.float32)

    # -------------------------------
    # 2. full 33-landmark vector (RF + future models)
    # -------------------------------
    full_features = np.array([
        safe_float(v)
        for lm in landmarks
        for v in [lm.x, lm.y, lm.z, lm.visibility]
    ], dtype=np.float32)

    # -------------------------------
    # 3. key points (SAFE)
    # -------------------------------
    def pt(lm):
        return np.array([safe_float(lm.x), safe_float(lm.y)], dtype=np.float32)

    left_shoulder = pt(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    right_shoulder = pt(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    left_elbow = pt(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
    right_elbow = pt(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    left_wrist = pt(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_wrist = pt(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    left_hip = pt(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_hip = pt(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    left_knee = pt(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    right_knee = pt(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    left_ankle = pt(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_ankle = pt(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    left_heel = pt(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value])
    right_heel = pt(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value])
    nose = pt(landmarks[mp_pose.PoseLandmark.NOSE.value])

    # -------------------------------
    # 4. derived metrics (SAFE)
    # -------------------------------
    ankle_mid = (left_ankle + right_ankle) / 2
    heel_mid = (left_heel + right_heel) / 2
    heel_lift = safe_float(ankle_mid[1] - heel_mid[1])

    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    knee_mid = (left_knee + right_knee) / 2
    elbow_mid = (left_elbow + right_elbow) / 2
    wrist_mid = (left_wrist + right_wrist) / 2

    knee_angle = safe_float(angle(left_hip, left_knee, left_ankle))
    hip_angle = safe_float(angle(left_shoulder, left_hip, left_knee))
    elbow_angle = safe_float(angle(left_shoulder, left_elbow, left_wrist))

    torso_angle = safe_float(angle(shoulder_mid, hip_mid, hip_mid + np.array([0, -1])))

    valgus_ratio = safe_float(
        np.clip(
            abs(left_knee[0] - right_knee[0]) /
            (abs(left_ankle[0] - right_ankle[0]) + 1e-6),
            0.5, 1.5
        )
    )

    biomechanics = {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "elbow_angle": elbow_angle,
        "torso_angle": torso_angle,

        "hip_y": safe_float(hip_mid[1]),
        "knee_y": safe_float(knee_mid[1]),
        "shoulder_y": safe_float(shoulder_mid[1]),
        "wrist_y": safe_float(wrist_mid[1]),
        "elbow_y": safe_float(elbow_mid[1]),

        "hip_x": safe_float(hip_mid[0]),
        "knee_x": safe_float(knee_mid[0]),
        "shoulder_x": safe_float(shoulder_mid[0]),

        "wrist_x": safe_float(wrist_mid[0]),

        "shoulder_hip_distance": safe_float(np.linalg.norm(shoulder_mid - hip_mid)),
        "hip_knee_distance": safe_float(np.linalg.norm(hip_mid - knee_mid)),
        "wrist_shoulder_distance": safe_float(np.linalg.norm(wrist_mid - shoulder_mid)),

        "valgus_ratio": valgus_ratio,
        "bar_distance": safe_float(abs(wrist_mid[0] - ankle_mid[0])),

        "head_drop": safe_float(nose[1] - shoulder_mid[1]),
        "head_forward": safe_float(abs(nose[0] - shoulder_mid[0])),

        "heel_lift": heel_lift,

        # RF input
        "full_features": full_features,
    }

    return features, biomechanics

