import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

LABELS = ['bench_press', 'deadlift', 'push_press', 'squat']
LABEL_DISPLAY = {
    'bench_press': 'Bench Press',
    'deadlift': 'Deadlift',
    'push_press': 'Push Press',
    'squat': 'Squat',
}
SEQ_LEN = 30
CLIP_VAL = 5.0

LM = {
    'NOSE': 0, 'LEFT_EAR': 7, 'RIGHT_EAR': 8,
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
}

FEATURE_LM_NAMES = [
    'NOSE', 'LEFT_EAR', 'RIGHT_EAR',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST',
    'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE',
    'LEFT_HEEL', 'RIGHT_HEEL'
]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def angle2d(ax, ay, bx, by, cx, cy):
    bax = ax - bx
    bay = ay - by
    bcx = cx - bx
    bcy = cy - by
    dot = bax * bcx + bay * bcy
    norm_a = math.sqrt(bax * bax + bay * bay)
    norm_c = math.sqrt(bcx * bcx + bcy * bcy)
    denom = max(norm_a * norm_c, 1e-6)
    return math.degrees(math.acos(clamp(dot / denom, -1.0, 1.0)))


def extract_features(lms) -> Tuple[np.ndarray | None, float]:
    try:
        raw: Dict[str, Dict[str, float]] = {}
        vis = []
        for name in FEATURE_LM_NAMES:
            idx = LM[name]
            lm = lms[idx]
            raw[name] = {
                'x': float(lm.x),
                'y': float(lm.y),
                'z': float(getattr(lm, 'z', 0.0) or 0.0),
            }
            vis.append(float(getattr(lm, 'visibility', 0.0) or 0.0))

        lh = raw['LEFT_HIP']
        rh = raw['RIGHT_HIP']
        ls = raw['LEFT_SHOULDER']
        rs = raw['RIGHT_SHOULDER']

        hip_x = (lh['x'] + rh['x']) / 2.0
        hip_y = (lh['y'] + rh['y']) / 2.0
        hip_z = (lh['z'] + rh['z']) / 2.0
        sh_x = (ls['x'] + rs['x']) / 2.0
        sh_y = (ls['y'] + rs['y']) / 2.0
        sh_z = (ls['z'] + rs['z']) / 2.0
        torso = max(math.sqrt((sh_x - hip_x) ** 2 + (sh_y - hip_y) ** 2 + (sh_z - hip_z) ** 2), 1e-6)

        norm: Dict[str, Dict[str, float]] = {}
        for name in FEATURE_LM_NAMES:
            norm[name] = {
                'x': clamp((raw[name]['x'] - hip_x) / torso, -CLIP_VAL, CLIP_VAL),
                'y': clamp((raw[name]['y'] - hip_y) / torso, -CLIP_VAL, CLIP_VAL),
                'z': clamp((raw[name]['z'] - hip_z) / torso, -CLIP_VAL, CLIP_VAL),
            }

        base_feat: List[float] = []
        for name in FEATURE_LM_NAMES:
            base_feat.extend([norm[name]['x'], norm[name]['y'], norm[name]['z']])

        lsn = norm['LEFT_SHOULDER']; rsn = norm['RIGHT_SHOULDER']
        lhn = norm['LEFT_HIP']; rhn = norm['RIGHT_HIP']
        lkn = norm['LEFT_KNEE']; rkn = norm['RIGHT_KNEE']
        lan = norm['LEFT_ANKLE']; ran = norm['RIGHT_ANKLE']
        len_ = norm['LEFT_ELBOW']; ren = norm['RIGHT_ELBOW']
        lwn = norm['LEFT_WRIST']; rwn = norm['RIGHT_WRIST']

        sh_mid_x = (lsn['x'] + rsn['x']) / 2.0
        sh_mid_y = (lsn['y'] + rsn['y']) / 2.0
        hip_mid_y = (lhn['y'] + rhn['y']) / 2.0
        knee_mid_y = (lkn['y'] + rkn['y']) / 2.0

        torso_angle = math.degrees(math.atan2(sh_mid_y - hip_mid_y, sh_mid_x - ((lhn['x'] + rhn['x']) / 2.0)))
        hip_depth = hip_mid_y - knee_mid_y
        lkt = lkn['x'] - lan['x']
        rkt = rkn['x'] - ran['x']
        knee_track_mean = (lkt + rkt) / 2.0
        ler = len_['y'] - lsn['y']
        rer = ren['y'] - rsn['y']
        elbow_rel_mean = (ler + rer) / 2.0
        lwr = lwn['y'] - lsn['y']
        rwr = rwn['y'] - rsn['y']
        wrist_rel_mean = (lwr + rwr) / 2.0
        shoulder_width = math.sqrt((lsn['x'] - rsn['x']) ** 2 + (lsn['y'] - rsn['y']) ** 2)
        hip_width = math.sqrt((lhn['x'] - rhn['x']) ** 2 + (lhn['y'] - rhn['y']) ** 2)
        stance_width = math.sqrt((lan['x'] - ran['x']) ** 2 + (lan['y'] - ran['y']) ** 2)
        lka = angle2d(lhn['x'], lhn['y'], lkn['x'], lkn['y'], lan['x'], lan['y'])
        rka = angle2d(rhn['x'], rhn['y'], rkn['x'], rkn['y'], ran['x'], ran['y'])
        knee_angle_mean = (lka + rka) / 2.0
        lha = angle2d(lsn['x'], lsn['y'], lhn['x'], lhn['y'], lkn['x'], lkn['y'])
        rha = angle2d(rsn['x'], rsn['y'], rhn['x'], rhn['y'], rkn['x'], rkn['y'])
        hip_angle_mean = (lha + rha) / 2.0

        engineered = np.array([
            torso_angle, hip_depth,
            lkt, rkt, knee_track_mean,
            ler, rer, elbow_rel_mean,
            lwr, rwr, wrist_rel_mean,
            shoulder_width, hip_width, stance_width,
            lka, rka, knee_angle_mean,
            lha, rha, hip_angle_mean
        ], dtype=np.float32)

        feat = np.array(base_feat, dtype=np.float32)
        return np.concatenate([feat, engineered]).astype(np.float32), float(np.mean(vis))
    except Exception:
        return None, 0.0


class NumpyFormCheckModel:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._load_weights()

    def _load_weights(self):
        model_json_path = self.model_dir / 'model.json'
        model_bin_path = self.model_dir / 'group1-shard1of1.bin'

        with model_json_path.open('r') as f:
            model_json = json.load(f)

        weights_manifest = model_json['weightsManifest'][0]['weights']
        raw = np.fromfile(model_bin_path, dtype=np.float32)
        cursor = 0
        weights = {}

        for item in weights_manifest:
            shape = item['shape']
            size = int(np.prod(shape))
            arr = raw[cursor: cursor + size].reshape(shape).astype(np.float32)
            weights[item['name']] = arr
            cursor += size

        self.bn1_gamma = weights['batch_normalization/gamma']
        self.bn1_beta = weights['batch_normalization/beta']
        self.bn1_mean = weights['batch_normalization/moving_mean']
        self.bn1_var = weights['batch_normalization/moving_variance']

        self.bn2_gamma = weights['batch_normalization_1/gamma']
        self.bn2_beta = weights['batch_normalization_1/beta']
        self.bn2_mean = weights['batch_normalization_1/moving_mean']
        self.bn2_var = weights['batch_normalization_1/moving_variance']

        self.dense1_kernel = weights['dense/kernel']
        self.dense1_bias = weights['dense/bias']
        self.dense2_kernel = weights['dense_1/kernel']
        self.dense2_bias = weights['dense_1/bias']

        self.lstm1_kernel = weights['lstm/lstm_cell/kernel']
        self.lstm1_recurrent = weights['lstm/lstm_cell/recurrent_kernel']
        self.lstm1_bias = weights['lstm/lstm_cell/bias']

        self.lstm2_kernel = weights['lstm_1/lstm_cell/kernel']
        self.lstm2_recurrent = weights['lstm_1/lstm_cell/recurrent_kernel']
        self.lstm2_bias = weights['lstm_1/lstm_cell/bias']

    def _batch_norm(self, x, gamma, beta, mean, var, eps=1e-3):
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def _lstm_forward(self, seq, kernel, recurrent, bias, units):
        h = np.zeros((units,), dtype=np.float32)
        c = np.zeros((units,), dtype=np.float32)
        outputs = []

        for t in range(seq.shape[0]):
            z = seq[t] @ kernel + h @ recurrent + bias
            i = sigmoid(z[:units])
            f = sigmoid(z[units:2 * units])
            g = np.tanh(z[2 * units:3 * units])
            o = sigmoid(z[3 * units:])
            c = f * c + i * g
            h = o * np.tanh(c)
            outputs.append(h.copy())

        return np.stack(outputs, axis=0), h, c

    def predict_proba(self, seq_142: np.ndarray) -> np.ndarray:
        out1, _, _ = self._lstm_forward(seq_142, self.lstm1_kernel, self.lstm1_recurrent, self.lstm1_bias, 64)
        out1 = self._batch_norm(out1, self.bn1_gamma, self.bn1_beta, self.bn1_mean, self.bn1_var)

        out2_seq, out2_h, _ = self._lstm_forward(out1, self.lstm2_kernel, self.lstm2_recurrent, self.lstm2_bias, 32)
        x = self._batch_norm(out2_h, self.bn2_gamma, self.bn2_beta, self.bn2_mean, self.bn2_var)

        x = x @ self.dense1_kernel + self.dense1_bias
        x = np.maximum(x, 0.0)
        x = x @ self.dense2_kernel + self.dense2_bias

        return softmax(x).astype(np.float32)
