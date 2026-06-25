import numpy as np


class SignalEngine:
    def __init__(self, sequence):
        self.sequence = sequence

        self.wrist_y = np.array([f.get("wrist_y", 0.0) for f in sequence])        
        self.wrist_x = np.array([f.get("wrist_x", 0.0) for f in sequence])
        self.hip_y = np.array([f.get("hip_y", 0.0) for f in sequence])
        self.knee = np.array([f.get("knee_angle", 0.0) for f in sequence])
        self.elbow = np.array([f.get("elbow_angle", 180.0) for f in sequence])

    def velocity(self, signal):
        return np.gradient(signal)

    def wrist_velocity(self):
        return self.velocity(self.wrist_y)

    def hip_velocity(self):
        return self.velocity(self.hip_y)

    def extension_peak(self):
        if len(self.knee) < 3:
            return 0

        # Olympic extension = body reaches its most open/tall position.
        # Use knee extension + hip/bar rise instead of wrist speed alone.
        n = len(self.knee)
        start = max(1, int(n * 0.15))
        end = max(start + 2, int(n * 0.70))

        knee_norm = (self.knee - np.min(self.knee)) / (np.ptp(self.knee) + 1e-6)

        # hip_y smaller means hips are higher on screen
        hip_rise = -self.hip_y
        hip_norm = (hip_rise - np.min(hip_rise)) / (np.ptp(hip_rise) + 1e-6)

        signal = 0.65 * knee_norm + 0.35 * hip_norm

        return start + int(np.argmax(signal[start:end]))

    def hip_extension_peak(self):
        return int(np.argmax(self.hip_velocity()))

    def turnover_start(self):
        n = len(self.wrist_y)
        if n < 5:
            return 0

        # Ignore the static setup.
        search_start = int(n * 0.25)

        wrist = self.wrist_y[search_start:]
        vel = np.gradient(wrist)

        # Strongest upward wrist movement (smallest y velocity).
        return search_start + int(np.argmin(vel))

    def stabilization_point(self, start_idx=0):
        n = len(self.wrist_y)
        if n < 5:
            return 0

        # Don't allow stabilization during the setup.
        start_idx = max(start_idx, int(n * 0.30))
        start_idx = min(start_idx, n - 2)

        # Catch = highest wrist after turnover.
        return start_idx + int(np.argmin(self.wrist_y[start_idx:]))
