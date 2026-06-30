import numpy as np

class BottomCalibrator:
    """
    Learns systematic frame offset between predicted bottom and gold labels.
    """

    def __init__(self):
        self.errors = []

    def update(self, pred_frame, gold_frame):
        self.errors.append(pred_frame - gold_frame)

    def estimate_offset(self):
        if not self.errors:
            return 0
        return int(np.median(self.errors))

    def apply(self, pred_frame):
        return pred_frame - self.estimate_offset()