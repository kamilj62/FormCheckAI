from dataclasses import dataclass

@dataclass
class CropWindow:
    x1: int
    y1: int
    x2: int
    y2: int

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1


class CropTracker:

    def __init__(self):
        self.window = None

    def initialize(self, frame_width, frame_height):
        # Good default centered crop.
        self.window = CropWindow(
            int(frame_width * 0.18),
            int(frame_height * 0.05),
            int(frame_width * 0.72),
            frame_height,
        )

    def update(self, landmarks, frame_width, frame_height):

        if landmarks is None:
            return self.window

        xs = [p.x for p in landmarks.landmark]
        ys = [p.y for p in landmarks.landmark]

        x1 = int(max(0, min(xs) * frame_width - 80))
        y1 = int(max(0, min(ys) * frame_height - 80))

        x2 = int(min(frame_width, max(xs) * frame_width + 80))
        y2 = int(min(frame_height, max(ys) * frame_height + 120))

        # Smooth crop movement.
        a = 0.15

        self.window.x1 = int(self.window.x1 * (1-a) + x1 * a)
        self.window.y1 = int(self.window.y1 * (1-a) + y1 * a)
        self.window.x2 = int(self.window.x2 * (1-a) + x2 * a)
        self.window.y2 = int(self.window.y2 * (1-a) + y2 * a)

        return self.window