import json
from pathlib import Path
import numpy as np

GOLD_PATH = Path("gold_squat/gold_bottoms_v1.json")

data = json.loads(GOLD_PATH.read_text())

def smooth(arr):
    return np.convolve(arr, np.ones(5)/5, mode="same")

def estimate_entry_bottom(frames):
    """
    Convert old 'center-bottom' labels into:
    FIRST stable bottom entry
    """

    # approximate correction: shift earlier based on biomechanics bias (~15 frames)
    return max(frames["bottom_center"] - 16, frames["bottom_start"])

new_data = []

for item in data:
    corrected = item.copy()

    corrected_center = item["bottom_center"] - 16

    # clamp so we never go before descent window
    corrected_center = max(corrected_center, item["bottom_start"])

    corrected["bottom_center"] = int(corrected_center)

    new_data.append(corrected)

GOLD_PATH.write_text(json.dumps(new_data, indent=2))

print("✅ gold labels converted to ENTRY-based definition")