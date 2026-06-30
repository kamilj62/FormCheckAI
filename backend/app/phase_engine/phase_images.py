def extract_squat_phase_frames(records, start, bottom, end):

    def mid(a, b):
        return int((a + b) / 2)

    return {
        "setup": records[start],
        "descent": records[mid(start, bottom)],
        "bottom": records[bottom],
        "ascent": records[mid(bottom, end)],
        "lockout": records[end]
    }