GOOD = "good"
WARNING = "warning"
POOR = "poor"

POOR_STATES = {
    "poor",
    "severe_flare",
    "limited_range",
    "incomplete",
    "drifting",
    "shallow",
    "high",
    "soft",
    "weak",
    "slow",
    "sagging",
    "knee_cave",
    "leaning",
    "leg_drive",
    "excessive",
    "stiff",
    "short",
    "severe",
    "off",
    "disconnected",
    "early_press",
    "bent",
}

WARNING_STATES = {
    "borderline",
    "needs_work",
    "possible",
    "fair",
    "possibly_shallow",
    "review",
    "minor_knee_bend",
    "leaning_back",
    "leaning_forward",
    "unknown",
    "controlled",
}


def section(title, state, good_text, warning_text):
    normalized = str(state or "good").lower()

    if normalized == "good":
        status = GOOD
        message = good_text
    elif normalized in POOR_STATES:
        status = POOR
        message = warning_text
    elif normalized in WARNING_STATES:
        status = WARNING
        message = warning_text
    else:
        status = WARNING
        message = warning_text

    return {
        "title": title,
        "status": status,
        "message": message,
    }


def build_priority(sections):
    for s in sections:
        if s["status"] == POOR:
            return s["message"]

    for s in sections:
        if s["status"] == WARNING:
            return s["message"]

    return "Excellent technique. Keep building consistency."
