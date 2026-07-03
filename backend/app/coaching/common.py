GOOD = "good"
WARNING = "warning"


def section(title, state, good_text, warning_text):
    return {
        "title": title,
        "status": GOOD if state == "good" else WARNING,
        "message": good_text if state == "good" else warning_text,
    }


def build_priority(sections):
    """
    Return the first warning.
    """

    for s in sections:
        if s["status"] == WARNING:
            return s["message"]

    return "Excellent technique. Keep building consistency."