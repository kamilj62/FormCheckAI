from .common import section, build_priority


def build_squat_coaching(rep, exercise_label="squat_back"):
    b = rep.get("breakdown", {})

    sections = [
        section("Depth", b.get("depth", "good"),
                "Good squat depth.",
                "Sink a little deeper while keeping your chest up."),
        section("Torso", b.get("torso", "good"),
                "Strong torso position.",
                "Stay braced and keep your chest proud."),
        section("Knees", b.get("knees", "good"),
                "Knees track well over the toes.",
                "Drive knees out over your toes."),
        section("Heels", b.get("heels", "good"),
                "Good foot pressure.",
                "Keep your heels planted and drive through midfoot."),
        section("Neck", b.get("neck", "good"),
                "Good neutral head position.",
                "Keep your head aligned with your torso."),
    ]

    if exercise_label == "squat_front":
        sections.extend([
            section("Front Rack", b.get("front_rack", "good"),
                    "Strong front-rack position.",
                    "Drive elbows higher to keep the bar secure."),
            section("Bar Position", b.get("bar_position", "good"),
                    "Bar stays secure over the shoulders.",
                    "Keep the bar close to your throat and elbows pointed forward."),
        ])

    return {
        "priority": build_priority(sections),
        "sections": sections,
    }


def build_squat_metrics(rep):
    b = rep.get("breakdown", {})

    def grade_value(key):
        status = b.get(key, "good")
        if status == "good":
            return 1.0
        if status == "borderline":
            return 0.5
        if status == "poor":
            return 0.2
        return 0.5

    metrics = {
        "depth": grade_value("depth"),
        "torso": grade_value("torso"),
        "knees": grade_value("knees"),
        "heels": grade_value("heels"),
        "neck": grade_value("neck"),
    }

    if "front_rack" in b:
        metrics["front_rack"] = grade_value("front_rack")
    if "bar_position" in b:
        metrics["bar_position"] = grade_value("bar_position")

    return metrics
