from .common import section, build_priority


def build_split_jerk_coaching(rep):

    b = rep.get("breakdown", {})

    sections = [

        section(
            "Dip",
            b.get("dip", "good"),
            "Strong vertical dip.",
            "Keep the dip straight and balanced.",
        ),

        section(
            "Drive",
            b.get("drive", "good"),
            "Powerful leg drive.",
            "Finish driving through the legs before pressing.",
        ),

        section(
            "Split Catch",
            b.get("split_catch", "good"),
            "Excellent receiving position.",
            "Create a longer, more stable split position.",
        ),

        section(
            "Lockout",
            b.get("lockout", "good"),
            "Strong overhead lockout.",
            "Punch aggressively into lockout.",
        ),

        section(
            "Torso",
            b.get("torso_stack", "good"),
            "Good stacked overhead position.",
            "Keep the ribs down and torso stacked.",
        ),

        section(
            "Bar Path",
            b.get("bar_path", "good"),
            "Bar stayed over mid-foot.",
            "Keep the bar moving vertically.",
        ),
    ]

    return {
        "priority": build_priority(sections),
        "sections": sections,
    }