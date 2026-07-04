from .common import section, build_priority

def build_overhead_squat_coaching(rep):
    b = rep.get("breakdown", {})

    sections = [
        section(
            "Overhead Stability",
            b.get("overhead", "good"),
            "Bar is stable overhead.",
            "Lock bar directly over midfoot and stay stacked."
        ),
        section(
            "Depth",
            b.get("depth", "good"),
            "Strong squat depth.",
            "Reach full depth while maintaining overhead stability."
        ),
        section(
            "Bar Stack",
            b.get("bar_path", "good"),
            "Bar remains stacked over midfoot.",
            "Prevent forward drift — keep bar over midfoot."
        ),
        section(
            "Knees",
            b.get("knees", "good"),
            "Good knee tracking.",
            "Drive knees out and stay stable under overhead load."
        ),
    ]

    return {
        "priority": build_priority(sections),
        "sections": sections,
    }