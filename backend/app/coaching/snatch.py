from .common import section, build_priority


def build_snatch_coaching(rep):

    b = rep.get("breakdown", {})

    sections = [

        section(
            "First Pull",
            b.get("first_pull", "good"),
            "Strong first pull from the floor.",
            "Stay balanced and keep the chest over the bar longer.",
        ),

        section(
            "Extension",
            b.get("extension", "good"),
            "Excellent full extension.",
            "Finish extending before pulling under the bar.",
        ),

        section(
            "Turnover",
            b.get("turnover", "good"),
            "Fast pull under the bar.",
            "Pull yourself under the bar more aggressively.",
        ),

        # Use the analyzer's actual key
        section(
            "Overhead Catch",
            b.get("overhead_catch", "good"),
            "Stable receiving position.",
            "Receive the bar in a stronger overhead position.",
        ),

        # Use the analyzer's actual key
        section(
            "Stability",
            b.get("stability", "good"),
            "Excellent overhead stability.",
            "Stabilize the overhead position before standing.",
        ),

        section(
            "Bar Path",
            b.get("bar_path", "good"),
            "Bar stayed close throughout the lift.",
            "Keep the bar closer to your body during the pull.",
        ),
    ]

    return {
        "priority": build_priority(sections),
        "sections": sections,
    }