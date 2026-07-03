from .common import section, build_priority


def build_clean_coaching(rep):

    b = rep.get("breakdown", {})

    sections = [

        section(
            "First Pull",
            b.get("first_pull"),
            "Strong first pull off the floor.",
            "Stay balanced over the mid-foot during the first pull.",
        ),

        section(
            "Extension",
            b.get("extension"),
            "Excellent extension.",
            "Finish extending the hips before bending the arms.",
        ),

        section(
            "Turnover",
            b.get("turnover"),
            "Fast elbows into the rack.",
            "Rotate the elbows faster into the front rack.",
        ),

        section(
            "Catch",
            b.get("catch"),
            "Stable receiving position.",
            "Receive the bar with a stronger front rack.",
        ),
    ]

    return {
        "priority": build_priority(sections),
        "sections": sections,
    }