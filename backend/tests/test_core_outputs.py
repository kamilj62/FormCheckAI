from app.main import build_set_summary, build_coaching_zones


def test_squat_knee_valgus_priority_over_depth():
    rep_feedback = [
        {
            "rep": 1,
            "score": 8.2,
            "breakdown": {
                "depth": "borderline",
                "torso": "good",
                "knees": "poor",
                "heels": "good",
                "neck": "good",
            },
            "feedback": [
                "Sink a little deeper while keeping your chest up.",
                "Drive knees out over your toes.",
            ],
        }
    ]

    summary = build_set_summary(rep_feedback)
    zones = build_coaching_zones("Squat", rep_feedback)

    assert summary["biggest_fix"] == "Drive knees out over your toes."
    assert zones["knees"]["status"] == "needs_work"
    assert zones["knees"]["message"] == "Drive knees out over your toes."
    assert zones["depth"]["status"] == "needs_work"


def test_summary_empty_reps():
    summary = build_set_summary([])

    assert summary["detected_reps"] == 0
    assert summary["avg_rep_score"] == 0
    assert summary["best_rep"] is None
    assert summary["worst_rep"] is None


def test_summary_best_and_worst_rep():
    rep_feedback = [
        {"rep": 1, "score": 7.0, "breakdown": {}, "feedback": []},
        {"rep": 2, "score": 9.0, "breakdown": {}, "feedback": []},
        {"rep": 3, "score": 8.0, "breakdown": {}, "feedback": []},
    ]

    summary = build_set_summary(rep_feedback)

    assert summary["detected_reps"] == 3
    assert summary["avg_rep_score"] == 8.0
    assert summary["best_rep"] == 2
    assert summary["worst_rep"] == 1


def test_squat_good_zones():
    rep_feedback = [
        {
            "rep": 1,
            "score": 9.8,
            "breakdown": {
                "depth": "good",
                "torso": "good",
                "knees": "good",
                "heels": "good",
                "neck": "good",
            },
            "feedback": ["Strong rep."],
        }
    ]

    zones = build_coaching_zones("Squat", rep_feedback)

    assert zones["knees"]["status"] == "good"
    assert zones["depth"]["status"] == "good"
    assert zones["torso"]["status"] == "good"
    assert zones["heels"]["status"] == "good"
    assert zones["neck"]["status"] == "good"


def test_deadlift_torso_priority():
    rep_feedback = [
        {
            "rep": 1,
            "score": 6.0,
            "breakdown": {
                "back": "poor",
                "lockout": "good",
                "knees": "good",
                "hinge": "good",
                "neck": "good",
            },
            "feedback": ["Brace your core and keep a neutral spine."],
        }
    ]

    zones = build_coaching_zones("Deadlift", rep_feedback)

    assert zones["torso"]["status"] == "needs_work"
    assert zones["torso"]["message"] == "Keep your back flat and brace your core."


def test_bench_press_lockout_zone():
    rep_feedback = [
        {
            "rep": 1,
            "score": 8.0,
            "breakdown": {
                "depth": "good",
                "lockout": "incomplete",
                "elbows": "good",
                "arch": "controlled",
            },
            "feedback": ["Fully extend your arms at the top."],
        }
    ]

    zones = build_coaching_zones("Bench Press", rep_feedback)

    assert zones["lockout"]["status"] == "needs_work"
    assert zones["lockout"]["message"] == "Fully extend your arms at the top."


def test_push_press_lockout_zone():
    rep_feedback = [
        {
            "rep": 1,
            "score": 8.0,
            "breakdown": {
                "dip": "good",
                "drive": "good",
                "lockout": "incomplete",
                "core": "good",
            },
            "feedback": ["Finish with arms locked out overhead."],
        }
    ]

    zones = build_coaching_zones("Push Press", rep_feedback)

    assert zones["lockout"]["status"] == "needs_work"
    assert zones["lockout"]["message"] == "Fully extend arms overhead."