# backend/app/phase_detection/phase_config.py

PHASE_CONFIG = {
    # -------------------------
    # OLYMPIC LIFTS
    # -------------------------
    "clean_and_jerk": {"handler": "olympic"},
    "snatch": {"handler": "olympic"},
    "clean": {"handler": "olympic"},
    "jerk": {"handler": "olympic"},
    "split_jerk": {"handler": "olympic"},

    # -------------------------
    # SQUATS
    # -------------------------
    "squat": {"handler": "squat"},
    "squat_back": {"handler": "squat"},
    "squat_front": {"handler": "squat"},
    "overhead_squat": {"handler": "squat"},

    # -------------------------
    # DEADLIFT
    # -------------------------
    "deadlift": {"handler": "deadlift"},

    # -------------------------
    # PRESSING
    # -------------------------
    "push_press": {"handler": "press"},
    "strict_press": {"handler": "press"},
    "thruster": {"handler": "press"},
    "overhead_press": {"handler": "press"},

    # -------------------------
    # BENCH
    # -------------------------
    "bench_press": {"handler": "bench"},

    # -------------------------
    # PULLING / BODYWEIGHT
    # -------------------------
    "pull_up": {"handler": "bodyweight"},
    "chin_up": {"handler": "bodyweight"},
    "muscle_up": {"handler": "bodyweight"},
    "bar_muscle_up": {"handler": "bodyweight"},
    "ring_muscle_up": {"handler": "bodyweight"},

    # -------------------------
    # PUSH UPS
    # -------------------------
    "push_up": {"handler": "bodyweight"},
    "handstand_push_up": {"handler": "bodyweight"},

    # -------------------------
    # CONDITIONING
    # -------------------------
    "burpee": {"handler": "bodyweight"},
}