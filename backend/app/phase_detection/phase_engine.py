# backend/app/phase_detection/phase_engine.py

from .signal_engine import SignalEngine

OVERLAY_DIR = "outputs"


def get_phase_images(label, video_path, biomechanics):
    """
    SignalEngine-based phase extraction.
    No configs. No handlers. No imports of missing modules.
    """

    if not biomechanics or len(biomechanics) < 5:
        return None

    engine = SignalEngine(biomechanics)

    n = len(biomechanics)

    # -------------------------
    # UNIVERSAL SIGNAL EVENTS
    # -------------------------

    setup = 0

    # Clean / Olympic extension peak
    clean_catch = engine.extension_peak()

    # Pull under / turnover moment
    jerk_dip = engine.turnover_start()

    # Stabilization after catch
    jerk_catch = engine.stabilization_point(jerk_dip)

    # End frame
    finish = n - 1

    # -------------------------
    # RETURN PHASE MAP
    # -------------------------

    if label == "thruster":
        knee = [b.get("knee_angle", 180) for b in biomechanics]

        bottom = int(min(range(len(knee)), key=lambda i: knee[i]))

        descent = max(0, bottom // 2)

        lockout = finish

        drive = min(
            lockout,
            bottom + max(1, (lockout - bottom) // 2)
        )

        return {
            "setup": setup,
            "descent": descent,
            "bottom": bottom,
            "drive": drive,
            "lockout": lockout,
        }

    return {
        "setup": setup,
        "clean_catch": clean_catch,
        "jerk_dip": jerk_dip,
        "jerk_catch": jerk_catch,
        "finish": finish,
    }
