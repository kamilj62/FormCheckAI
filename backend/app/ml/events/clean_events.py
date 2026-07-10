from typing import Any

import numpy as np

from .timeline import EventTimeline


def detect_clean_events(
    biomechanics: list[dict[str, Any]],
    frames: dict[str, Any],
) -> EventTimeline:
    """
    Clean event detector V2 in shadow mode.

    Production output is unchanged. This detector exposes true extension
    separately from pull-under so the results can be evaluated first.
    """

    if not biomechanics:
        return EventTimeline()

    frame_numbers = np.array([
        int(b.get("frame_number", i))
        for i, b in enumerate(biomechanics)
    ])

    hip = np.array([
        float(b.get("hip_angle", 180.0))
        for b in biomechanics
    ], dtype=np.float32)

    knee = np.array([
        float(b.get("knee_angle", 180.0))
        for b in biomechanics
    ], dtype=np.float32)

    def frame_to_idx(frame: int | None, default: int) -> int:
        if frame is None or len(frame_numbers) == 0:
            return default

        return int(np.argmin(np.abs(frame_numbers - int(frame))))

    start_idx = frame_to_idx(
        frames.get("start_frame"),
        0,
    )
    catch_idx = frame_to_idx(
        frames.get("catch_frame"),
        len(frame_numbers) - 1,
    )
    recovery_idx = frame_to_idx(
        frames.get("end_frame"),
        len(frame_numbers) - 1,
    )

    start_idx = max(0, min(start_idx, len(frame_numbers) - 1))
    catch_idx = max(start_idx + 1, min(catch_idx, len(frame_numbers) - 1))
    recovery_idx = max(catch_idx, min(recovery_idx, len(frame_numbers) - 1))

    pre_start = max(
        start_idx + 1,
        int(start_idx + (catch_idx - start_idx) * 0.35),
    )
    pre_end = max(pre_start + 1, catch_idx)

    extension_score = hip[pre_start:pre_end] + knee[pre_start:pre_end]

    if len(extension_score):
        extension_idx = pre_start + int(np.argmax(extension_score))
    else:
        extension_idx = max(start_idx + 1, catch_idx - 2)

    extension_idx = max(
        start_idx + 1,
        min(extension_idx, catch_idx - 1),
    )

    first_pull_idx = frame_to_idx(
        frames.get("first_pull_frame"),
        max(start_idx + 1, extension_idx - 1),
    )
    first_pull_idx = max(
        start_idx + 1,
        min(first_pull_idx, extension_idx),
    )

    # Pull-under must happen after full extension and before the catch.
    # Do not reuse the legacy extension_frame here because that field can
    # occur before true extension and collapse the two phase images.
    extension_to_catch = max(1, catch_idx - extension_idx)

    pull_under_idx = extension_idx + max(
        2,
        int(extension_to_catch * 0.45),
    )

    pull_under_idx = max(
        extension_idx + 1,
        min(pull_under_idx, catch_idx - 1),
    )

    transition_idx = max(
        first_pull_idx,
        int(first_pull_idx + (extension_idx - first_pull_idx) * 0.50),
    )

    power_position_idx = max(
        transition_idx,
        int(transition_idx + (extension_idx - transition_idx) * 0.65),
    )

    return EventTimeline(
        setup=int(frame_numbers[start_idx]),
        first_pull=int(frame_numbers[first_pull_idx]),
        transition=int(frame_numbers[transition_idx]),
        power_position=int(frame_numbers[power_position_idx]),
        extension=int(frame_numbers[extension_idx]),
        pull_under=int(frame_numbers[pull_under_idx]),
        catch=int(frame_numbers[catch_idx]),
        recovery=int(frame_numbers[recovery_idx]),
        confidence=0.50,
    )
