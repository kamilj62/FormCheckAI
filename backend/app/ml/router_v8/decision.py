from .collectors import collect_predictions
from .fusion import fuse_predictions


def decide(
    *,
    raw_label=None,
    raw_conf=None,
    bio_label=None,
    bio_conf=None,
    squat_label=None,
    squat_conf=None,
    olympic_label=None,
    olympic_conf=None,
    bodyweight_label=None,
    bodyweight_conf=None,
):
    """
    Router V8 shadow decision.

    Returns:
        {
            "label": ...,
            "confidence": ...,
            "scores": ...,
            "evidence": ...
        }

    This function NEVER changes production routing.
    """

    predictions = collect_predictions(
        raw_label=raw_label,
        raw_conf=raw_conf,
        bio_label=bio_label,
        bio_conf=bio_conf,
        squat_label=squat_label,
        squat_conf=squat_conf,
        olympic_label=olympic_label,
        olympic_conf=olympic_conf,
        bodyweight_label=bodyweight_label,
        bodyweight_conf=bodyweight_conf,
    )

    return fuse_predictions(predictions)
