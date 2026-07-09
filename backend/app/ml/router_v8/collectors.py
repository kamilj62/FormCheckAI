from .models import RouterPrediction


def collect_predictions(
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
    Build a list of router predictions.

    This module DOES NOT decide anything.
    It simply collects opinions from every router.
    """

    preds = []

    preds.append(
        RouterPrediction(
            router="base",
            label=raw_label,
            confidence=float(raw_conf or 0.0),
        )
    )

    preds.append(
        RouterPrediction(
            router="biomechanics",
            label=bio_label,
            confidence=float(bio_conf or 0.0),
        )
    )

    if squat_label:
        preds.append(
            RouterPrediction(
                router="squat",
                label=squat_label,
                confidence=float(squat_conf or 0.0),
            )
        )

    if olympic_label:
        preds.append(
            RouterPrediction(
                router="olympic",
                label=olympic_label,
                confidence=float(olympic_conf or 0.0),
            )
        )

    if bodyweight_label:
        preds.append(
            RouterPrediction(
                router="bodyweight",
                label=bodyweight_label,
                confidence=float(bodyweight_conf or 0.0),
            )
        )

    return preds
