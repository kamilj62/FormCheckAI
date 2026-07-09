from .models import RouterPrediction


def build_debug(predictions, fusion_result):
    """
    Build a clean Router V8 debug payload.

    Parameters
    ----------
    predictions : list[RouterPrediction]
    fusion_result : dict

    Returns
    -------
    dict
    """

    return {
        "version": "router_v8_shadow",
        "predictions": [
            {
                "router": p.router,
                "label": p.label,
                "confidence": round(float(p.confidence), 3),
                "reason": p.reason,
                "lock": p.lock,
            }
            for p in predictions
        ],
        "scores": fusion_result.get("scores", {}),
        "winner": fusion_result.get("label"),
        "winner_confidence": round(
            float(fusion_result.get("confidence", 0.0)),
            3,
        ),
        "evidence": fusion_result.get("evidence", {}),
    }
