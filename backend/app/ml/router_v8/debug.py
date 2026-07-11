from .models import RouterPrediction


def build_debug(
    predictions: list[RouterPrediction],
    fusion_result: dict,
) -> dict:
    return {
        "version": "router_v8_hierarchical_shadow",
        "predictions": [
            {
                "router": prediction.router,
                "label": prediction.label,
                "confidence": round(
                    float(prediction.confidence or 0.0),
                    3,
                ),
                "reason": prediction.reason,
                "lock": prediction.lock,
            }
            for prediction in predictions
        ],
        "decision": fusion_result.get("decision"),
        "winning_family": fusion_result.get("winning_family"),
        "family_scores": fusion_result.get("family_scores", {}),
        "scores": fusion_result.get("scores", {}),
        "winner": fusion_result.get("label"),
        "winner_confidence": round(
            float(fusion_result.get("confidence", 0.0)),
            3,
        ),
        "locks": fusion_result.get("locks", []),
        "selected_lock": fusion_result.get("lock"),
        "evidence": fusion_result.get("evidence", {}),
    }
