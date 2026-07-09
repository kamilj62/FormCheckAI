from collections import defaultdict
from .models import RouterPrediction


def fuse_predictions(predictions: list[RouterPrediction]) -> dict:
    scores = defaultdict(float)
    evidence = defaultdict(list)

    for p in predictions:
        if not p.label:
            continue

        weight = 1.0

        if p.router == "base":
            weight = 1.00
        elif p.router == "biomechanics":
            weight = 0.90
        elif p.router == "squat":
            weight = 1.15
        elif p.router == "olympic":
            weight = 1.10
        elif p.router == "bodyweight":
            weight = 0.85

        score = float(p.confidence or 0.0) * weight
        scores[p.label] += score
        evidence[p.label].append({
            "router": p.router,
            "confidence": round(float(p.confidence or 0.0), 3),
            "weight": weight,
            "reason": p.reason,
        })

    if not scores:
        return {
            "label": None,
            "confidence": 0.0,
            "scores": {},
            "evidence": {},
        }

    winner = max(scores.items(), key=lambda kv: kv[1])[0]
    total = sum(scores.values()) or 1.0

    return {
        "label": winner,
        "confidence": min(1.0, scores[winner] / total),
        "scores": {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])},
        "evidence": dict(evidence),
    }
