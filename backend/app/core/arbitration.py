def arbitrate(candidates, biomechanics):
    best = max(candidates, key=lambda c: c.get("confidence", 0))

    return {
        "label": best["label"],
        "confidence": float(best["confidence"])
    }