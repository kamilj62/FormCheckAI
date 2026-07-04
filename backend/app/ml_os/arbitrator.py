def arbitrate(results):

    candidates = []

    for _, r in results.items():
        if isinstance(r, dict) and "label" in r:
            candidates.append(r)

    def score(c):
        if c["source"] == "oly":
            return c["confidence"] * 1.3
        if c["source"] == "lstm":
            return c["confidence"] * 1.0
        if c["source"] == "rules":
            return c["confidence"] * 0.8
        return c["confidence"]

    return max(candidates, key=score)