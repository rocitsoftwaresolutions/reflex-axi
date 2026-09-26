from typing import Any


def metrics(samples: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    labeled = [(r, o) for r, o in samples if o.get("expected") in r["distribution"]]
    brier = (
        sum(
            sum((p - float(label == o["expected"])) ** 2 for label, p in r["distribution"].items())
            for r, o in labeled
        )
        / len(labeled)
        if labeled
        else None
    )
    ece = 0.0
    for bucket in range(10):
        group = [(r, o) for r, o in labeled if min(int(r["confidence"] * 10), 9) == bucket]
        if group:
            confidence = sum(r["confidence"] for r, _ in group) / len(group)
            accuracy = sum(r["selected"] == o["expected"] for r, o in group) / len(group)
            ece += len(group) / len(labeled) * abs(confidence - accuracy)
    latency = [r["execution"].get("source_latency_ms", r["latency_ms"]) for r, _ in samples]
    costs = [r["execution"].get("source_cost", r["cost"]) for r, _ in samples]
    return {
        "samples": len(samples),
        "labeled": len(labeled),
        "brier": brier,
        "ece": ece if labeled else None,
        "mean_latency_ms": sum(latency) / len(latency) if latency else None,
        "mean_cost": sum(costs) / len(costs)
        if costs and all(c is not None for c in costs)
        else None,
    }
