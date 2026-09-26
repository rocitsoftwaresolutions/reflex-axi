"""Measure the version path against interpreter startup and cache reuse by provider calls."""

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from reflex_axi.catalog import builtin_packs, default_binding
from reflex_axi.engine import Engine
from reflex_axi.providers import MockProvider
from reflex_axi.store import Store


def timing(command: list[str], n: int = 15) -> float:
    values = []
    for _ in range(n):
        start = time.perf_counter()
        subprocess.run(command, stdout=subprocess.DEVNULL, check=True)
        values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values)


def benchmark() -> dict:
    floor = timing([sys.executable, "-c", "print('0.1.0')"])
    version = timing([sys.executable, "-m", "reflex_axi.entry", "--version"])
    if version > floor * 3 + 5:
        raise AssertionError(
            f"Version path regressed: {version:.2f}ms vs {floor:.2f}ms interpreter floor"
        )
    Path(".scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=".scratch", prefix="benchmark-") as directory:
        store = Store(directory)
        binding = default_binding()
        provider = MockProvider(binding.provider)
        engine = Engine(store, provider)
        pack = next(p for p in builtin_packs() if p.id == "filter-context")
        items = [(pack, {"task": "relevance", "context": str(i)}) for i in range(256)]
        start = time.perf_counter()
        initial = engine.evaluate_many(items, binding)
        cold_ms = (time.perf_counter() - start) * 1000
        first_calls = provider.calls
        start = time.perf_counter()
        reused = engine.evaluate_many(items, binding)
        warm_ms = (time.perf_counter() - start) * 1000
        assert provider.calls == first_calls
        assert [r.inference_hash for r in initial] == [r.inference_hash for r in reused]
        assert all(r.execution["cache_hit"] for r in reused)
    return {
        "python": sys.version.split()[0],
        "version_samples": 15,
        "interpreter_median_ms": round(floor, 3),
        "version_median_ms": round(version, 3),
        "version_to_floor": round(version / floor, 3),
        "states": 256,
        "cold_batch_ms": round(cold_ms, 3),
        "warm_batch_ms": round(warm_ms, 3),
        "cold_provider_calls": first_calls,
        "additional_warm_provider_calls": provider.calls - first_calls,
        "cache_hits": 256,
        "note": "Local mock timings measure framework overhead, not live inference latency or semantic quality.",
    }


if __name__ == "__main__":
    result = benchmark()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
