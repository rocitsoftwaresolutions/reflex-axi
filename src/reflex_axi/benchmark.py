"""Versioned provider experiments, isolated from deployment and learning records."""

import importlib.metadata
import os
import platform
import resource
import threading
import time
import uuid
from contextlib import suppress
from importlib.resources import files
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import builtin_packs
from .engine import Engine
from .errors import ReflexError
from .identity import canonical, digest, read_json
from .metrics import metrics
from .models import Binding, ExecutionPolicy, Pack, Policy, Prediction
from .providers import Provider, Request, provider_from_spec, validate_predictions
from .store import Store

BENCHMARK_VERSION = "1"
SUITES = {
    "starter": "Every starter use case, with positive and negative labeled judgments",
    "robustness": "Negation, contradiction, injection, OOD, paraphrases, long and noisy text",
    "boundaries": "Ambiguity, confidence bands and policy:none",
    "conformance": "Invalid schema, normalization, partial failure and stable association",
}
MODES = ("single", "batch", "bundle")


def corpus() -> dict[str, Any]:
    return read_json(files("reflex_axi").joinpath("benchmarks/agentic-v1.json").read_text())


def suites() -> dict[str, Any]:
    data = corpus()
    return {
        "version": BENCHMARK_VERSION,
        "corpus_version": data["version"],
        "count": len(SUITES),
        "suites": [
            {"id": k, "cases": sum(c["suite"] == k for c in data["cases"]), "description": v}
            for k, v in SUITES.items()
        ],
        "help": [
            "reflex-axi benchmark run --provider mock --suite starter",
            "reflex-axi benchmark run --help",
        ],
    }


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "reflex": __version__,
        "implementation_hash": digest(
            {
                p.name: digest(p.read_text())
                for p in files("reflex_axi").iterdir()
                if p.name.endswith(".py")
            }
        ),
        "packages": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            )
        ),
    }


def footprint(binding: Binding) -> dict[str, Any]:
    """Observable file sizes, not a claim about allocated pages or remote infrastructure."""
    model_bytes = None
    runtime_bytes = None
    if binding.provider.adapter == "gliner":
        root = Path(binding.provider.settings["model_path"])
        with suppress(OSError):
            model_bytes = sum(
                (root / name).stat().st_size for name in binding.provider.settings["files"]
            )
        try:
            paths = {
                Path(str(d.locate_file(f)))
                for name in binding.provider.settings["runtime"]
                for d in [importlib.metadata.distribution(name)]
                for f in (d.files or [])
            }
            runtime_bytes = sum(p.stat().st_size for p in paths if p.is_file())
        except (OSError, importlib.metadata.PackageNotFoundError):
            pass
    return {
        "model_file_bytes": model_bytes,
        "pinned_runtime_file_bytes": runtime_bytes,
        "note": "Logical file sizes; exclude download caches, unpinned transitive dependencies and filesystem allocation overhead.",
    }


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


class ObservedProvider:
    """Measure actual calls, including native batching, without altering predictions."""

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self.spec = provider.spec
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.inflight = 0

    def infer(self, requests: list[Request]) -> dict[str, Prediction]:
        start = time.perf_counter()
        trace: dict[str, Any] = {
            "requests": [r.id for r in requests],
            "started": time.time(),
            "count": len(requests),
        }
        with self.lock:
            self.inflight += 1
            trace["inflight_at_start"] = self.inflight
        try:
            predictions = self.provider.infer(requests)
            trace["raw"] = {k: v.model_dump() for k, v in predictions.items()}
            validate_predictions(requests, predictions)
            trace["valid"] = True
            return predictions
        except Exception as error:
            safe = (
                error
                if isinstance(error, ReflexError)
                else ReflexError(
                    "provider_response", "provider failed or returned an invalid response"
                )
            )
            trace["error"] = safe.code
            trace["valid"] = False
            # Invalid raw data may be non-JSON. Do not persist exception text or unsafe payloads.
            trace.pop("raw", None)
            raise safe from None
        finally:
            trace["latency_ms"] = (time.perf_counter() - start) * 1000
            with self.lock:
                self.inflight -= 1
                self.calls.append(trace)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import math

    samples = [(r["result"], r["case"]) for r in rows if "result" in r]
    labeled = [(r, c) for r, c in samples if c.get("expected") is not None]
    report = metrics(samples)
    report.pop("mean_latency_ms")
    report.pop("mean_cost")
    report["accuracy"] = mean([float(r["selected"] == c["expected"]) for r, c in labeled])
    report["log_loss"] = mean(
        [-math.log(max(r["distribution"][c["expected"]], 1e-15)) for r, c in labeled]
    )
    score_errors = []
    for row in rows:
        if (
            "result" in row
            and row["case"].get("expected")
            and row["pack"]["judgment"]["primitive"] == "score"
        ):
            score_errors.append(
                abs(
                    row["result"]["expected_score"]
                    - row["pack"]["judgment"]["values"][row["case"]["expected"]]
                )
            )
    report["score_mae"] = mean(score_errors)
    report["selective"] = []
    for threshold in [0.5, 0.8, 0.95]:
        accepted = [(r, c) for r, c in labeled if r["confidence"] >= threshold]
        ambiguous = [(r, c) for r, c in samples if c.get("abstain")]
        report["selective"].append(
            {
                "threshold": threshold,
                "coverage": len(accepted) / len(labeled) if labeled else None,
                "risk": mean([float(r["selected"] != c["expected"]) for r, c in accepted]),
                "ambiguity_escalation_rate": mean(
                    [float(r["confidence"] < threshold) for r, _ in ambiguous]
                ),
                # Explicit utility schedule: correct=1, wrong=-1, defer=0. No authority implied.
                "utility": mean(
                    [
                        0.0
                        if r["confidence"] < threshold
                        else (1.0 if r["selected"] == c["expected"] else -1.0)
                        for r, c in labeled
                    ]
                ),
            }
        )
    return report


def summarize(rows: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [call for chunk in chunks for call in chunk["calls"]]
    valid = [r for r in rows if "result" in r]
    checks = [
        float((r.get("error") == r["case"]["error"]) if r["case"].get("error") else "result" in r)
        for r in rows
    ]
    errors: dict[str, int] = {}
    for r in rows:
        if "error" in r:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    comparisons: dict[str, list[float]] = {"repeat": [], "execution_mode": [], "paraphrase": []}
    distances: dict[str, list[float]] = {k: [] for k in comparisons}
    lookup = {(r["case"]["id"], r["mode"], r["repeat"]): r for r in valid}
    for row in valid:
        candidates = [
            ("repeat", (row["case"]["id"], row["mode"], 0)) if row["repeat"] else None,
            ("execution_mode", (row["case"]["id"], "single", row["repeat"]))
            if row["mode"] != "single"
            else None,
            ("paraphrase", (row["case"]["pair"], row["mode"], row["repeat"]))
            if row["case"].get("pair")
            else None,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            kind, key = candidate
            if key in lookup:
                a, b = row["result"], lookup[key]["result"]
                comparisons[kind].append(float(a["selected"] == b["selected"]))
                distances[kind].append(
                    sum(abs(a["distribution"][k] - b["distribution"][k]) for k in a["distribution"])
                    / 2
                )
    elapsed = sum(c["wall_ms"] for c in chunks) / 1000
    costs = [p["cost"] for c in calls for p in c.get("raw", {}).values()]
    usage: dict[str, float] = {}
    for call in calls:
        for prediction in call.get("raw", {}).values():
            for k, v in prediction["usage"].items():
                usage[k] = usage.get(k, 0.0) + v
    latencies = sorted(c["latency_ms"] for c in calls)
    return {
        "conformance": {
            "cases": len(rows),
            "pass_rate": mean(checks),
            "raw_distribution_and_association_validity": mean(
                [
                    float(c["valid"])
                    for c in calls
                    if c["valid"]
                    or c.get("error")
                    in {"provider_association", "provider_distribution", "provider_response"}
                ]
            ),
            "policy_safety": all(
                not r["result"].get("policy_result")
                or r["result"]["policy_result"]["executed"] is False
                for r in valid
            ),
            "consistency": {
                k: {
                    "pairs": len(v),
                    "agreement": mean(v),
                    "mean_total_variation": mean(distances[k]),
                }
                for k, v in comparisons.items()
            },
        },
        # Repeats/modes are correlated; primary quality uses each scenario once.
        "quality": quality([r for r in rows if r["primary"]]),
        "quality_by_suite": {
            s: quality([r for r in rows if r["primary"] and r["case"]["suite"] == s])
            for s in sorted({r["case"]["suite"] for r in rows})
        },
        "quality_by_pack": {
            pack: quality([r for r in rows if r["primary"] and r["case"]["pack"] == pack])
            for pack in sorted({r["case"]["pack"] for r in rows})
        },
        "performance": {
            "wall_seconds": elapsed,
            "provider_calls": len(calls),
            "provider_requests": sum(c["count"] for c in calls),
            "requests_per_second": sum(c["count"] for c in calls) / elapsed if elapsed else None,
            "mean_call_ms": mean([c["latency_ms"] for c in calls]),
            "call_p50_ms": latencies[(len(latencies) - 1) // 2] if latencies else None,
            "call_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
            if latencies
            else None,
            "call_max_ms": max(latencies) if latencies else None,
            "max_observed_inflight": max((c["inflight_at_start"] for c in calls), default=0),
            "max_native_batch": max((c["count"] for c in calls), default=0),
            "cold_call_ms": [
                c["latency_ms"]
                for chunk in chunks
                for c in chunk["calls"]
                if c.get("temperature") == "process-first"
            ],
            "warm_call_mean_ms": mean(
                [c["latency_ms"] for c in calls if c.get("temperature") == "warm"]
            ),
            "process_peak_rss_bytes": max((c["rss_bytes"] for c in chunks), default=0),
            "cost": sum(costs)
            if costs and all(c is not None for c in costs) and all(c["valid"] for c in calls)
            else None,
            "usage": usage,
            "usage_complete": bool(calls)
            and all(c.get("raw") and all(p["usage"] for p in c["raw"].values()) for c in calls),
        },
        "resilience": {
            "errors": errors,
            "expected_rejections": sum(
                bool(r["case"].get("error")) and r.get("error") == r["case"]["error"] for r in rows
            ),
            "unexpected_failures": sum(
                "error" in r and r.get("error") != r["case"].get("error") for r in rows
            ),
            "completed_cells": len(valid),
            "checkpoint_chunks": len(chunks),
            "provider_call_success_rate": mean([float(c["valid"]) for c in calls]),
        },
    }


class BenchmarkRunner:
    def __init__(self, store: Store, provider: Provider | None = None) -> None:
        self.store = store
        self.provider = provider

    def key(self, run_id: str) -> str:
        return self.store.project_key() + ":" + run_id

    def run(
        self,
        binding: Binding,
        *,
        selected: list[str] | None = None,
        modes: list[str] | None = None,
        repeats: int = 2,
        concurrency: int = 1,
    ) -> dict[str, Any]:
        selected = selected or list(SUITES)
        modes = modes or list(MODES)
        if (
            not set(selected) <= set(SUITES)
            or not set(modes) <= set(MODES)
            or len(set(modes)) != len(modes)
            or len(set(selected)) != len(selected)
        ):
            raise ReflexError("usage", "choose distinct known benchmark suites and modes")
        if not 1 <= repeats <= 20 or not 1 <= concurrency <= 32:
            raise ReflexError("usage", "repeats must be 1-20 and concurrency 1-32")
        data = corpus()
        packs = {p.id + "@" + p.version: p.model_dump(by_alias=True) for p in builtin_packs()}
        cases = [c for c in data["cases"] if c["suite"] in selected]
        # Binding checks happen before the first provider call, including calibration scope.
        for case in cases:
            binding.check(Pack.model_validate(packs[case["pack"]]), "benchmark", "benchmark")
        definition = {
            "benchmark_version": BENCHMARK_VERSION,
            "corpus_version": data["version"],
            "corpus_hash": digest(data),
            "cases": cases,
            "packs": packs,
            "binding": binding.model_dump(),
            "provider_hash": binding.provider.identity(),
            "suites": selected,
            "modes": modes,
            "repeats": repeats,
            "concurrency": concurrency,
            "environment": environment(),
            "footprint": footprint(binding),
            "protected": True,
            "learnable": False,
            "quality_label_source": data["description"],
            "cache": False,
            "project": self.store.project_key(),
        }
        run_id = uuid.uuid4().hex
        meta = {
            "id": run_id,
            "created": time.time(),
            "definition": definition,
            "definition_hash": digest(definition),
        }
        self.store.put("benchmark_runs", self.key(run_id), meta, immutable=True)
        return self.resume(run_id)

    def resume(self, run_id: str) -> dict[str, Any]:
        meta = self.store.require("benchmark_runs", self.key(run_id))
        final = self.store.get("benchmark_summaries", self.key(run_id))
        if final:
            return final
        d = meta["definition"]
        if (
            digest(d) != meta["definition_hash"]
            or d["environment"] != environment()
            or d["benchmark_version"] != BENCHMARK_VERSION
        ):
            raise ReflexError(
                "benchmark_identity", "run definition or runtime changed; start a new run"
            )
        owner = self.store.claim("benchmark:" + self.key(run_id), ttl=3600)
        if owner is None:
            raise ReflexError(
                "benchmark_busy", "run is already leased; resume after the active runner finishes"
            )
        try:
            binding = Binding.model_validate(d["binding"])
            observed = ObservedProvider(self.provider or provider_from_spec(binding.provider))
            engine = Engine(self.store, observed)
            session = uuid.uuid4().hex
            tasks = []
            for repeat in range(d["repeats"]):
                for mode in d["modes"]:
                    cells = [
                        {
                            "case": case,
                            "repeat": repeat,
                            "mode": mode,
                            "primary": repeat == 0 and mode == d["modes"][0],
                        }
                        for case in d["cases"]
                    ]
                    if mode == "single":
                        tasks.extend([[cell] for cell in cells])
                    elif mode == "batch":
                        for pack in sorted({c["case"]["pack"] for c in cells}):
                            tasks.append([c for c in cells if c["case"]["pack"] == pack])
                    else:
                        # All packs share a StateBuilder. Cross each selected state with
                        # every selected pack, retaining labels only for its authored pack.
                        refs = sorted({c["case"]["pack"] for c in cells})
                        bundle = []
                        for cell in cells:
                            for ref in refs:
                                case = cell["case"]
                                if ref != case["pack"]:
                                    case = {
                                        **case,
                                        "id": case["id"] + ":" + ref,
                                        "pack": ref,
                                        "expected": None,
                                        "pair": None,
                                        "abstain": False,
                                    }
                                bundle.append(
                                    {
                                        **cell,
                                        "case": case,
                                        "primary": cell["primary"] and ref == cell["case"]["pack"],
                                    }
                                )
                        tasks.extend([bundle[i : i + 64] for i in range(0, len(bundle), 64)])
            for index, cells in enumerate(tasks):
                bucket = "benchmark_chunks:" + self.key(run_id)
                if self.store.get(bucket, str(index)):
                    continue
                before = len(observed.calls)
                items = []
                for cell in cells:
                    pack = Pack.model_validate(d["packs"][cell["case"]["pack"]])
                    if cell["case"].get("policy_none"):
                        pack = pack.model_copy(update={"policy": Policy()})
                    items.append((pack, cell["case"]["state"]))
                start = time.perf_counter()
                results = engine.evaluate_many(
                    items,
                    binding,
                    execution=ExecutionPolicy(cache=False, workload="benchmark", scope="benchmark"),
                    decide=True,
                    record=False,
                    concurrency=d["concurrency"],
                )
                rows = []
                for i, (cell, (pack, _), result) in enumerate(
                    zip(cells, items, results, strict=True)
                ):
                    rows.append(
                        {
                            **cell,
                            "cell": f"{index}:{i}",
                            "pack": pack.model_dump(by_alias=True),
                            "protected": True,
                            **(
                                {"error": result.code}
                                if isinstance(result, ReflexError)
                                else {"result": result.model_dump()}
                            ),
                        }
                    )
                calls = observed.calls[before:]
                first = min(observed.calls, key=lambda c: c["started"]) if observed.calls else None
                for call in calls:
                    call["temperature"] = "process-first" if call is first else "warm"
                chunk = {
                    "index": index,
                    "rows": rows,
                    "calls": calls,
                    "wall_ms": (time.perf_counter() - start) * 1000,
                    "rss_bytes": rss_bytes(),
                    "time": time.time(),
                    "process_session": session,
                }
                with self.store.transaction() as db:
                    self._fence(run_id, owner, db)
                    self.store.put(bucket, str(index), chunk, db, immutable=True)
            chunks = sorted(
                self.store.list("benchmark_chunks:" + self.key(run_id)), key=lambda c: c["index"]
            )
            rows = [r for c in chunks for r in c["rows"]]
            summary = summarize(rows, chunks)
            final = {
                "id": run_id,
                "status": "PARTIAL" if summary["resilience"]["unexpected_failures"] else "COMPLETE",
                "definition_hash": meta["definition_hash"],
                "finished": time.time(),
                "summary": summary,
                "results_hash": digest(chunks),
                "artifact_bytes": len(canonical(chunks).encode()),
                "artifact": str(self.store.path),
                "protected": True,
                "learnable": False,
            }
            with self.store.transaction() as db:
                self._fence(run_id, owner, db)
                self.store.put("benchmark_summaries", self.key(run_id), final, db, immutable=True)
            return final
        finally:
            self.store.release("benchmark:" + self.key(run_id), owner)

    def _fence(self, run_id: str, owner: str, db: Any) -> None:
        if not self.store.owns("benchmark:" + self.key(run_id), owner, db):
            raise ReflexError("lease_lost", "benchmark lease expired; resume saved run")

    def show(self, run_id: str, *, full: bool = False) -> dict[str, Any]:
        meta = self.store.require("benchmark_runs", self.key(run_id))
        final = self.store.get("benchmark_summaries", self.key(run_id)) or {
            "id": run_id,
            "status": "INCOMPLETE",
        }
        if full:
            chunks = sorted(
                self.store.list("benchmark_chunks:" + self.key(run_id)), key=lambda c: c["index"]
            )
            return {**final, "metadata": meta, "chunks": chunks}
        return final

    def compare(self, a: str, b: str) -> dict[str, Any]:
        left, right = self.show(a, full=True), self.show(b, full=True)
        fields = ("benchmark_version", "corpus_hash", "packs", "cases", "modes", "repeats")
        mismatches = [
            k
            for k in fields
            if left["metadata"]["definition"][k] != right["metadata"]["definition"][k]
        ]
        if mismatches or "summary" not in left or "summary" not in right:
            raise ReflexError(
                "benchmark_comparison",
                "comparison requires completed runs with identical corpus, packs, modes and repeats",
            )
        return {
            "runs": [
                {
                    "id": r["id"],
                    "provider": r["metadata"]["definition"]["binding"]["provider"]["id"],
                    "summary": r["summary"],
                    "settings": r["metadata"]["definition"]["binding"]["provider"]["settings"],
                    "calibration": r["metadata"]["definition"]["binding"]["calibration"],
                    "concurrency": r["metadata"]["definition"]["concurrency"],
                }
                for r in [left, right]
            ],
            "environment_equal": left["metadata"]["definition"]["environment"]
            == right["metadata"]["definition"]["environment"],
            "note": "No aggregate ranking; assess each metric and operating condition separately.",
        }
