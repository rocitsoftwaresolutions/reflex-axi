import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from reflex_axi.benchmark import BenchmarkRunner, corpus, quality
from reflex_axi.catalog import builtin_packs
from reflex_axi.errors import ReflexError
from reflex_axi.evidence import Evidence
from reflex_axi.lifecycle import Lifecycle
from reflex_axi.models import Deployment, Prediction, Scope
from reflex_axi.providers import MockProvider


def test_cli_benchmark_workflow(cli, tmp_path):
    assert cli("benchmark", "list")["count"] == 4
    result = cli(
        "benchmark",
        "run",
        "--provider",
        "mock",
        "--suite",
        "starter",
        "--mode",
        "batch",
        "--repeats",
        "1",
    )
    assert result["status"] == "COMPLETE"
    full = cli("benchmark", "show", "--run", result["id"], "--full")
    assert full["summary"]["conformance"]["pass_rate"] == 1
    assert full["metadata"]["definition"]["binding"]["provider"]["adapter"] == "mock"
    assert full["summary"]["quality"]["labeled"] == 19
    assert full["summary"]["quality"]["accuracy"] < 1  # mock is not a semantic oracle
    assert cli("benchmark", "resume", "--run", result["id"])["id"] == result["id"]
    path = tmp_path / "run.json"
    cli("benchmark", "show", "--run", result["id"], "--full", "--output", str(path))
    assert json.loads(path.read_text())["results_hash"] == full["results_hash"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert cli("benchmark", "compare", "--run", result["id"], "--against", result["id"])[
        "environment_equal"
    ]
    cli("benchmark", "run", expected=2)
    cli("benchmark", "run", "--provider", "mock", "--concurrency", "0", expected=2)
    cli("benchmark", "resume", "--run", result["id"], "--provider", "mock", expected=2)
    cli("benchmark", "run", "--provider", "mock", "--suite", "typo", expected=2)


def test_corpus_coverage_and_labels():
    data = corpus()
    packs = {p.id + "@" + p.version: p for p in builtin_packs()}
    assert {c["pack"] for c in data["cases"]} == set(packs)
    assert len({c["id"] for c in data["cases"]}) == len(data["cases"])
    assert {p.judgment.primitive for p in packs.values()} == {"binary", "choice", "score"}
    for case in data["cases"]:
        if case["expected"]:
            assert case["expected"] in packs[case["pack"]].judgment.labels
        if not case.get("error"):
            packs[case["pack"]].state.build(case["state"])
    tags = {t for c in data["cases"] for t in c["tags"]}
    assert {
        "adversarial",
        "negated",
        "contradictory",
        "ood",
        "long",
        "short",
        "structured",
        "noisy",
        "policy:none",
        "paraphrase",
        "confidence-band",
    } <= tags


def test_benchmarks_are_observable_never_learnable_or_deploying(store, binding):
    pack = builtin_packs()[0]
    Lifecycle(store).activate(Deployment(pack=pack, binding=binding))
    Evidence(store).scope(Scope(name="benchmark", optimizable=True, allowed_to_mutate=True))
    before = {
        b: store.list(b)
        for b in ["deployments", "evidence", "decisions", "candidates", "datasets", "cache"]
    }
    events = store.events()
    run = BenchmarkRunner(store).run(binding)
    full = BenchmarkRunner(store).show(run["id"], full=True)
    assert full["metadata"]["definition"]["protected"]
    assert not full["learnable"]
    assert full["chunks"] and all(r["protected"] for c in full["chunks"] for r in c["rows"])
    assert full["summary"]["resilience"]["expected_rejections"] > 0
    assert full["summary"]["conformance"]["consistency"]["repeat"]["agreement"] == 1
    assert full["summary"]["conformance"]["consistency"]["execution_mode"]["agreement"] == 1
    assert any(
        len({r["pack"]["id"] for r in c["rows"]}) > 1
        for c in full["chunks"]
        if c["rows"][0]["mode"] == "bundle"
    )
    assert full["summary"]["performance"]["max_native_batch"] > 1
    for p in builtin_packs():
        assert Evidence(store).select(p.id, "optimizable", scopes=["benchmark"], mutate=True) == []
    assert before == {b: store.list(b) for b in before}
    assert store.events() == events


def test_resume_after_interrupt_preserves_checkpoints(store, binding):
    class Interrupting(MockProvider):
        def infer(self, requests):
            if self.calls == 2:
                raise KeyboardInterrupt
            return super().infer(requests)

    provider = Interrupting(binding.provider)
    runner = BenchmarkRunner(store, provider)
    with pytest.raises(KeyboardInterrupt):
        runner.run(binding, selected=["starter"], modes=["single"], repeats=1)
    meta = store.list("benchmark_runs")[0]
    before = store.list("benchmark_chunks:" + runner.key(meta["id"]))
    assert len(before) == 2
    result = BenchmarkRunner(store).resume(meta["id"])
    assert result["status"] == "COMPLETE"
    assert all(c in store.list("benchmark_chunks:" + runner.key(meta["id"])) for c in before)
    assert len(store.list("benchmark_runs")) == 1


def test_raw_failures_and_partial_results_are_reported(store, binding):
    class Bad(MockProvider):
        def infer(self, requests):
            if requests[0].pack.id == "filter-context":
                return {
                    r.id: Prediction(distribution={"false": 0.8, "true": 0.8}) for r in requests
                }
            return super().infer(requests)

    result = BenchmarkRunner(store, Bad(binding.provider)).run(
        binding, selected=["starter"], modes=["batch"], repeats=1
    )
    assert result["status"] == "PARTIAL"
    assert result["summary"]["resilience"]["errors"] == {"provider_distribution": 2}
    assert result["summary"]["resilience"]["completed_cells"] == 17
    assert result["summary"]["conformance"]["raw_distribution_and_association_validity"] < 1
    assert result["summary"]["performance"]["cost"] is None


def test_association_omissions_and_unexpected_errors_are_sanitized(store, binding):
    class Missing(MockProvider):
        def infer(self, requests):
            return {}

    result = BenchmarkRunner(store, Missing(binding.provider)).run(
        binding, selected=["starter"], modes=["batch"], repeats=1
    )
    assert result["summary"]["resilience"]["errors"] == {"provider_association": 19}
    assert result["summary"]["quality"]["accuracy"] is None


def test_concurrent_resume_is_leased(store, binding):
    class Interrupted(MockProvider):
        def infer(self, requests):
            raise KeyboardInterrupt

    runner = BenchmarkRunner(store, Interrupted(binding.provider))
    with pytest.raises(KeyboardInterrupt):
        runner.run(binding, selected=["starter"], modes=["batch"], repeats=1)
    run_id = store.list("benchmark_runs")[0]["id"]
    owner = store.claim("benchmark:" + runner.key(run_id))
    with pytest.raises(ReflexError, match="leased"):
        runner.resume(run_id)
    store.release("benchmark:" + runner.key(run_id), owner)
    with ThreadPoolExecutor(2) as pool:
        result = pool.submit(BenchmarkRunner(store).resume, run_id).result()
    assert result["status"] == "COMPLETE"


def test_quality_metrics_known_predictions():
    pack = builtin_packs()[0].model_dump(by_alias=True)
    rows = []
    for expected, selected in [("false", "false"), ("true", "false")]:
        rows.append(
            {
                "case": {"expected": expected},
                "pack": pack,
                "result": {
                    "distribution": {"false": 0.8, "true": 0.2},
                    "selected": selected,
                    "confidence": 0.8,
                    "execution": {},
                    "latency_ms": 1,
                    "cost": 0,
                },
            }
        )
    result = quality(rows)
    assert result["accuracy"] == 0.5
    assert result["brier"] == pytest.approx(0.68)
    assert result["ece"] == pytest.approx(0.3)
    assert result["selective"][2]["coverage"] == 0
    assert result["selective"][2]["risk"] is None


def test_real_loopback_native_parallelism_association_and_partial_failure(store):
    import threading
    import time

    from test_providers import server

    from reflex_axi.identity import digest
    from reflex_axi.models import Binding, Capabilities, ProviderSpec

    lock = threading.Lock()
    active = peak = 0

    def respond(body):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        if any("cafeteria" in r["state"]["context"] for r in body["requests"]):
            return 503, {"error": "private upstream detail"}
        return 200, {
            "results": [
                {
                    "id": r["id"],
                    "distribution": {
                        label: 1.0 if label == r["labels"][-1] else 0.0 for label in r["labels"]
                    },
                    "usage": {"units": 1.0},
                    "cost": 0.01,
                }
                for r in reversed(body["requests"])
            ]
        }

    with server(respond) as (endpoint, requests):
        binding = Binding(
            provider=ProviderSpec(
                adapter="jev",
                id="native-fixture",
                model="fixture",
                version="1",
                endpoint=endpoint,
                capabilities=Capabilities(
                    native_batch=True,
                    multi_question=True,
                    max_batch=2,
                    concurrency=2,
                    requests_per_second=10000.0,
                ),
            )
        )
        runner = BenchmarkRunner(store)
        result = runner.run(
            binding, selected=["starter"], modes=["batch"], repeats=1, concurrency=2
        )
    assert 1 < peak <= 2
    assert any(len(body["requests"]) == 2 for body, _ in requests)
    assert result["status"] == "PARTIAL"
    assert result["summary"]["resilience"]["errors"]["provider_http"] == 2
    # Transport availability is not raw distribution validity.
    assert result["summary"]["conformance"]["raw_distribution_and_association_validity"] == 1
    full = runner.show(result["id"], full=True)
    assert "private upstream detail" not in json.dumps(full)
    for chunk in full["chunks"]:
        for row in chunk["rows"]:
            if "result" in row:
                assert row["result"]["state_hash"] == digest(row["case"]["state"])
                assert row["result"]["selected"] == row["pack"]["judgment"]["labels"][-1]


@pytest.mark.parametrize("confidence", [0.79, 0.8, 0.81])
def test_confidence_threshold_boundary_is_reported(store, binding, confidence):
    from reflex_axi.models import Binding

    data = binding.model_dump()
    data["provider"]["settings"]["confidence"] = confidence
    run = BenchmarkRunner(store).run(
        Binding.model_validate(data), selected=["boundaries"], modes=["single"], repeats=1
    )
    full = BenchmarkRunner(store).show(run["id"], full=True)
    for chunk in full["chunks"]:
        for row in chunk["rows"]:
            if row["case"].get("policy_none"):
                assert row["result"]["policy_result"] is None
            else:
                assert row["result"]["policy_result"]["uncertain"] == (confidence < 0.8)
                assert row["result"]["policy_result"]["executed"] is False


def test_comparison_rejects_incompatible_runs_and_cross_project(
    store, binding, monkeypatch, tmp_path
):
    runner = BenchmarkRunner(store)
    a = runner.run(binding, selected=["starter"], modes=["batch"], repeats=1)
    b = runner.run(binding, selected=["boundaries"], modes=["batch"], repeats=1)
    with pytest.raises(ReflexError) as error:
        runner.compare(a["id"], b["id"])
    assert error.value.code == "benchmark_comparison"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ReflexError) as error:
        runner.show(a["id"])
    assert error.value.code == "not_found"


def test_changed_environment_cannot_resume(store, binding, monkeypatch):
    class Interrupted(MockProvider):
        def infer(self, requests):
            raise KeyboardInterrupt

    runner = BenchmarkRunner(store, Interrupted(binding.provider))
    with pytest.raises(KeyboardInterrupt):
        runner.run(binding, selected=["starter"], modes=["batch"], repeats=1)
    run_id = store.list("benchmark_runs")[0]["id"]
    monkeypatch.setattr("reflex_axi.benchmark.environment", lambda: {"changed": True})
    with pytest.raises(ReflexError) as error:
        BenchmarkRunner(store).resume(run_id)
    assert error.value.code == "benchmark_identity"


def test_export_preserves_partial_exit_code(cli, tmp_path):
    from conftest import write_json

    from reflex_axi.catalog import default_binding

    spec = default_binding().provider.model_dump()
    spec["settings"] = {"fail": True}
    path = write_json(tmp_path, "failure.json", spec)
    output = tmp_path / "failed.json"
    receipt = cli(
        "benchmark",
        "run",
        "--provider",
        path,
        "--suite",
        "boundaries",
        "--mode",
        "batch",
        "--repeats",
        "1",
        "--full",
        "--output",
        str(output),
        expected=1,
    )
    assert receipt["status"] == "written"
    assert json.loads(output.read_text())["status"] == "PARTIAL"


def test_expired_runner_cannot_publish_and_records_are_immutable(store, binding):
    class Expiring(MockProvider):
        def infer(self, requests):
            with store.transaction() as db:
                db.execute("UPDATE leases SET expires=0 WHERE key LIKE 'benchmark:%'")
            return super().infer(requests)

    runner = BenchmarkRunner(store, Expiring(binding.provider))
    with pytest.raises(ReflexError) as error:
        runner.run(binding, selected=["starter"], modes=["batch"], repeats=1)
    assert error.value.code == "lease_lost"
    meta = store.list("benchmark_runs")[0]
    assert store.list("benchmark_chunks:" + runner.key(meta["id"])) == []
    assert store.list("benchmark_summaries") == []
    with pytest.raises(ReflexError) as error:
        store.put("benchmark_runs", runner.key(meta["id"]), {**meta, "created": 0}, immutable=True)
    assert error.value.code == "version_conflict"
    assert BenchmarkRunner(store).resume(meta["id"])["status"] == "COMPLETE"
