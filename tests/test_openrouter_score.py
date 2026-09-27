"""Continuous score conformance through serialized HTTP and the universal engine."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_openrouter import MODEL, envelope, fixture, spec, typed_pack

from reflex_axi.benchmark import BenchmarkRunner, corpus
from reflex_axi.engine import Engine
from reflex_axi.errors import ReflexError
from reflex_axi.identity import canonical
from reflex_axi.models import Binding, Calibration, ExecutionPolicy
from reflex_axi.openrouter import OpenRouterJevProvider
from reflex_axi.openrouter_diagnostics import MESSAGES, ScoreDiagnostics, score_snapshot
from reflex_axi.providers import Request


def score_answer(native=1.99, probabilities=None, **updates):
    return {
        "type": "score",
        "score": native,
        "probabilities": probabilities if probabilities is not None else {"0": 0, "1": 0, "2": 1},
        **updates,
    }


@pytest.mark.parametrize(
    "native,probabilities,selected,semantic,warning",
    [
        (0, {"0": 1, "1": 0, "2": 0}, "low", 10, "none"),
        (1, {"0": 0, "1": 1, "2": 0}, "medium", 20, "none"),
        (2, {"0": 0, "1": 0, "2": 1}, "high", 80, "none"),
        (1.98, {"0": 0.01, "1": 0, "2": 0.99}, "high", 79.3, "none"),
        (1.99, {"0": 0.01, "1": 0, "2": 0.99}, "high", 79.3, "small_disagreement"),
        (1.99, {"0": 0, "1": 0, "2": 1}, "high", 80, "small_disagreement"),
        (0.99, {"0": 0, "1": 1, "2": 0}, "medium", 20, "small_disagreement"),
        (0, {"0": 0, "1": 0, "2": 1}, "high", 80, "large_disagreement"),
        (2, {"0": 1, "1": 0, "2": 0}, "low", 10, "large_disagreement"),
        (1, {"2": 0.5, "1": 0, "0": 0.5}, "low", 45, "none"),
        (0.02, {"0": 1, "1": 0, "2": 0}, "low", 10, "small_disagreement"),
        (1.98, {"0": 0, "1": 0, "2": 1}, "high", 80, "small_disagreement"),
        (0.02001, {"0": 1, "1": 0, "2": 0}, "low", 10, "large_disagreement"),
        (1e-12, {"0": 1, "1": 0, "2": 0}, "low", 10, "none"),
        (2 - 1e-12, {"0": 0, "1": 0, "2": 1}, "high", 80, "none"),
    ],
)
def test_score_conformance(
    monkeypatch, store, pack, native, probabilities, selected, semantic, warning
):
    pack = typed_pack(pack, "score")
    send, calls = fixture(monkeypatch, envelope(score_answer(native, probabilities)))
    binding = Binding(provider=spec())
    sink = ScoreDiagnostics(store)
    engine = Engine(
        store, OpenRouterJevProvider(binding.provider, transport=send, score_diagnostics=sink)
    )
    for _ in range(2):
        result = engine.evaluate(
            pack, {"text": "synthetic"}, binding, execution=ExecutionPolicy(cache=False)
        )
        expected_index = sum(int(i) * p for i, p in probabilities.items())
        assert result.expected_score == pytest.approx(semantic)
        assert result.selected == selected
        assert list(result.distribution) == pack.judgment.labels
        assert list(result.distribution.values()) == [probabilities[str(i)] for i in range(3)]
        diagnostics = result.execution["provider_diagnostics"]
        assert diagnostics["native_score"] == native
        assert diagnostics["native_expected_index"] == pytest.approx(expected_index)
        assert diagnostics["native_score_disagreement"] == pytest.approx(
            abs(native - expected_index)
        )
        assert diagnostics["native_score_warning"] == warning
    rows = store.list(sink.bucket)
    assert len(rows) == len(calls) == 2
    assert rows[0]["probabilities"] == probabilities
    assert rows[0]["category"] == "accepted"
    assert rows[0]["native_score_warning"] == warning
    assert not store.list("cache")


@pytest.mark.parametrize("value", [True, "1.99", None, -0.000001, 2.000001, 10**400])
def test_invalid_native_score(monkeypatch, store, pack, value):
    assert_score_rejected(monkeypatch, store, pack, envelope(score_answer(value)), "native_score")


def assert_score_rejected(monkeypatch, store, pack, raw, category):
    pack = typed_pack(pack, "score")
    send, calls = fixture(monkeypatch, raw)
    sink = ScoreDiagnostics(store)
    binding = Binding(provider=spec())
    with pytest.raises(ReflexError):
        Engine(
            store, OpenRouterJevProvider(binding.provider, transport=send, score_diagnostics=sink)
        ).evaluate(
            pack,
            {"text": "private-state-sentinel"},
            binding,
        )
    (row,) = store.list(sink.bucket)
    assert row["category"] == category
    assert row["message"] == MESSAGES[category]
    assert len(canonical(row).encode()) <= sink.MAX_BYTES
    assert len(calls) == 1 and not store.list("cache")
    assert "sentinel" not in canonical(row)
    assert "offline-fixture-secret" not in canonical(row)
    return row


@pytest.mark.parametrize(
    "probabilities,category",
    [
        ({"0": 0, "1": 0}, "probability_keys"),
        ({"0": 0, "1": 0, "2": 1, "provider-secret-sentinel": 0}, "probability_keys"),
        ({"low": 0, "medium": 0, "high": 1}, "probability_keys"),
        ({"0": -0.01, "1": 0.01, "2": 1}, "probability_value"),
        ({"0": 0, "1": 0, "2": 1.000001}, "probability_value"),
        ({"0": 0, "1": 0, "2": True}, "probability_value"),
        ({"0": 0, "1": 0, "2": "1"}, "probability_value"),
        ({"0": 0, "1": 0, "2": None}, "probability_value"),
        ({"0": 0, "1": 0, "2": 0.99}, "probability_sum"),
        ({"0": 0, "1": 0.000002, "2": 1}, "probability_sum"),
        ({"0": 0, "1": 0, "2": 10**400}, "probability_value"),
        ([], "probability_keys"),
    ],
)
def test_rejected_distribution_diagnostics(monkeypatch, store, pack, probabilities, category):
    assert_score_rejected(
        monkeypatch, store, pack, envelope(score_answer(probabilities=probabilities)), category
    )


@pytest.mark.parametrize(
    "answer,category",
    [
        ({"type": "score", "score": 1}, "schema"),
        (score_answer(legend={"0": "high", "1": "medium", "2": "low"}), "legend"),
        (score_answer(legend={"0": "low", "1": "medium"}), "legend"),
        (score_answer(legend=None), "legend"),
        (score_answer(confidence=None), "native_confidence"),
        (score_answer(type="provider-secret-sentinel"), "primitive"),
        (score_answer(type=["provider-secret-sentinel"]), "primitive"),
        (score_answer(extra="provider-secret-sentinel"), "schema"),
    ],
)
def test_score_schema_diagnostics(monkeypatch, store, pack, answer, category):
    assert_score_rejected(monkeypatch, store, pack, envelope(answer), category)


@pytest.mark.parametrize(
    "updates,category",
    [
        ({"model": "provider-secret-sentinel"}, "provider_version"),
        ({"provider": "provider-secret-sentinel"}, "provider_version"),
        ({"usage": {"input_tokens": True, "output_tokens": 0}}, "usage"),
        ({"id": []}, "request_id"),
        ({"answers": ["provider-secret-sentinel"]}, "schema"),
    ],
)
def test_score_envelope_diagnostics(monkeypatch, store, pack, updates, category):
    assert_score_rejected(monkeypatch, store, pack, envelope(score_answer(), **updates), category)


@pytest.mark.parametrize(
    "raw,category",
    [
        (b"x" * 8_000_001, "provider_response"),
        (b'{"provider-secret-sentinel":0,"provider-secret-sentinel":1}', "provider_transport"),
        (json.dumps(envelope(score_answer(float("nan")))).encode(), "provider_transport"),
        (json.dumps(envelope(score_answer(float("inf")))).encode(), "provider_transport"),
        (
            json.dumps(
                envelope(score_answer(probabilities={"0": 0, "1": 0, "2": float("nan")}))
            ).encode(),
            "provider_transport",
        ),
        (b"[]", "schema"),
    ],
)
def test_score_wire_diagnostics(monkeypatch, store, pack, raw, category):
    assert_score_rejected(monkeypatch, store, pack, raw, category)


def test_diagnostics_bounds_privacy_and_concurrency(store):
    raw = envelope(
        score_answer(
            probabilities={
                **{str(i): i / 100 for i in range(10)},
                **{f"secret-sentinel-{i}": 0 for i in range(1000)},
            },
            legend={"secret-sentinel": "secret-sentinel"},
            extra="secret-sentinel",
        ),
        id="secret-sentinel",
        provider="secret-sentinel",
        model="secret-sentinel",
    )
    snapshot = score_snapshot(raw, 3, MODEL)
    assert "sentinel" not in canonical(snapshot)
    assert snapshot["probability_count"] == 1010
    assert snapshot["probabilities_complete"] is False
    assert len(snapshot["probabilities"]) == 10
    assert snapshot["legend_count"] == 1
    sink = ScoreDiagnostics(store)
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(lambda _: sink.record(snapshot, "probability_keys"), range(300)))
    rows = store.list(sink.bucket)
    assert len(rows) == sink.MAX_RECORDS + 1
    assert rows[-1]["category"] == "diagnostic_limit"
    assert all(len(canonical(row).encode()) <= sink.MAX_BYTES for row in rows)
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_diagnostics_absent_by_default(monkeypatch, store, pack, capsys):
    pack = typed_pack(pack, "score")
    send, _ = fixture(monkeypatch, envelope(score_answer()))
    provider = OpenRouterJevProvider(spec(), transport=send)
    provider.infer([Request("private-id-sentinel", pack, {})])
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM records").fetchone()[0] == 0
    assert capsys.readouterr() == ("", "")


def test_diagnostic_byte_bound_preserves_failure_category(store):
    # The configured snapshot name is trusted configuration, but still bounded on disk.
    model = "typesafe/jev-" + "1" * 5000 + ".13-20260917"
    snapshot = score_snapshot(envelope(score_answer(-1), model=model), 3, model)
    sink = ScoreDiagnostics(store)
    sink.record(snapshot, "native_score")
    (row,) = store.list(sink.bucket)
    assert row == {
        "category": "native_score",
        "message": MESSAGES["native_score"],
        "diagnostic_truncated": True,
    }


def test_unexpected_failure_never_recorded_as_accepted(monkeypatch, store, pack):
    send, _ = fixture(monkeypatch, error=RuntimeError("provider-secret-sentinel"))
    sink = ScoreDiagnostics(store)
    with pytest.raises(RuntimeError):
        OpenRouterJevProvider(spec(), transport=send, score_diagnostics=sink).infer(
            [Request("id", typed_pack(pack, "score"), {})]
        )
    (row,) = store.list(sink.bucket)
    assert row["category"] == "invalid_response"
    assert "sentinel" not in canonical(row)


def test_calibration_uses_distribution_not_native_score(monkeypatch, store, pack):
    pack = typed_pack(pack, "score")
    configured = spec()
    binding = Binding(
        provider=configured,
        calibration=Calibration(
            id="score-fixture",
            version="1",
            provider_hash=configured.identity(),
            pack_id=pack.id,
            pack_version=pack.version,
            workload="default",
            scope="default",
            temperature=2,
        ),
    )
    send, _ = fixture(monkeypatch, envelope(score_answer(0, {"0": 0.01, "1": 0, "2": 0.99})))
    result = Engine(store, OpenRouterJevProvider(configured, transport=send)).evaluate(
        pack, {"text": "synthetic"}, binding
    )
    assert result.raw_distribution == {"low": 0.01, "medium": 0, "high": 0.99}
    assert result.expected_score == pytest.approx(
        sum(result.distribution[label] * value for label, value in pack.judgment.values.items())
    )
    assert result.execution["provider_diagnostics"]["native_score"] == 0
    assert result.execution["provider_diagnostics"]["native_expected_index"] == pytest.approx(1.98)


def test_starter_benchmark_with_rounded_scores(monkeypatch, store):
    # Authored labels create deterministic wire evidence, never a claim of live quality.
    cases = [c for c in corpus()["cases"] if c["suite"] == "starter"]
    index = 0

    def response(body):
        nonlocal index
        case = cases[index]
        index += 1
        q = body["questions"]["judgment"]
        if q["type"] == "noul":
            answer = {"type": "noul", "noul": float(case["expected"] == "true")}
        elif q["type"] == "choice":
            answer = {
                "type": "choice",
                "choice": case["expected"],
                "probabilities": {
                    label: float(label == case["expected"]) for label in q["criteria"]
                },
            }
        else:
            position = q["criteria"].index(case["expected"])
            answer = score_answer(
                max(0, position - 0.01), {str(i): float(i == position) for i in range(3)}
            )
        return envelope(answer)

    send, calls = fixture(monkeypatch, response)
    binding = Binding(provider=spec())
    runner = BenchmarkRunner(
        store, OpenRouterJevProvider(binding.provider, transport=send), score_diagnostics=True
    )
    run = runner.run(binding, selected=["starter"], modes=["single"], repeats=1, concurrency=1)
    assert run["status"] == "COMPLETE"
    assert run["summary"]["conformance"]["pass_rate"] == 1
    assert run["summary"]["quality"]["accuracy"] == 1
    assert len(calls) == 19
    rows = store.list("openrouter_score_diagnostics:" + run["id"])
    assert len(rows) == 3
    assert sorted(r["native_score"] for r in rows) == [0, 0.99, 1.99]
    sealed = runner.show(run["id"], full=True)
    assert runner.resume(run["id"]) == run
    assert runner.show(run["id"], full=True) == sealed
    assert len(calls) == 19
    assert not store.list("cache") and not store.list("evidence") and not store.list("decisions")


def test_score_diagnostic_cli_restrictions(cli):
    wrong_provider = cli(
        "benchmark", "run", "--provider", "mock", "--score-diagnostics", expected=2
    )
    assert wrong_provider["code"] == "usage"
    wrong_operation = cli("benchmark", "list", "--score-diagnostics", expected=2)
    assert wrong_operation["code"] == "usage"


@pytest.mark.parametrize("count", [2, 10])
def test_score_supported_cardinality_boundaries(monkeypatch, pack, count):
    from reflex_axi.models import Pack

    data = typed_pack(pack, "score").model_dump(by_alias=True)
    labels = [f"level-{i}" for i in range(count)]
    data["judgment"].update(
        labels=labels, values={label: i * 10.0 for i, label in enumerate(labels)}
    )
    pack = Pack.model_validate(data)
    probabilities = {str(i): float(i == count - 1) for i in range(count)}
    send, _ = fixture(
        monkeypatch,
        envelope(
            score_answer(
                count - 1, probabilities, legend={str(i): label for i, label in enumerate(labels)}
            )
        ),
    )
    result = OpenRouterJevProvider(spec(), transport=send).infer([Request("id", pack, {})])["id"]
    assert result.distribution == {label: float(i == count - 1) for i, label in enumerate(labels)}
    assert result.diagnostics["native_score"] == count - 1


def test_probability_tolerance_unchanged(monkeypatch, store, pack):
    # Accepted mass is preserved exactly, never silently normalized.
    pack = typed_pack(pack, "score")
    probabilities = {"0": 0, "1": 0.0000005, "2": 1}
    send, _ = fixture(monkeypatch, envelope(score_answer(1.99, probabilities)))
    result = OpenRouterJevProvider(spec(), transport=send).infer([Request("id", pack, {})])["id"]
    assert list(result.distribution.values()) == list(probabilities.values())


def test_diagnostics_do_not_mutate_provider_identity(monkeypatch, store):
    send, _ = fixture(monkeypatch, envelope(score_answer()))
    original = spec()
    provider = OpenRouterJevProvider(original, transport=send)
    binding = Binding(provider=original)
    runner = BenchmarkRunner(store, provider, score_diagnostics=True)
    # Envelope mismatches for non-score cells are expected here, with no network retries.
    result = runner.run(binding, selected=["starter"], modes=["single"], repeats=1)
    assert provider.score_diagnostics is None
    assert provider.spec.identity() == original.identity()
    assert (
        runner.show(result["id"], full=True)["metadata"]["definition"]["provider_hash"]
        == original.identity()
    )
