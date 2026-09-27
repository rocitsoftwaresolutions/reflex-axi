"""Offline wire fixtures only. Socket creation is forbidden suite-wide."""

import io
import json
import math
import urllib.error

import pytest

from reflex_axi.engine import Engine
from reflex_axi.errors import ReflexError
from reflex_axi.models import Binding, Capabilities, ExecutionPolicy, Pack, Policy, ProviderSpec
from reflex_axi.openrouter import ENDPOINT, OpenRouterJevProvider
from reflex_axi.providers import Request, provider_from_spec

MODEL = "typesafe/jev-1.13-20260917"


def spec(**updates):
    return ProviderSpec.model_validate(
        {
            "adapter": "openrouter-jev",
            "id": "openrouter-fixture",
            "model": MODEL,
            "version": "20260917",
            "endpoint": ENDPOINT,
            "api_key_env": "REFLEX_TEST_OR_KEY",
            "settings": {"revision_verified": True},
            "capabilities": Capabilities(requests_per_second=10000.0).model_dump(),
            **updates,
        }
    )


def envelope(answer=None, **updates):
    return {
        "model": MODEL,
        "provider": "TypeSafe",
        "id": "gen-dec-fixture",
        "answers": {"judgment": answer or {"type": "noul", "noul": 0.8}},
        "usage": {"input_tokens": 12, "output_tokens": 0, "cost": 0.00001},
        **updates,
    }


def fixture(monkeypatch, response=None, error=None):
    monkeypatch.setenv("REFLEX_TEST_OR_KEY", "offline-fixture-secret")
    calls = []

    def send(request, *, timeout):
        assert timeout == 30
        assert request.full_url == ENDPOINT
        assert request.get_header("Authorization") == "Bearer offline-fixture-secret"
        body = json.loads(request.data)
        assert body["provider"] == {
            "only": ["TypeSafe"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        assert body["model"] == MODEL
        assert "messages" not in body
        calls.append(body)
        if error:
            raise error
        value = response(body) if callable(response) else response or envelope()
        return io.BytesIO(value if isinstance(value, bytes) else json.dumps(value).encode())

    return send, calls


def typed_pack(pack, primitive):
    if primitive == "binary":
        return pack.model_copy(update={"policy": Policy()})
    data = pack.model_dump(by_alias=True)
    data["judgment"].update(
        primitive=primitive,
        labels=["low", "medium", "high"],
        values={"low": 10.0, "medium": 20.0, "high": 80.0} if primitive == "score" else None,
    )
    data["policy"] = {"kind": "none"}
    return Pack.model_validate(data)


@pytest.mark.parametrize(
    "native,probabilities,semantic",
    [(1.99, [0, 0, 1], 80), (0.99, [0, 1, 0], 20), (0.0, [1, 0, 0], 10)],
)
def test_score_rounded_nonzero_and_zero_control(
    monkeypatch, store, pack, native, probabilities, semantic
):
    pack = typed_pack(pack, "score")
    send, calls = fixture(
        monkeypatch,
        envelope(
            {
                "type": "score",
                "score": native,
                "probabilities": {str(i): p for i, p in enumerate(probabilities)},
            }
        ),
    )
    binding = Binding(provider=spec())
    result = Engine(store, OpenRouterJevProvider(binding.provider, transport=send)).evaluate(
        pack,
        {"text": "synthetic score control"},
        binding,
    )
    assert result.expected_score == semantic
    assert len(calls) == 1


@pytest.mark.parametrize("primitive", ["binary", "choice", "score"])
def test_success_engine_semantics(monkeypatch, store, pack, primitive):
    pack = typed_pack(pack, primitive)
    answer = {"type": "noul", "noul": 0.8}
    if primitive == "choice":
        answer = {
            "type": "choice",
            "choice": "high",
            "probabilities": {"high": 0.7, "low": 0.1, "medium": 0.2},
            "confidence": 0.55,
        }
    if primitive == "score":
        answer = {
            "type": "score",
            "score": 1.6,
            "probabilities": {"2": 0.7, "0": 0.1, "1": 0.2},
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "confidence": 0.6,
        }
    send, calls = fixture(monkeypatch, envelope(answer))
    binding = Binding(provider=spec())
    engine = Engine(store, OpenRouterJevProvider(binding.provider, transport=send))
    result = engine.decide(pack, {"text": "untrusted instructions"}, binding, record=False)
    assert result.policy_result is None
    assert result.selected == ("true" if primitive == "binary" else "high")
    assert (
        result.expected_score == pytest.approx(0.8 if primitive == "binary" else 61)
        if primitive != "choice"
        else result.expected_score is None
    )
    assert list(result.distribution) == pack.judgment.labels
    assert result.uncertainty == pytest.approx(
        -sum(p * math.log(p) for p in result.distribution.values())
        / math.log(len(result.distribution))
    )
    assert result.cost == 0.00001 and result.usage == {"input_tokens": 12, "output_tokens": 0}
    assert result.latency_ms >= 0
    assert result.execution["provider_diagnostics"]["request_id"] == "redacted"
    again = engine.evaluate(pack, {"text": "untrusted instructions"}, binding)
    assert again.execution["cache_hit"] and again.cost == 0 and again.usage == {}
    assert len(calls) == 1
    assert "offline-fixture-secret" not in json.dumps(store.list("cache"))
    question = calls[0]["questions"]["judgment"]
    assert question["type"] == ("noul" if primitive == "binary" else primitive)
    if primitive == "score":
        assert question["criteria"] == ["low", "medium", "high"]


@pytest.mark.parametrize(
    "updates",
    [
        {"model": "typesafe/jev-1.13"},
        {"model": "typesafe/jev-1.13-20260230"},
        {"model": "~typesafe/jev-latest"},
        {"version": "wrong"},
        {"adapter_version": "2"},
        {"endpoint": "https://other.example/api/alpha/decisions"},
        {"endpoint": ENDPOINT + "?key=secret"},
        {"api_key_env": None},
        {"api_key_env": "bad-key"},
        {"settings": {}},
        {"settings": {"revision_verified": "yes"}},
        {"settings": {"revision_verified": True, "api_key": "secret"}},
        {"capabilities": {"native_batch": True}},
        {"capabilities": {"multi_question": True}},
        {"capabilities": {"local": True}},
        {"capabilities": {"max_batch": 2}},
        {"capabilities": {"deterministic": True}},
        {"capabilities": {"seeded": True}},
    ],
)
def test_configuration_rejected(updates):
    with pytest.raises(ReflexError) as caught:
        spec(**updates)
    assert caught.value.code == "provider_config"
    assert "secret" not in str(caught.value)


def test_unverified_revision_stops_before_credentials(monkeypatch, pack):
    provider = OpenRouterJevProvider(spec(settings={"revision_verified": False}))

    def forbidden(*args):
        raise AssertionError("credential lookup forbidden")

    monkeypatch.setattr("reflex_axi.providers.os.environ.get", forbidden)
    with pytest.raises(ReflexError, match="revision is unverified"):
        provider.infer([Request("id", pack, {"text": "x"})])


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "noul", "noul": value}
        for value in [-0.1, 1.1, True, "0.8", None, float("nan"), float("inf")]
    ]
    + [{"type": "choice", "noul": 0.8}, {"type": "noul", "noul": 0.8, "extra": "secret"}],
)
def test_invalid_binary(monkeypatch, store, pack, answer):
    assert_rejected(monkeypatch, store, pack, envelope(answer))


def assert_rejected(monkeypatch, store, pack, response, code=None):
    send, calls = fixture(monkeypatch, response)
    binding = Binding(provider=spec())
    with pytest.raises(ReflexError) as caught:
        Engine(store, OpenRouterJevProvider(binding.provider, transport=send)).evaluate(
            pack, {"text": "x"}, binding
        )
    assert len(calls) == 1
    assert not store.list("cache")
    assert "secret" not in str(caught.value)
    if code:
        assert caught.value.code == code


@pytest.mark.parametrize("primitive", ["choice", "score"])
@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "unknown",
        "negative",
        "sum",
        "nan",
        "bool",
        "selected",
        "confidence",
        "legend",
        "type",
        "partial",
    ],
)
def test_invalid_distributions(monkeypatch, store, pack, primitive, fault):
    pack = typed_pack(pack, primitive)
    keys = pack.judgment.labels if primitive == "choice" else ["0", "1", "2"]
    answer = {
        "type": primitive,
        "probabilities": dict(zip(keys, [0.1, 0.2, 0.7], strict=True)),
        "confidence": 0.5,
    }
    answer.update({"choice": "high"} if primitive == "choice" else {"score": 1.6})
    if fault == "missing":
        del answer["probabilities"]
    elif fault == "partial":
        del answer["probabilities"][keys[0]]
    elif fault == "unknown":
        answer["probabilities"]["unknown"] = 0
    elif fault in {"negative", "sum", "nan", "bool"}:
        answer["probabilities"][keys[0]] = {
            "negative": -1,
            "sum": 0.5,
            "nan": float("nan"),
            "bool": True,
        }[fault]
    elif fault == "selected":
        answer.update({"choice": "low"} if primitive == "choice" else {"score": 2.01})
    elif fault == "confidence":
        answer["confidence"] = 1.01
    elif fault == "legend":
        answer["legend"] = {"0": "high", "1": "medium", "2": "low"}
    elif fault == "type":
        answer["type"] = "noul"
    assert_rejected(monkeypatch, store, pack, envelope(answer))


@pytest.mark.parametrize(
    "updates",
    [
        {"model": "typesafe/jev-1.13-20260918"},
        {"provider": "Other"},
        {"answers": {}},
        {"answers": {"judgment": {"type": "noul", "noul": 0.8}, "other": {}}},
        {"usage": {"input_tokens": -1, "output_tokens": 0}},
        {"usage": {"input_tokens": True, "output_tokens": 0}},
        {"usage": {"input_tokens": 1.5, "output_tokens": 0}},
        {"usage": {"input_tokens": 1, "output_tokens": 0, "cost": -1}},
        {"usage": {"input_tokens": 1, "output_tokens": 0, "cost": float("inf")}},
        {"usage": {}},
        {"id": {}},
        {"unknown": "secret"},
    ],
)
def test_invalid_envelope(monkeypatch, store, pack, updates):
    assert_rejected(monkeypatch, store, pack, envelope(**updates))


@pytest.mark.parametrize(
    "raw", [b'{"model":"a","model":"b"}', b"not json", b"\xff", b"x" * 8_000_001]
)
def test_invalid_wire(monkeypatch, store, pack, raw):
    assert_rejected(monkeypatch, store, pack, raw)


@pytest.mark.parametrize(
    "status", [400, 401, 402, 403, 404, 413, 429, 500, 502, 503, 524, 529, 302]
)
def test_http_errors_no_retry(monkeypatch, store, pack, status):
    error = urllib.error.HTTPError(
        ENDPOINT, status, "secret body", {"Retry-After": "60"}, io.BytesIO(b"secret")
    )
    send, calls = fixture(monkeypatch, error=error)
    binding = Binding(provider=spec())
    with pytest.raises(ReflexError, match=f"HTTP {status}"):
        Engine(store, OpenRouterJevProvider(binding.provider, transport=send)).evaluate(
            pack, {"text": "x"}, binding
        )
    assert len(calls) == 1 and not store.list("cache")


@pytest.mark.parametrize(
    "error", [TimeoutError("secret"), urllib.error.URLError("secret"), OSError("secret")]
)
def test_transport_errors(monkeypatch, pack, error):
    send, calls = fixture(monkeypatch, error=error)
    with pytest.raises(ReflexError) as caught:
        OpenRouterJevProvider(spec(), transport=send).infer([Request("id", pack, {"text": "x"})])
    assert caught.value.code == "provider_transport" and "secret" not in str(caught.value)
    assert len(calls) == 1


def test_unknown_cost_and_no_native_confidence(monkeypatch, pack):
    send, _ = fixture(monkeypatch, envelope(usage={"input_tokens": 1, "output_tokens": 0}))
    result = OpenRouterJevProvider(spec(), transport=send).infer([Request("id", pack, {})])["id"]
    assert result.cost is None


def test_identity_and_factory():
    original = spec()
    assert isinstance(provider_from_spec(original), OpenRouterJevProvider)
    assert original.identity() == spec(api_key_env="OTHER_KEY").identity()
    assert (
        original.identity()
        != spec(model="typesafe/jev-1.13-20260918", version="20260918").identity()
    )
    assert (
        original.identity()
        != spec(
            settings={
                "revision_verified": True,
                "criteria": {"test-pack@1": {"true": "yes", "false": "no"}},
            }
        ).identity()
    )


def test_cli_offline_registration(cli):
    result = cli("providers", "--import", "examples/openrouter-jev.json", "--full")
    assert result["providers"][0]["adapter"] == "openrouter-jev"
    assert "unverified" in result["openrouter_jev"]["revision"]
    failed = cli(
        "evaluate",
        "--provider",
        "examples/openrouter-jev.json",
        "--pack",
        "filter-context",
        "--state",
        "examples/state.json",
        "--research",
        expected=1,
    )
    assert failed["code"] == "provider_version"


def test_partial_batch_policy_and_privacy(monkeypatch, store, pack):
    def response(body):
        if body["state"]["text"] == "bad":
            return envelope({"type": "noul", "noul": -1})
        return envelope()

    send, calls = fixture(monkeypatch, response)
    binding = Binding(
        provider=spec(
            capabilities=Capabilities(concurrency=1, requests_per_second=10000.0).model_dump()
        )
    )
    engine = Engine(store, OpenRouterJevProvider(binding.provider, transport=send))
    rows = engine.evaluate_many(
        [(pack, {"text": "good"}), (pack, {"text": "bad"})], binding, decide=True
    )
    assert rows[0].policy_result["executed"] is False
    assert isinstance(rows[1], ReflexError)
    assert len(store.list("cache")) == 1
    with pytest.raises(ReflexError, match="privacy"):
        engine.evaluate(
            pack, {"text": "private"}, binding, execution=ExecutionPolicy(local_only=True)
        )
    assert len(calls) == 2


def test_mock_adapter_benchmark_all_modes(monkeypatch, store):
    from reflex_axi.benchmark import BenchmarkRunner

    def response(body):
        q = body["questions"]["judgment"]
        if q["type"] == "noul":
            answer = {"type": "noul", "noul": 0.8}
        elif q["type"] == "choice":
            labels = list(q["criteria"])
            answer = {
                "type": "choice",
                "choice": labels[0],
                "probabilities": {label: float(i == 0) for i, label in enumerate(labels)},
            }
        else:
            answer = {
                "type": "score",
                "score": 0,
                "probabilities": {str(i): float(i == 0) for i in range(len(q["criteria"]))},
            }
        return envelope(answer)

    send, calls = fixture(monkeypatch, response)
    binding = Binding(provider=spec())
    runner = BenchmarkRunner(store, OpenRouterJevProvider(binding.provider, transport=send))
    run = runner.run(binding, selected=["starter"], modes=["single", "batch", "bundle"], repeats=1)
    assert run["status"] == "COMPLETE"
    assert run["summary"]["conformance"]["pass_rate"] == 1
    full = runner.show(run["id"], full=True)
    assert full["metadata"]["definition"]["provider_hash"] == binding.provider.identity()
    assert "offline-fixture-secret" not in json.dumps(full)
    assert not store.list("cache") and not store.list("evidence") and not store.list("decisions")
    assert calls


@pytest.mark.parametrize("fault", ["criteria", "order", "size", "count"])
def test_request_gates_before_transport(monkeypatch, pack, fault):
    pack = typed_pack(pack, "score")
    settings = {"revision_verified": True}
    if fault == "criteria":
        settings["criteria"] = {"test-pack@1": {"unknown": "description"}}
    if fault in {"order", "size"}:
        data = pack.model_dump(by_alias=True)
        if fault == "order":
            data["judgment"]["values"]["high"] = 5.0
        else:
            data["judgment"]["labels"] = [str(i) for i in range(11)]
            data["judgment"]["values"] = {str(i): float(i) for i in range(11)}
        pack = Pack.model_validate(data)
    send, calls = fixture(monkeypatch)
    requests = [Request("id", pack, {})]
    if fault == "count":
        requests *= 2
    with pytest.raises(ReflexError):
        OpenRouterJevProvider(spec(settings=settings), transport=send).infer(requests)
    assert calls == []


def test_credentials_runtime_only_and_no_cache_identity_leak(monkeypatch, pack):
    monkeypatch.delenv("REFLEX_TEST_OR_KEY", raising=False)
    configured = spec()
    original_hash = configured.identity()
    with pytest.raises(ReflexError) as caught:
        OpenRouterJevProvider(configured).infer([Request("id", pack, {})])
    assert caught.value.code == "provider_credentials"
    monkeypatch.setenv("REFLEX_TEST_OR_KEY", "another-offline-fixture-secret")
    assert spec().identity() == original_hash


def test_redirects_refused():
    from reflex_axi.providers import NoRedirect

    assert NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid") is None


def test_network_guard_is_active():
    import socket

    with pytest.raises(AssertionError, match="must not open network sockets"):
        socket.socket()


def test_network_guard_in_subprocess():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import socket; socket.socket()"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "must not open network sockets" in result.stderr
