"""Preserve safe rejection reasons in failed batch cells without changing acceptance."""

import json

import pytest
from test_openrouter import envelope, fixture, spec, typed_pack

from reflex_axi.engine import Engine
from reflex_axi.jobs import BatchRunner
from reflex_axi.models import Binding
from reflex_axi.openrouter import OpenRouterJevProvider


@pytest.mark.parametrize(
    "fault,category",
    [
        ("missing_map", "schema"),
        ("missing_label", "probability_keys"),
        ("unknown_label", "probability_keys"),
        ("rounded_sum", "probability_sum"),
        ("boolean", "probability_value"),
        ("choice", "choice"),
        ("confidence", "native_confidence"),
        ("usage", "usage"),
        ("extra_metadata", "schema"),
        ("id", "request_id"),
    ],
)
def test_reason_survives_batch_without_caching_or_retry(monkeypatch, store, pack, fault, category):
    pack = typed_pack(pack, "choice")
    answer = {
        "type": "choice",
        "choice": "high",
        "probabilities": {"low": 0.1, "medium": 0.2, "high": 0.7},
    }
    raw = envelope(answer)
    if fault == "missing_map":
        del answer["probabilities"]
    elif fault == "missing_label":
        del answer["probabilities"]["low"]
    elif fault == "unknown_label":
        answer["probabilities"]["UNTRUSTED_PROVIDER_TEXT"] = 0
    elif fault == "rounded_sum":
        answer["probabilities"]["low"] = 0.09
    elif fault == "boolean":
        answer["probabilities"]["low"] = True
    elif fault == "choice":
        answer["choice"] = "low"
    elif fault == "confidence":
        answer["confidence"] = "UNTRUSTED_PROVIDER_TEXT"
    elif fault == "usage":
        raw["usage"]["cost"] = "UNTRUSTED_PROVIDER_TEXT"
    elif fault == "extra_metadata":
        raw["UNTRUSTED_PROVIDER_TEXT"] = "UNTRUSTED_PROVIDER_TEXT"
    elif fault == "id":
        raw["id"] = ["UNTRUSTED_PROVIDER_TEXT"]

    send, calls = fixture(monkeypatch, raw)
    binding = Binding(provider=spec())
    runner = BatchRunner(
        store, Engine(store, OpenRouterJevProvider(binding.provider, transport=send))
    )
    job = runner.run([{"text": "synthetic input"}], [pack], binding)
    error = runner.results(job["id"])[0]["error"]
    assert job["failed"] == 1
    assert error["code"] == "provider_response"
    assert len(calls) == 1
    assert not store.list("cache")
    assert "UNTRUSTED_PROVIDER_TEXT" not in json.dumps(error)
    assert "offline-fixture-secret" not in json.dumps(error)
    assert f"({category})" in error["message"]
