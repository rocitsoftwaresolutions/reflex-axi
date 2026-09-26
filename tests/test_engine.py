from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import changed_binding
from pydantic import ValidationError

from reflex_axi.engine import Engine
from reflex_axi.errors import ReflexError
from reflex_axi.models import Binding, Calibration, ExecutionPolicy, Pack, Policy, Prediction
from reflex_axi.providers import MockProvider


def test_evaluate_decide_policy_none_and_cache(store, pack, binding):
    provider = MockProvider(binding.provider)
    engine = Engine(store, provider)
    result = engine.evaluate(pack, {"text": "hello"}, binding)
    assert result.policy_result is None
    assert result.raw_distribution == result.distribution
    assert sum(result.distribution.values()) == 1
    assert result.expected_score == pytest.approx(0.1)
    assert result.uncertainty > 0
    decided = engine.decide(pack, {"text": "hello"}, binding)
    assert decided.policy_result["recommendation"] == "exclude"
    assert decided.policy_result["executed"] is False
    assert decided.execution["cache_hit"] is True
    assert provider.calls == 1
    assert decided.id != result.id
    assert decided.inference_hash == result.inference_hash
    assert len(store.list("decisions")) == 1
    none_pack = pack.model_copy(update={"policy": Policy()})
    assert engine.decide(none_pack, {"text": "hello"}, binding).policy_result is None
    assert provider.calls == 1


def test_bundle_native_call_and_deterministic_association(store, pack, binding):
    provider = MockProvider(binding.provider)
    engine = Engine(store, provider)
    other = pack.model_copy(update={"id": "other-pack"})
    items = [(p, {"text": str(i)}) for i in range(20) for p in [pack, other]]
    result = engine.evaluate_many(items, binding)
    assert provider.calls == 1
    assert [r.pack for r in result] == [p.id for p, _ in items]
    assert len({r.inference_hash for r in result}) == 40


def test_parallel_cache_stampede_and_duplicate_inputs(store, pack, binding):
    provider = MockProvider(binding.provider)
    engine = Engine(store, provider)
    with ThreadPoolExecutor(max_workers=8) as pool:
        result = list(
            pool.map(lambda _: engine.evaluate(pack, {"text": "same"}, binding), range(12))
        )
    assert provider.calls == 1
    assert len({r.id for r in result}) == 12
    provider2 = MockProvider(binding.provider)
    duplicate = Engine(store, provider2).evaluate_many([(pack, {"text": "new"})] * 10, binding)
    assert provider2.calls == 1
    assert len({r.id for r in duplicate}) == 10


def test_cache_identity_all_versions_and_settings(store, pack, binding):
    engine = Engine(store)
    original = engine.evaluate(pack, {"text": "hi"}, binding)
    new_provider = changed_binding(binding)
    changed = [engine.evaluate(pack, {"text": "hi"}, new_provider)]
    changed.append(
        engine.evaluate(pack.model_copy(update={"version": "2"}), {"text": "hi"}, binding)
    )
    changed.append(
        engine.evaluate(
            pack.model_copy(update={"state": pack.state.model_copy(update={"version": "2"})}),
            {"text": "hi"},
            binding,
        )
    )
    changed.append(
        engine.evaluate(pack, {"text": "hi"}, binding, execution=ExecutionPolicy(workload="other"))
    )
    assert all(r.inference_hash != original.inference_hash for r in changed)
    assert all(not r.execution["cache_hit"] for r in changed)


def test_calibration_scoped_and_raw_preserved(store, pack, binding):
    c = Calibration(
        id="temperature",
        version="1",
        provider_hash=binding.provider.identity(),
        pack_id=pack.id,
        pack_version=pack.version,
        workload="default",
        scope="default",
        temperature=2.0,
    )
    calibrated = Binding(provider=binding.provider, calibration=c)
    result = Engine(store).evaluate(pack, {"text": "hi"}, calibrated)
    assert result.raw_distribution["false"] == 0.9
    assert result.confidence == pytest.approx(0.75)
    for changed, p, execution in [
        (Binding(provider=changed_binding(binding).provider, calibration=c), pack, None),
        (calibrated, pack.model_copy(update={"version": "2"}), None),
        (calibrated, pack, ExecutionPolicy(workload="other")),
        (calibrated, pack, ExecutionPolicy(scope="other")),
    ]:
        with pytest.raises(ReflexError, match="calibration"):
            Engine(store).evaluate(p, {"text": "hi"}, changed, execution=execution)


def test_extreme_temperature_stable(store, pack, binding):
    calibration = Calibration(
        id="sharp",
        version="1",
        provider_hash=binding.provider.identity(),
        pack_id=pack.id,
        pack_version="1",
        workload="default",
        scope="default",
        temperature=0.0001,
    )
    result = Engine(store).evaluate(
        pack, {"text": "hi"}, Binding(provider=binding.provider, calibration=calibration)
    )
    assert result.confidence == 1


def test_unknown_state_rejected_before_inference(store, pack, binding):
    provider = MockProvider(binding.provider)
    with pytest.raises(ReflexError, match="contract"):
        Engine(store, provider).evaluate(pack, {"text": "x", "typo": "y"}, binding)
    assert provider.calls == 0


def test_privacy_capabilities_no_silent_switch(store, pack, binding):
    data = binding.model_dump()
    data["provider"].update(adapter="jev", endpoint="https://invalid.example/jev")
    data["provider"]["capabilities"]["local"] = False
    remote = Binding.model_validate(data)
    with pytest.raises(ReflexError, match="privacy"):
        Engine(store).evaluate(
            pack, {"text": "x"}, remote, execution=ExecutionPolicy(local_only=True)
        )
    data = binding.model_dump()
    data["provider"]["capabilities"]["primitives"] = ["choice"]
    with pytest.raises(ReflexError, match="primitive"):
        Engine(store).evaluate(pack, {"text": "x"}, Binding.model_validate(data))


def test_malformed_distribution_never_cached(store, pack, binding):
    class Bad(MockProvider):
        def infer(self, requests):
            return {r.id: Prediction(distribution={"false": 0.5}) for r in requests}

    with pytest.raises(ReflexError, match="complete"):
        Engine(store, Bad(binding.provider)).evaluate(pack, {"text": "x"}, binding)
    assert not store.list("cache")


def test_score_expected_value(store, pack, binding):
    data = pack.model_dump(by_alias=True)
    data["judgment"].update(
        primitive="score",
        labels=["low", "mid", "high"],
        values={"low": 0.0, "mid": 0.5, "high": 1.0},
    )
    data["policy"] = {"kind": "none"}
    score = Pack.model_validate(data)
    result = Engine(store).evaluate(score, {"text": "x"}, binding)
    assert result.expected_score == pytest.approx(0.075)


def test_explicit_cascade_only(store, pack, binding):
    weak = changed_binding(binding, label="false", confidence=0.55)
    strong = changed_binding(binding, label="true", version="3")
    result = Engine(store).cascade(pack, {"text": "x"}, [weak, strong])
    assert result.selected == "true"
    assert len(result.execution["attempts"]) == 2
    assert result.execution["system2_required"] is False
    ordinary = Engine(store).evaluate(pack, {"text": "x"}, weak)
    assert "attempts" not in ordinary.execution


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(unknown=1),
        lambda d: d["judgment"].update(labels=["x", "x"]),
        lambda d: d["state"]["schema"].update(additionalProperties=True),
        lambda d: d["state"]["schema"].update({"$ref": "https://untrusted.invalid/schema"}),
    ],
)
def test_strict_pack_schema(pack, mutation):
    data = pack.model_dump(by_alias=True)
    mutation(data)
    with pytest.raises(ValidationError):
        Pack.model_validate(data)
