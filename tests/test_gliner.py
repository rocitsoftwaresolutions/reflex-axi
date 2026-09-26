import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from reflex_axi.engine import Engine
from reflex_axi.errors import ReflexError
from reflex_axi.gliner import MODEL_FILES, RUNTIME_PACKAGES, GLiNERProvider, file_hash
from reflex_axi.models import Binding, Capabilities, ExecutionPolicy, ProviderSpec
from reflex_axi.providers import Request, provider_from_spec


@pytest.fixture
def gliner_spec(tmp_path):
    for name in MODEL_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("offline model fixture " + name)
    return ProviderSpec(
        adapter="gliner",
        id="gliner-fixture",
        model="fastino/GLiNER2.5-Decide",
        version="fixture-v1",
        capabilities=Capabilities(
            local=True,
            native_batch=True,
            max_batch=8,
            concurrency=1,
            deterministic=True,
            seeded=True,
            cost_class="free",
            requests_per_second=10000.0,
        ),
        settings={
            "model_path": str(tmp_path),
            "files": {name: file_hash(tmp_path / name) for name in MODEL_FILES},
            "runtime": {
                name: "2.0.0" if name == "gliner2" else "fixture" for name in RUNTIME_PACKAGES
            },
            "max_len": 1024,
        },
    )


@pytest.fixture
def runtime(monkeypatch, gliner_spec):
    calls = []

    class Schema:
        def single(self, name, labels, **kwargs):
            self.name, self.labels, self.kwargs = name, labels, kwargs
            return self

    class Classifier:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == gliner_spec.settings["model_path"]
            assert kwargs == {"local_files_only": True, "device": "cpu", "dtype": "float32"}
            calls.append("load")
            return cls()

        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def batch_score(self, texts, schema, *, config):
            calls.append((texts, schema, config))
            # Public 2.0.0 ClassificationScores contract: text + task -> label logits.
            return [
                SimpleNamespace(
                    text=text,
                    tasks={
                        "judgment": {label: float(i * 2) for i, label in enumerate(schema.labels)}
                    },
                )
                for text in texts
            ]

    torch = SimpleNamespace(
        manual_seed=lambda value: calls.append(("seed", value)),
        use_deterministic_algorithms=lambda value: calls.append(("deterministic", value)),
        set_num_threads=lambda value: calls.append(("threads", value)),
    )
    modules = {
        "torch": torch,
        "gliner2.classification.engine": SimpleNamespace(
            Classifier=Classifier, ClassificationConfig=SimpleNamespace
        ),
        "gliner2.classification.schema": SimpleNamespace(ClassificationSchema=Schema),
    }
    monkeypatch.setattr(
        "reflex_axi.gliner.runtime_versions", lambda: gliner_spec.settings["runtime"]
    )
    import importlib

    original_import = importlib.import_module
    monkeypatch.setattr(
        "reflex_axi.gliner.importlib.import_module",
        lambda name: modules[name] if name in modules else original_import(name),
    )
    return calls


def test_realistic_logits_boundary(store, gliner_spec, runtime, pack):
    provider = provider_from_spec(gliner_spec)
    engine = Engine(store, provider)
    results = engine.evaluate_many(
        [(pack, {"text": "Relevant"}), (pack, {"text": "Another"})],
        Binding(provider=gliner_spec),
        execution=ExecutionPolicy(cache=False),
    )
    assert all(r.selected == "true" for r in results)
    assert results[0].raw_distribution["true"] == pytest.approx(1 / (1 + math.exp(-2)))
    assert all(r.cost == 0 for r in results)
    assert results[0].inference_hash != results[1].inference_hash
    assert runtime.count("load") == 1
    assert ("seed", 0) in runtime
    assert runtime[-1][1].kwargs["instruction"] == pack.judgment.question
    assert len(runtime[-1][0]) == 2


@pytest.mark.parametrize("mutation", ["nan", "missing", "reordered", "omitted"])
def test_invalid_score_contract(gliner_spec, runtime, pack, mutation):
    provider = GLiNERProvider(gliner_spec)
    provider._load()
    a, b = Request("a", pack, {"text": "a"}), Request("b", pack, {"text": "b"})
    original = provider.classifier.batch_score

    def broken(*args, **kwargs):
        rows = original(*args, **kwargs)
        if mutation == "nan":
            rows[0].tasks["judgment"]["true"] = float("nan")
        elif mutation == "missing":
            del rows[0].tasks["judgment"]["true"]
        elif mutation == "reordered":
            rows.reverse()
        else:
            rows.pop()
        return rows

    provider.classifier.batch_score = broken
    with pytest.raises(ReflexError) as error:
        provider.infer([a, b])
    assert error.value.code in {"provider_distribution", "provider_association"}


def test_model_and_runtime_identity_checked_before_load(gliner_spec, runtime, pack, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr("reflex_axi.gliner.runtime_versions", lambda: {})
    with pytest.raises(ReflexError, match="runtime"):
        GLiNERProvider(gliner_spec).infer([Request("a", pack, {"text": "a"})])
    monkeypatch.setattr(
        "reflex_axi.gliner.runtime_versions", lambda: gliner_spec.settings["runtime"]
    )
    Path(gliner_spec.settings["model_path"], "model.safetensors").write_text("changed")
    with pytest.raises(ReflexError, match="manifest"):
        GLiNERProvider(gliner_spec).infer([Request("a", pack, {"text": "a"})])
    assert "load" not in runtime


def test_dependency_missing_normalized(gliner_spec, pack, monkeypatch):
    def missing():
        raise ReflexError("provider_dependency", "missing optional runtime")

    monkeypatch.setattr("reflex_axi.gliner.runtime_versions", missing)
    with pytest.raises(ReflexError) as error:
        GLiNERProvider(gliner_spec).infer([Request("a", pack, {"text": "a"})])
    assert error.value.code == "provider_dependency"


def test_capabilities_and_settings_cannot_lie(gliner_spec):
    for field, value in [
        ("local", False),
        ("multi_question", True),
        ("concurrency", 2),
        ("max_batch", 9),
        ("seeded", False),
    ]:
        data = gliner_spec.model_dump()
        data["capabilities"][field] = value
        with pytest.raises(ValidationError):
            ProviderSpec.model_validate(data)
    data = gliner_spec.model_dump()
    data["settings"]["unknown"] = "reject"
    with pytest.raises(ValidationError):
        ProviderSpec.model_validate(data)


def test_long_input_rejected_without_silent_truncation(gliner_spec, runtime, pack):
    with pytest.raises(ReflexError) as error:
        GLiNERProvider(gliner_spec).infer([Request("a", pack, {"text": "x" * 2000})])
    assert error.value.code == "state_too_large"


def test_primitives_and_seed_identity(gliner_spec, runtime):
    from reflex_axi.catalog import builtin_packs

    provider = GLiNERProvider(gliner_spec)
    for pack in builtin_packs():
        result = provider.infer([Request(pack.id, pack, {"task": "bounded", "context": "short"})])[
            pack.id
        ]
        assert set(result.distribution) == set(pack.judgment.labels)
        assert sum(result.distribution.values()) == pytest.approx(1)
    altered = gliner_spec.model_dump()
    altered["settings"]["seed"] = 1
    assert ProviderSpec.model_validate(altered).identity() != gliner_spec.identity()


def test_gliner_fixture_runs_benchmark(store, gliner_spec, runtime):
    from reflex_axi.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(store, GLiNERProvider(gliner_spec))
    result = runner.run(
        Binding(provider=gliner_spec), selected=["starter"], modes=["single", "batch"], repeats=2
    )
    assert result["status"] == "COMPLETE"
    assert result["summary"]["conformance"]["consistency"]["repeat"]["agreement"] == 1
    assert result["summary"]["conformance"]["consistency"]["execution_mode"]["agreement"] == 1
    assert result["summary"]["performance"]["max_native_batch"] > 1
    assert result["summary"]["performance"]["max_observed_inflight"] == 1
