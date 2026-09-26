from importlib.resources import files
from pathlib import Path
from typing import Any

from .identity import read_json
from .models import Binding, Capabilities, Pack, ProviderSpec
from .store import Store

DESCRIPTION = (
    "Fast judgment before expensive thought; bounded probabilities, optional policy, no actions."
)
GUIDANCE = [
    "reflex-axi evaluate --pack filter-context --state state.json",
    "reflex-axi batch --bundle bundle.json --states states.jsonl",
    "reflex-axi status --full",
]


def default_binding() -> Binding:
    return Binding(
        provider=ProviderSpec(
            adapter="mock",
            id="mock",
            model="fixture",
            version="1",
            capabilities=Capabilities(
                local=True,
                native_batch=True,
                multi_question=True,
                max_batch=128,
                requests_per_second=10000.0,
                deterministic=True,
                latency_class="low",
                cost_class="free",
            ),
        )
    )


def builtin_packs() -> list[Pack]:
    root = files("reflex_axi").joinpath("packs")
    return [
        Pack.model_validate(read_json(p.read_text()))
        for p in root.iterdir()
        if p.name.endswith(".json")
    ]


def load_file(path: str) -> Any:
    return read_json(Path(path).read_text())


def load_pack(reference: str, store: Store) -> Pack:
    if Path(reference).is_file():
        return Pack.model_validate(load_file(reference))
    if "@" not in reference:
        row = store.get("deployments", store.project_key() + ":" + reference)
        if row:
            return Pack.model_validate(row["active"]["pack"])
    for pack in builtin_packs():
        if reference in {pack.id, f"{pack.id}@{pack.version}"}:
            registered = store.get("packs", f"{pack.id}@{pack.version}")
            return Pack.model_validate(registered) if registered else pack
    return Pack.model_validate(store.require("packs", reference))


def load_binding(reference: str | None, store: Store, pack: Pack | None = None) -> Binding:
    if reference == "mock":
        return default_binding()
    if reference:
        data = (
            load_file(reference)
            if Path(reference).is_file()
            else store.require("providers", reference)
        )
        return (
            Binding.model_validate(data)
            if "provider" in data
            else Binding(provider=ProviderSpec.model_validate(data))
        )
    if pack:
        row = store.get("deployments", store.project_key() + ":" + pack.id)
        if row and row["active"]["pack"]["version"] == pack.version:
            return Binding.model_validate(row["active"]["binding"])
    return default_binding()
