import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from reflex_axi.catalog import default_binding
from reflex_axi.models import Binding, Deployment, Pack
from reflex_axi.store import Store


def pytest_configure(config):
    # pyproject sets --basetemp=.scratch/pytest; pytest creates only the leaf, so a fresh
    # clone without .scratch/ would error at every tmp_path setup.
    if config.option.basetemp:
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)


def pytest_make_parametrize_id(config, val, argname):
    # Oversized payloads would otherwise become multi-megabyte node IDs, and printing
    # them (failure summaries, -v) stalls CI log processing for hours.
    if isinstance(val, bytes | str) and len(val) > 100:
        return f"{argname}-{type(val).__name__}-{len(val)}"
    return None


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "state")


@pytest.fixture
def pack():
    return Pack.model_validate(
        {
            "id": "test-pack",
            "version": "1",
            "state": {
                "version": "1",
                "schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
            "judgment": {
                "version": "1",
                "primitive": "binary",
                "labels": ["false", "true"],
                "question": "Is the text relevant?",
            },
            "policy": {
                "version": "1",
                "kind": "recommend",
                "min_confidence": 0.8,
                "actions": {"true": "include", "false": "exclude"},
            },
        }
    )


@pytest.fixture
def binding():
    return default_binding()


def changed_binding(binding, label="true", version="2", **settings):
    data = binding.model_dump()
    data["provider"]["version"] = version
    data["provider"]["settings"] = {"label": label, **settings}
    return Binding.model_validate(data)


def deployment(pack, binding):
    return Deployment(pack=pack, binding=binding)


@pytest.fixture
def cli(tmp_path):
    root = tmp_path / "cli-state"

    def invoke(*args, expected=0, stdin=None):
        result = subprocess.run(
            [sys.executable, "-m", "reflex_axi.entry", *args, "--state-root", str(root), "--json"],
            text=True,
            input=stdin,
            capture_output=True,
        )
        assert result.returncode == expected, (result.stdout, result.stderr)
        assert not result.stderr
        return json.loads(result.stdout)

    return invoke


def write_json(tmp_path, name, value):
    path = Path(tmp_path) / name
    path.write_text(json.dumps(value))
    return str(path)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent / "network_guard"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not open network sockets; inject a transport fixture")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
