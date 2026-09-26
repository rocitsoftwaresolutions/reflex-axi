import json
import subprocess
import sys
from pathlib import Path

import pytest

from reflex_axi.catalog import default_binding
from reflex_axi.models import Binding, Deployment, Pack
from reflex_axi.store import Store


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
