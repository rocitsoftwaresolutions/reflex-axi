"""Install the wheel into an isolated venv and exercise the actual console entry point."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

Path(".scratch").mkdir(exist_ok=True)
wheel = sorted(Path("dist").glob("reflex_axi-*.whl"))[-1].resolve()
with tempfile.TemporaryDirectory(dir=".scratch", prefix="installed-") as directory:
    root = Path(directory).resolve()
    venv = root / "venv"
    subprocess.run(["uv", "venv", str(venv)], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--cache-dir",
            ".scratch/uv-cache",
            "--python",
            str(venv / "bin/python"),
            str(wheel),
        ],
        check=True,
    )
    binary = venv / "bin/reflex-axi"
    env = {
        **os.environ,
        "REFLEX_STATE_ROOT": str(root / "state"),
        "PYTHONPATH": str(Path("tests/network_guard").resolve()),
    }
    for flag in ["-v", "-V", "--version"]:
        assert subprocess.check_output([str(binary), flag], env=env, text=True).strip() == "0.1.0"
    assert not (root / "state").exists()

    def invoke(*args):
        return json.loads(
            subprocess.check_output([str(binary), *args, "--json"], env=env, text=True)
        )

    assert invoke()["count"] == 0
    first = invoke(
        "decide",
        "--pack",
        "filter-context",
        "--state",
        "examples/state.json",
        "--policy",
        "none",
        "--full",
    )
    second = invoke(
        "evaluate", "--pack", "filter-context", "--state", "examples/state.json", "--full"
    )
    assert first["policy_result"] is None and second["execution"]["cache_hit"]
    bundle = invoke(
        "evaluate", "--bundle", "examples/bundle.json", "--state", "examples/state.json"
    )
    assert bundle["count"] == 3
    batch = invoke(
        "batch",
        "--bundle",
        "examples/bundle.json",
        "--states",
        "examples/states.jsonl",
        "--job",
        "installed",
    )
    assert batch["completed"] == 15
    assert invoke("batch", "--resume", "installed")["completed"] == 15
    registered = invoke("providers", "--import", "examples/openrouter-jev.json", "--full")
    assert registered["providers"][0]["adapter"] == "openrouter-jev"
    assert "unverified" in registered["openrouter_jev"]["revision"]
    refused = subprocess.run(
        [
            str(binary),
            "evaluate",
            "--provider",
            "examples/openrouter-jev.json",
            "--pack",
            "filter-context",
            "--state",
            "examples/state.json",
            "--research",
            "--json",
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert refused.returncode == 1 and json.loads(refused.stdout)["code"] == "provider_version"
    assert not refused.stderr
    assert invoke("benchmark", "list")["count"] == 4
    benchmark = invoke(
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
    assert benchmark["conformance"] == 1
    assert invoke("benchmark", "resume", "--run", benchmark["id"])["status"] == "COMPLETE"
    assert (root / "state").stat().st_mode & 0o777 == 0o700
print(
    "Installed wheel smoke: passed (clean state, version, policy:none, bundle, batch, resume, cache, provider benchmark)"
)
