import json
import os
import subprocess
import sys

import pytest
from conftest import deployment, write_json

from reflex_axi.cli import parser
from reflex_axi.engine import Engine
from reflex_axi.evidence import Evidence
from reflex_axi.lifecycle import Lifecycle, SuppliedProposal
from reflex_axi.models import Outcome, Scope
from reflex_axi.store import Store


def test_cli_evaluate_decide_research_and_full(cli, tmp_path):
    state = write_json(tmp_path, "state.json", {"task": "judge", "context": "hello"})
    tiny = cli("evaluate", "--pack", "filter-context", "--state", state)
    assert set(tiny) == {"id", "pack", "selected", "confidence"}
    full = cli("decide", "--pack", "filter-context", "--state", state, "--policy", "none", "--full")
    assert full["policy_result"] is None
    assert full["execution"]["cache_hit"]
    research = cli("decide", "--pack", "filter-context", "--state", state, "--research", "--full")
    assert research["policy_result"] is None
    cli(
        "evaluate",
        "--pack",
        "filter-context",
        "--state",
        state,
        "--research",
        "--record",
        expected=2,
    )


def test_home_content_first_empty(cli):
    result = cli()
    assert result["count"] == 0
    assert "0 active deployments" in result["status"]
    assert result["bin"]
    assert result["description"]


def test_unknown_flags_before_state_or_network(tmp_path):
    root = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reflex_axi.entry",
            "evaluate",
            "--state",
            "missing",
            "--pack",
            "filter-context",
            "--stat",
            "x",
            "--state-root",
            str(root),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--stat" in result.stdout
    assert "valid flags" in result.stdout
    assert not result.stderr
    assert not root.exists()


def test_global_flags_before_command_are_preserved(tmp_path):
    root = tmp_path / "global-root"
    env = {**os.environ, "REFLEX_STATE_ROOT": str(tmp_path / "wrong-root")}
    result = subprocess.run(
        [sys.executable, "-m", "reflex_axi.entry", "--state-root", str(root), "--json", "status"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["count"] == 0
    assert root.exists()
    assert not (tmp_path / "wrong-root").exists()


@pytest.mark.parametrize("version", ["--version", "-v", "-V"])
def test_version_no_state_and_no_heavy_import(tmp_path, version):
    env = {**os.environ, "REFLEX_STATE_ROOT": str(tmp_path / "missing")}
    code = (
        "import sys; sys.argv=['reflex-axi',"
        + repr(version)
        + "]; from reflex_axi.entry import main; main(); assert 'pydantic' not in sys.modules; assert 'reflex_axi.cli' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, env=env)
    assert result.stdout == "0.1.0\n"
    assert result.returncode == 0
    assert not (tmp_path / "missing").exists()


def test_all_commands_have_help_without_storage(tmp_path):
    graph = parser()
    subcommands = next(
        a for a in graph._actions if hasattr(a, "choices") and isinstance(a.choices, dict)
    )
    for name in subcommands.choices:
        result = subprocess.run(
            [sys.executable, "-m", "reflex_axi.entry", name, "--help"],
            capture_output=True,
            text=True,
            env={**os.environ, "REFLEX_STATE_ROOT": str(tmp_path / "missing")},
        )
        assert result.returncode == 0, name
        assert "Examples:" in result.stdout
        assert "--help" in result.stdout
    assert not (tmp_path / "missing").exists()


def test_partial_batch_and_resume(cli, tmp_path):
    path = tmp_path / "states.jsonl"
    path.write_text(json.dumps({"task": "ok", "context": "a"}) + '\n{"typo":"bad"}\n')
    result = cli(
        "batch",
        "--pack",
        "filter-context",
        "--states",
        str(path),
        "--job",
        "test-job",
        "--results",
        expected=1,
    )
    assert result["completed"] == 1
    assert result["failed"] == 1
    rows = result["results"]
    assert [r["state_index"] for r in rows] == [0, 1]
    resumed = cli("batch", "--resume", "test-job", "--results", "--retry-failed", expected=1)
    assert resumed["results"][0]["result"]["id"] == rows[0]["result"]["id"]
    cli("batch", "--resume", "test-job", "--pack", "filter-context", expected=2)


def test_cli_bundle_export_and_fields(cli, tmp_path):
    state = write_json(tmp_path, "state.json", {"task": "judge", "context": "hello"})
    bundle = write_json(
        tmp_path,
        "bundle.json",
        {
            "id": "triage",
            "version": "1",
            "packs": ["result-sufficiency@1", "retry-worthiness@1", "escalation-need@1"],
        },
    )
    result = cli("evaluate", "--bundle", bundle, "--state", state)
    assert result["count"] == 3
    out = tmp_path / "private.json"
    result = cli(
        "evaluate",
        "--pack",
        "filter-context",
        "--state",
        state,
        "--fields",
        "distribution,provider",
        "--output",
        str(out),
    )
    assert result["status"] == "written"
    assert set(json.loads(out.read_text())) == {"distribution", "provider"}
    assert out.stat().st_mode & 0o777 == 0o600


def test_setup_snippets_only(cli, tmp_path):
    for app in ["codex", "claude", "opencode"]:
        output = tmp_path / (app + ".txt")
        result = cli("setup", "--app", app, "--output", str(output))
        assert result["status"] == "snippet written"
        assert "status" in output.read_text()


def test_cli_complete_lifecycle(cli, tmp_path, pack, binding):
    from conftest import changed_binding

    active = {"pack": pack.model_dump(by_alias=True), "binding": binding.model_dump()}
    active_path = write_json(tmp_path, "active.json", active)
    cli("activate", "--file", active_path)
    scope = write_json(
        tmp_path,
        "scope.json",
        {"name": "training", "evaluatable": True, "optimizable": True, "allowed_to_mutate": True},
    )
    cli("scopes", "--import", scope)
    for i in range(3):
        state = write_json(tmp_path, "state.json", {"text": f"sample-{i}"})
        result = cli("decide", "--pack", pack.id, "--state", state, "--full")
        outcome = write_json(
            tmp_path,
            "feedback.json",
            {
                "id": f"failure-{i}",
                "decision_id": result["id"],
                "scope": "training",
                "kind": "correction",
                "expected": "true",
            },
        )
        cli("feedback", "--file", outcome)
    proposed = {
        "pack": {**active["pack"], "version": "2"},
        "binding": changed_binding(binding).model_dump(),
    }
    proposal = write_json(tmp_path, "proposal.json", proposed)
    cid = cli(
        "improve", "--pack", pack.id, "--scope", "training", "--proposal", proposal, "--approve"
    )["id"]
    assert cli("eval", "--candidate", cid, "--scope", "training")["passed"]
    for i in range(3):
        state = write_json(tmp_path, "state.json", {"text": f"live-{i}"})
        pair = cli("shadow", "--candidate", cid, "--state", state)
        for side in ["base", "candidate"]:
            path = write_json(
                tmp_path,
                "feedback.json",
                {
                    "id": f"live-{i}-{side}",
                    "decision_id": pair[side],
                    "scope": "training",
                    "kind": "downstream",
                    "expected": "true",
                },
            )
            cli("feedback", "--file", path)
    assert cli("shadow", "--candidate", cid, "--assess", "--scope", "training")["ready"]
    revoked = write_json(tmp_path, "revoked.json", {"name": "training", "evaluatable": False})
    cli("scopes", "--import", revoked)
    assert cli("promote", "--candidate", cid, "--approve", expected=1)["code"] == "stale_evidence"
    cli("scopes", "--import", scope)
    cli("promote", "--candidate", cid, "--approve")
    assert cli("status")["active"][0]["version"] == "2"
    cli("rollback", "--pack", pack.id)
    assert cli("status")["active"][0]["version"] == "1"


def test_documented_examples_run(tmp_path, pack, binding):
    """The README's two runnable examples are public entry paths and must keep working."""
    root = tmp_path / "examples-state"
    env = {**os.environ, "REFLEX_STATE_ROOT": str(root)}
    library = subprocess.run(
        [sys.executable, "examples/library_usage.py"], env=env, text=True, capture_output=True
    )
    assert library.returncode == 0, (library.stdout, library.stderr)
    assert "decide policy:none: None" in library.stdout
    assert "'executed': False" in library.stdout
    assert "cache reuse: True True" in library.stdout
    assert "batch: COMPLETE 8 of 8" in library.stdout

    shim = tmp_path / "reflex-axi"
    shim.write_text(f'#!/bin/sh\nexec {sys.executable} -m reflex_axi.entry "$@"\n')
    shim.chmod(0o700)
    cursor = tmp_path / "cursor"

    def drain():
        done = subprocess.run(
            [
                sys.executable,
                "examples/event_watch.py",
                "--binary",
                str(shim),
                "--cursor",
                str(cursor),
                "--once",
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        assert done.returncode == 0, (done.stdout, done.stderr)
        return [json.loads(line) for line in done.stdout.splitlines()]

    # Routine local administration stays silent; the cursor still advances past it.
    store = Store(root)
    lifecycle = Lifecycle(store)
    lifecycle.activate(deployment(pack, binding))
    assert drain() == []
    activated = int(cursor.read_text())
    assert activated > 0

    Evidence(store).scope(
        Scope(
            name="training",
            readable=True,
            evaluatable=True,
            optimizable=True,
            allowed_to_mutate=True,
        )
    )
    engine = Engine(store)
    for index in range(3):
        result = engine.decide(pack, {"text": f"case {index}"}, binding)
        Evidence(store).feedback(
            Outcome(
                id=f"example-{index}",
                decision_id=result.id,
                scope="training",
                kind="correction",
                expected="true",
                gap="relevance",
            )
        )
    proposed = deployment(pack.model_copy(update={"version": "2"}), binding)
    lifecycle.improve(pack.id, SuppliedProposal(proposed), scopes=["training"], approve=True)

    kinds = [event["kind"] for event in drain()]
    assert kinds == ["optimization_ready", "optimizing", "candidate_created"]
    assert int(cursor.read_text()) > activated
    assert drain() == []  # the durable cursor never replays handled events
