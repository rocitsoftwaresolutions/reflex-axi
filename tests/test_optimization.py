import json
import subprocess
import sys

import pytest
from conftest import changed_binding
from test_lifecycle import seed, shadow_outcomes, start_candidate, training_scope
from test_providers import server

from reflex_axi.errors import ReflexError
from reflex_axi.evidence import Evidence
from reflex_axi.lifecycle import Lifecycle, SuppliedProposal
from reflex_axi.models import Deployment, Scope
from reflex_axi.optimization import StructuredOptimizer, SweepConfig


def test_one_strong_sweep_uses_only_authorized_evidence(store, pack, binding):
    lifecycle = Lifecycle(store)
    lifecycle.activate(Deployment(pack=pack, binding=binding))
    training_scope(store)
    seed(store, pack, binding)
    Evidence(store).scope(Scope(name="protected", evaluatable=True))
    seed(store, pack, binding, scope="protected", protected=True)
    proposal = Deployment(
        pack=pack.model_copy(update={"version": "2"}), binding=changed_binding(binding)
    )

    def respond(body):
        supplied = json.loads(body["messages"][1]["content"])
        assert len(supplied["evidence"]) == 3
        assert all(not r["outcome"]["protected"] for r in supplied["evidence"])
        content = {
            "deployment": proposal.model_dump(by_alias=True),
            "reason": "Fix recurring false exclusions",
            "evidence_ids": ["training-0", "training-1"],
        }
        return 200, {
            "model": "strong-pinned",
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"completion_tokens": 100},
        }

    with server(respond) as (endpoint, requests):
        optimizer = StructuredOptimizer(
            SweepConfig(endpoint=endpoint, model="strong-pinned", version="1")
        )
        result = lifecycle.improve(
            pack.id, optimizer, scopes=["training", "protected"], approve=True
        )
        assert result["status"] == "CANDIDATE"
        assert len(requests) == 1
        assert len(store.list("optimization_sweeps")) == 1


def test_optimizer_no_update_and_hallucinated_evidence(store, pack, binding):
    active = Deployment(pack=pack, binding=binding)
    training_scope(store)
    seed(store, pack, binding)
    rows = Evidence(store).select(pack.id, "optimizable")

    def response(ids):
        return {
            "model": "strong",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "deployment": None,
                                "reason": "No change is warranted",
                                "evidence_ids": ids,
                            }
                        )
                    }
                }
            ],
        }

    with server(lambda _: (200, response(["training-0"]))) as (endpoint, _):
        assert (
            StructuredOptimizer(
                SweepConfig(endpoint=endpoint, model="strong", version="1")
            ).propose(active, rows)
            is None
        )
    with (
        server(lambda _: (200, response(["invented"]))) as (endpoint, _),
        pytest.raises(ReflexError, match="authorized"),
    ):
        StructuredOptimizer(SweepConfig(endpoint=endpoint, model="strong", version="1")).propose(
            active, rows
        )


def test_protected_regression_cannot_be_omitted_by_scope(store, pack, binding):
    lifecycle, cid = start_candidate(store, pack, binding)
    Evidence(store).scope(Scope(name="protected", evaluatable=True))
    seed(store, pack, binding, scope="protected", protected=True, expected="false", count=1)
    assert not lifecycle.replay(cid, scopes=["training"])["passed"]
    assert lifecycle.candidate(cid)["status"] == "REJECTED"


def test_late_protected_regression_invalidates_promotion(store, pack, binding):
    lifecycle, cid = start_candidate(store, pack, binding)
    lifecycle.replay(cid, scopes=["training"])
    shadow_outcomes(store, lifecycle, cid)
    lifecycle.assess_shadow(cid, scopes=["training"])
    Evidence(store).scope(Scope(name="protected", evaluatable=True))
    seed(store, pack, binding, scope="protected", protected=True, expected="false", count=1)
    with pytest.raises(ReflexError, match="permissions changed"):
        lifecycle.promote(cid, approve=True)
    assert not lifecycle.replay(cid, scopes=["training"])["passed"]


def test_component_version_reuse_rejected(store, pack, binding):
    lifecycle = Lifecycle(store)
    lifecycle.activate(Deployment(pack=pack, binding=binding))
    training_scope(store)
    seed(store, pack, binding)
    changed = pack.model_copy(
        update={
            "version": "2",
            "judgment": pack.judgment.model_copy(update={"question": "Changed criteria"}),
        }
    )
    with pytest.raises(ReflexError, match="increment its version"):
        lifecycle.improve(
            pack.id,
            SuppliedProposal(Deployment(pack=changed, binding=binding)),
            scopes=["training"],
            approve=True,
        )


def test_optimizing_restart_after_process_crash(store, pack, binding, tmp_path):
    lifecycle = Lifecycle(store)
    lifecycle.activate(Deployment(pack=pack, binding=binding))
    training_scope(store)
    seed(store, pack, binding)
    script = """
import os,sys
from reflex_axi.store import Store
from reflex_axi.lifecycle import Lifecycle
class Crash:
 def propose(self,active,evidence): os._exit(75)
Lifecycle(Store(sys.argv[1])).improve('test-pack',Crash(),scopes=['training'],approve=True)
"""
    result = subprocess.run([sys.executable, "-c", script, str(store.root)])
    assert result.returncode == 75
    assert store.require("deployments", lifecycle.key(pack.id))["status"] == "OPTIMIZING"
    with store.transaction() as db:
        db.execute("UPDATE leases SET expires=0")
    restarted = Lifecycle(type(store)(store.root))
    assert (
        restarted.improve(pack.id, SuppliedProposal(None), scopes=["training"], approve=True)[
            "status"
        ]
        == "no update warranted"
    )


def test_replay_metrics_and_shadow_rejection(store, pack, binding):
    lifecycle, cid = start_candidate(store, pack, binding)
    replay = lifecycle.replay(cid, scopes=["training"])
    assert replay["candidate_metrics"]["brier"] < replay["base_metrics"]["brier"]
    assert replay["candidate_metrics"]["mean_cost"] == 0
    shadow_outcomes(store, lifecycle, cid, target="false")
    assert not lifecycle.assess_shadow(cid, scopes=["training"])["ready"]
    assert lifecycle.candidate(cid)["status"] == "REJECTED"


def test_project_evidence_isolation(store, pack, binding, tmp_path, monkeypatch):
    training_scope(store)
    seed(store, pack, binding)
    assert len(Evidence(store).select(pack.id, "readable")) == 3
    other = tmp_path / "other-project"
    other.mkdir()
    monkeypatch.chdir(other)
    assert Evidence(store).select(pack.id, "readable") == []
