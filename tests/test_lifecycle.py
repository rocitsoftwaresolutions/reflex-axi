import pytest
from conftest import changed_binding

from reflex_axi.distill import Distillation, FixtureTrainer
from reflex_axi.engine import Engine
from reflex_axi.errors import ReflexError
from reflex_axi.evidence import Evidence
from reflex_axi.identity import digest
from reflex_axi.lifecycle import Lifecycle, SuppliedProposal
from reflex_axi.models import Deployment, Outcome, Scope


def seed(store, pack, binding, *, scope="training", protected=False, count=3, expected="true"):
    evidence = Evidence(store)
    for i in range(count):
        result = Engine(store).evaluate(pack, {"text": f"{scope}-{i}"}, binding, record=True)
        evidence.feedback(
            Outcome(
                id=f"{scope}-{i}",
                decision_id=result.id,
                scope=scope,
                kind="correction",
                expected=expected,
                protected=protected,
            )
        )


def training_scope(store):
    Evidence(store).scope(
        Scope(name="training", evaluatable=True, optimizable=True, allowed_to_mutate=True)
    )


def start_candidate(store, pack, binding):
    lifecycle = Lifecycle(store)
    lifecycle.activate(Deployment(pack=pack, binding=binding))
    training_scope(store)
    seed(store, pack, binding)
    candidate_pack = pack.model_copy(update={"version": "2"})
    proposal = Deployment(pack=candidate_pack, binding=changed_binding(binding))
    candidate = lifecycle.improve(
        pack.id, SuppliedProposal(proposal), scopes=["training"], approve=True
    )
    return lifecycle, candidate["id"]


def shadow_outcomes(store, lifecycle, candidate_id, *, target="true", success=None):
    for i in range(3):
        pair = lifecycle.shadow(candidate_id, {"text": f"fresh-{i}"})
        for side in ["base", "candidate"]:
            Evidence(store).feedback(
                Outcome(
                    id=f"{candidate_id}-{i}-{side}",
                    decision_id=pair[side],
                    scope="training",
                    kind="downstream",
                    expected=target,
                    success=success,
                )
            )


def test_full_candidate_replay_shadow_promotion_rollback(store, pack, binding):
    lifecycle, cid = start_candidate(store, pack, binding)
    original = lifecycle.active(pack.id)
    with pytest.raises(ReflexError, match="approval"):
        lifecycle.promote(cid)
    with pytest.raises(ReflexError, match="replay"):
        lifecycle.promote(cid, approve=True)
    report = lifecycle.replay(cid, scopes=["training"])
    assert report["passed"]
    assert report["candidate_utility"] == 1
    assert report["base_utility"] == 0
    assert not lifecycle.assess_shadow(cid, scopes=["training"])["ready"]
    shadow_outcomes(store, lifecycle, cid)
    assert lifecycle.assess_shadow(cid, scopes=["training"])["ready"]
    lifecycle.promote(cid, approve=True)
    assert lifecycle.active(pack.id).pack.version == "2"
    assert len(store.list("regressions:" + lifecycle.key(pack.id))) == 3
    assert lifecycle.promote(cid, approve=True)["status"] == "already promoted"
    restarted = Lifecycle(type(store)(store.root))
    assert restarted.rollback(pack.id)["status"] == "ACTIVE"
    assert restarted.active(pack.id) == original
    assert (
        restarted.rollback(pack.id, target=digest(original.model_dump(by_alias=True)))["status"]
        == "already active"
    )
    assert {e["kind"] for e in store.events()} >= {
        "optimization_ready",
        "optimizing",
        "candidate_created",
        "shadow_started",
        "promotion_ready",
        "promoted",
        "rolled_back",
    }


def test_protected_observable_evidence_never_learnable(store, pack, binding):
    lifecycle = Lifecycle(store)
    lifecycle.activate(Deployment(pack=pack, binding=binding))
    Evidence(store).scope(
        Scope(name="protected", evaluatable=True, optimizable=True, allowed_to_mutate=True)
    )
    seed(store, pack, binding, scope="protected", protected=True)
    assert lifecycle.readiness(pack.id)["ready"]
    assert Evidence(store).select(pack.id, "evaluatable", scopes=["protected"])
    assert not Evidence(store).select(pack.id, "optimizable", scopes=["protected"], mutate=True)
    called = []

    class Spy:
        def propose(self, active, evidence):
            called.append(evidence)

    with pytest.raises(ReflexError, match="permission"):
        lifecycle.improve(pack.id, Spy(), scopes=["protected"], approve=True)
    assert not called


@pytest.mark.parametrize("permission", ["optimizable", "allowed_to_mutate"])
def test_readiness_not_authorization(store, pack, binding, permission):
    lifecycle = Lifecycle(store)
    lifecycle.activate(Deployment(pack=pack, binding=binding))
    grants = {
        "name": "training",
        "readable": True,
        "evaluatable": True,
        "optimizable": True,
        "allowed_to_mutate": True,
    }
    grants[permission] = False
    Evidence(store).scope(Scope.model_validate(grants))
    seed(store, pack, binding)
    assert lifecycle.readiness(pack.id)["ready"]
    with pytest.raises(ReflexError, match="insufficient"):
        lifecycle.improve(pack.id, SuppliedProposal(None), scopes=["training"], approve=True)


def test_no_update_and_cooldown(store, pack, binding):
    lifecycle = Lifecycle(store)
    lifecycle.activate(Deployment(pack=pack, binding=binding))
    training_scope(store)
    seed(store, pack, binding)
    assert (
        lifecycle.improve(pack.id, SuppliedProposal(None), scopes=["training"], approve=True)[
            "status"
        ]
        == "no update warranted"
    )
    with pytest.raises(ReflexError, match="cooldown"):
        lifecycle.improve(pack.id, SuppliedProposal(None), scopes=["training"], approve=True)


def test_optimizer_revocation_and_errors_restore_ready(store, pack, binding):
    lifecycle = Lifecycle(store)
    lifecycle.activate(Deployment(pack=pack, binding=binding))
    training_scope(store)
    seed(store, pack, binding)

    class Revoker:
        def propose(self, active, evidence):
            Evidence(store).scope(Scope(name="training"))
            return active

    with pytest.raises(ReflexError, match="permissions changed"):
        lifecycle.improve(pack.id, Revoker(), scopes=["training"], approve=True)
    assert store.require("deployments", lifecycle.key(pack.id))["status"] == "OPTIMIZATION_READY"


def test_automatic_replay_rejection_on_protected_regression(store, pack, binding):
    lifecycle, cid = start_candidate(store, pack, binding)
    Evidence(store).scope(Scope(name="protected", evaluatable=True))
    seed(store, pack, binding, scope="protected", protected=True, count=1, expected="false")
    report = lifecycle.replay(cid, scopes=["training", "protected"])
    assert not report["passed"]
    assert report["regressions"]
    assert lifecycle.candidate(cid)["status"] == "REJECTED"


def test_agreement_without_real_outcomes_cannot_promote(store, pack, binding):
    lifecycle, cid = start_candidate(store, pack, binding)
    lifecycle.replay(cid, scopes=["training"])
    for i in range(3):
        pair = lifecycle.shadow(cid, {"text": f"opinion-{i}"})
        for side in ["base", "candidate"]:
            Evidence(store).feedback(
                Outcome(
                    id=f"opinion-{side}-{i}",
                    decision_id=pair[side],
                    scope="training",
                    kind="retrospective",
                    expected="true",
                )
            )
    assert lifecycle.assess_shadow(cid, scopes=["training"])["real_pairs"] == 0


def test_repeated_shadow_state_not_independent_evidence(store, pack, binding):
    lifecycle, cid = start_candidate(store, pack, binding)
    lifecycle.replay(cid, scopes=["training"])
    for i in range(4):
        pair = lifecycle.shadow(cid, {"text": "same"})
        for side in ["base", "candidate"]:
            Evidence(store).feedback(
                Outcome(
                    id=f"repeated-{side}-{i}",
                    decision_id=pair[side],
                    scope="training",
                    kind="downstream",
                    expected="true",
                )
            )
    assert lifecycle.assess_shadow(cid, scopes=["training"])["real_pairs"] == 1


def test_evidence_idempotence_and_strict_scope(store, pack, binding):
    result = Engine(store).evaluate(pack, {"text": "x"}, binding, record=True)
    outcome = Outcome(id="o", decision_id=result.id, scope="missing", kind="test", success=True)
    with pytest.raises(ReflexError, match="not found"):
        Evidence(store).feedback(outcome)
    Evidence(store).scope(Scope(name="missing"))
    Evidence(store).feedback(outcome)
    assert Evidence(store).feedback(outcome)["status"] == "already recorded"
    with pytest.raises(ReflexError, match="different"):
        Evidence(store).feedback(outcome.model_copy(update={"success": False}))


def test_distillation_end_to_end_with_restart(store, pack, binding):
    binding = changed_binding(binding)
    lifecycle = Lifecycle(store)
    lifecycle.activate(Deployment(pack=pack, binding=binding))
    training_scope(store)
    seed(store, pack, binding)
    with pytest.raises(ReflexError, match="approval"):
        Distillation(store).run(pack.id, FixtureTrainer(), scopes=["training"])
    result = Distillation(store).run(pack.id, FixtureTrainer(), scopes=["training"], approve=True)
    cid = result["candidate"]
    assert store.require("datasets", result["job"])
    assert lifecycle.replay(cid, scopes=["training"])["passed"]
    shadow_outcomes(store, lifecycle, cid)
    assert lifecycle.assess_shadow(cid, scopes=["training"])["ready"]
    lifecycle.promote(cid, approve=True)
    student = lifecycle.active(pack.id)
    assert student.binding.provider.capabilities.local
    assert student.binding.calibration is None
    assert student.binding.provider.adapter == "local-student"
    assert Engine(store).evaluate(pack, {"text": "unseen"}, student.binding).selected == "true"
    lifecycle.rollback(pack.id)
    assert lifecycle.active(pack.id).binding == binding
