"""Durable lifecycle; every transition is fenced against a stale active deployment."""

import time
import uuid
from typing import Any, Protocol

from .engine import Engine
from .errors import ReflexError
from .evidence import REAL_KINDS, Evidence, deduplicate, utility
from .identity import digest
from .metrics import metrics
from .models import Deployment, ExecutionPolicy
from .store import Store


class Optimizer(Protocol):
    def propose(self, active: Deployment, evidence: list[dict[str, Any]]) -> Deployment | None: ...


class SuppliedProposal:
    """Offline strong-model or human proposal. Evidence authorization still applies."""

    def __init__(self, proposal: Deployment | None) -> None:
        self.proposal = proposal

    def propose(self, active: Deployment, evidence: list[dict[str, Any]]) -> Deployment | None:
        return self.proposal


class Lifecycle:
    def __init__(self, store: Store, engine: Engine | None = None) -> None:
        self.store = store
        self.engine = engine or Engine(store)
        self.evidence = Evidence(store)

    def key(self, pack_id: str) -> str:
        return self.store.project_key() + ":" + pack_id

    def active(self, pack_id: str) -> Deployment:
        return Deployment.model_validate(
            self.store.require("deployments", self.key(pack_id))["active"]
        )

    def activate(self, deployment: Deployment) -> dict[str, Any]:
        key = self.key(deployment.pack.id)
        data = deployment.model_dump(by_alias=True)
        with self.store.transaction() as db:
            old = self.store.get("deployments", key, db)
            if old:
                if old["active"] == data:
                    return {"pack": deployment.pack.id, "status": "already active"}
                raise ReflexError(
                    "active_exists", "active deployment exists; create a candidate for changes"
                )
            self._register(deployment, db)
            self.store.put(
                "deployments",
                key,
                {
                    "active": data,
                    "status": "ACTIVE",
                    "history": [],
                    "cooldown_until": 0.0,
                    "project": self.store.project_key(),
                },
                db,
            )
            self.store.event("activated", key, {"version": deployment.pack.version}, db)
        return {"pack": deployment.pack.id, "status": "ACTIVE"}

    def _register(self, deployment: Deployment, db: Any) -> None:
        pack = deployment.pack
        for component in ["state", "judgment", "policy", "evaluation"]:
            value = getattr(pack, component)
            self.store.put(
                "components",
                f"{pack.id}/{component}@{value.version}",
                value.model_dump(by_alias=True),
                db,
                immutable=True,
            )
        self.store.put(
            "packs", f"{pack.id}@{pack.version}", pack.model_dump(by_alias=True), db, immutable=True
        )
        spec = deployment.binding.provider
        self.store.put(
            "providers", f"{spec.id}@{spec.version}", spec.model_dump(), db, immutable=True
        )
        c = deployment.binding.calibration
        if c:
            self.store.put(
                "calibrations", f"{c.id}@{c.version}", c.model_dump(), db, immutable=True
            )

    def readiness(self, pack_id: str, minimum: int = 3) -> dict[str, Any]:
        active = self.active(pack_id)
        result = self.evidence.readiness(
            pack_id,
            minimum=minimum,
            version=active.pack.version,
            provider_hash=active.binding.provider.identity(),
        )
        if result["ready"]:
            with self.store.transaction() as db:
                key = self.key(pack_id)
                row = self.store.require("deployments", key, db)
                if (
                    row["active"] == active.model_dump(by_alias=True)
                    and row["status"] == "ACTIVE"
                    and row["cooldown_until"] <= time.time()
                ):
                    row["status"] = "OPTIMIZATION_READY"
                    self.store.put("deployments", key, row, db)
                    self.store.event("optimization_ready", key, result, db)
        return result

    def improve(
        self,
        pack_id: str,
        optimizer: Optimizer,
        *,
        scopes: list[str],
        approve: bool = False,
        minimum: int = 3,
        reason: str = "evidence-backed proposal",
    ) -> dict[str, Any]:
        self.readiness(pack_id, minimum)
        active = self.active(pack_id)
        rows = deduplicate(
            self.evidence.select(
                pack_id,
                "optimizable",
                scopes=scopes,
                mutate=True,
                version=active.pack.version,
                provider_hash=active.binding.provider.identity(),
            )
        )
        if not approve:
            raise ReflexError(
                "approval_required",
                "expensive optimization requires explicit approval",
                "reflex-axi improve --pack <id> --scope <name> --approve --proposal <file>",
            )
        if len(rows) < minimum:
            raise ReflexError(
                "evidence_authorization",
                "insufficient optimizable and mutation-authorized evidence; readiness never grants permission",
            )
        key = self.key(pack_id)
        owner = self.store.claim("optimization:" + key)
        if not owner:
            raise ReflexError(
                "optimization_busy", "optimization is running; expired workers can be resumed"
            )
        try:
            with self.store.transaction() as db:
                row = self.store.require("deployments", key, db)
                if row["cooldown_until"] > time.time():
                    raise ReflexError("cooldown", "optimization cooldown is active")
                if row["status"] not in {"OPTIMIZATION_READY", "OPTIMIZING"}:
                    raise ReflexError(
                        "not_ready", "recurring gaps are required before optimization"
                    )
                if row["active"] != active.model_dump(by_alias=True):
                    raise ReflexError("stale_active", "active deployment changed")
                row["status"] = "OPTIMIZING"
                self.store.put("deployments", key, row, db)
                self.store.event("optimizing", key, {"evidence": len(rows)}, db)
            proposal = optimizer.propose(active, rows)
            receipt = getattr(optimizer, "receipt", None)
            # Permissions may have been revoked during an expensive optimizer call.
            still_authorized = {
                r["outcome"]["id"]
                for r in self.evidence.select(
                    pack_id,
                    "optimizable",
                    scopes=scopes,
                    mutate=True,
                    version=active.pack.version,
                    provider_hash=active.binding.provider.identity(),
                )
            }
            if not {r["outcome"]["id"] for r in rows} <= still_authorized:
                raise ReflexError(
                    "evidence_authorization", "optimization permissions changed during the sweep"
                )
            with self.store.transaction() as db:
                if not self.store.owns("optimization:" + key, owner, db):
                    raise ReflexError(
                        "lease_lost", "optimizer lease expired; output was not accepted"
                    )
                row = self.store.require("deployments", key, db)
                if (
                    row["active"] != active.model_dump(by_alias=True)
                    or row["status"] != "OPTIMIZING"
                ):
                    raise ReflexError(
                        "stale_active", "active lifecycle changed during optimization"
                    )
                self.evidence.assert_authorized(
                    [r["outcome"]["id"] for r in rows], "optimizable", db, mutate=True
                )
                if receipt is not None:
                    self.store.put(
                        "optimization_sweeps",
                        owner,
                        {
                            "receipt": receipt,
                            "active_hash": digest(row["active"]),
                            "time": time.time(),
                        },
                        db,
                        immutable=True,
                    )
                if proposal is None or proposal == active:
                    row.update(status="ACTIVE", cooldown_until=time.time() + 60)
                    self.store.put("deployments", key, row, db)
                    self.store.event("no_update_warranted", key, {}, db)
                    return {"status": "no update warranted"}
                return self._candidate(row, active, proposal, rows, reason, "optimization", db)
        except Exception:
            with self.store.transaction() as db:
                row = self.store.require("deployments", key, db)
                if row["status"] == "OPTIMIZING" and self.store.owns(
                    "optimization:" + key, owner, db
                ):
                    row["status"] = "OPTIMIZATION_READY"
                    self.store.put("deployments", key, row, db)
                    self.store.event("optimization_failed", key, {}, db)
            raise
        finally:
            self.store.release("optimization:" + key, owner)

    def _candidate(
        self,
        row: dict[str, Any],
        active: Deployment,
        proposal: Deployment,
        evidence: list[dict[str, Any]],
        reason: str,
        kind: str,
        db: Any,
    ) -> dict[str, Any]:
        if proposal.pack.id != active.pack.id:
            raise ReflexError("candidate_invalid", "candidate must retain the pack ID")
        if (
            set(proposal.pack.judgment.labels) != set(active.pack.judgment.labels)
            or proposal.pack.judgment.primitive != active.pack.judgment.primitive
        ):
            raise ReflexError(
                "candidate_incompatible",
                "primitive or label changes require a new pack and explicit outcome migration",
            )
        signature = digest(
            {
                "proposal": proposal.model_dump(by_alias=True),
                "evidence": sorted(r["outcome"]["id"] for r in evidence),
            }
        )
        if self.store.get("rejections", signature, db):
            raise ReflexError(
                "proposal_rejected",
                "same proposal was rejected against the same evidence; provide materially new evidence",
            )
        self._register(proposal, db)
        candidate_id = uuid.uuid4().hex
        key = self.key(active.pack.id)
        candidate = {
            "id": candidate_id,
            "key": key,
            "pack": active.pack.id,
            "status": "CANDIDATE",
            "base": active.model_dump(by_alias=True),
            "deployment": proposal.model_dump(by_alias=True),
            "base_hash": digest(active.model_dump(by_alias=True)),
            "signature": signature,
            "evidence_ids": [r["outcome"]["id"] for r in evidence],
            "reason": reason,
            "kind": kind,
            "created": time.time(),
        }
        self.store.put("candidates", candidate_id, candidate, db)
        row.update(status="CANDIDATE", candidate=candidate_id)
        self.store.put("deployments", key, row, db)
        self.store.event("candidate_created", key, {"candidate": candidate_id, "kind": kind}, db)
        return {"id": candidate_id, "status": "CANDIDATE"}

    def candidate(self, candidate_id: str) -> dict[str, Any]:
        c = self.store.require("candidates", candidate_id)
        if c["key"] != self.key(c["pack"]):
            raise ReflexError("project_mismatch", "candidate belongs to another project directory")
        return c

    def _fresh(self, candidate: dict[str, Any], db: Any) -> dict[str, Any]:
        row = self.store.require("deployments", candidate["key"], db)
        if (
            digest(row["active"]) != candidate["base_hash"]
            or row.get("candidate") != candidate["id"]
        ):
            raise ReflexError(
                "stale_candidate", "candidate no longer targets the active deployment"
            )
        return row

    def replay(self, candidate_id: str, *, scopes: list[str]) -> dict[str, Any]:
        candidate = self.candidate(candidate_id)
        if candidate["status"] not in {"CANDIDATE", "SHADOWING", "PROMOTION_READY"}:
            raise ReflexError("transition", "only a candidate can enter replay")
        base = Deployment.model_validate(candidate["base"])
        proposed = Deployment.model_validate(candidate["deployment"])
        rows = self.evidence.select(base.pack.id, "evaluatable", scopes=scopes)
        protected = [
            r
            for r in self.evidence.select(base.pack.id, "evaluatable")
            if r["outcome"]["protected"]
        ]
        by_id = {r["outcome"]["id"]: r for r in rows + protected}
        rows = list(by_id.values())
        rows = [
            r
            for r in deduplicate(rows)
            if r["outcome"]["kind"] in REAL_KINDS and r["outcome"]["expected"] is not None
        ]
        # Regression locks are an explicitly authorized retained invariant, independent of training scopes.
        locks = self.store.list("regressions:" + candidate["key"])
        replay_ids = [r["outcome"]["id"] for r in rows]
        with self.store.connect() as db:
            guard = self.evidence.guard(replay_ids, candidate["key"], db)
        pairs: list[dict[str, Any]] = []
        base_samples = []
        candidate_samples = []
        failures = []
        for row in rows:
            state = row["decision"]["state"]
            old = self.engine.evaluate(
                base.pack, state, base.binding, execution=execution_for(base)
            )
            try:
                new = self.engine.evaluate(
                    proposed.pack, state, proposed.binding, execution=execution_for(proposed)
                )
                old_u = utility(base.pack, old.model_dump(), row["outcome"])
                new_u = utility(base.pack, new.model_dump(), row["outcome"])
                assert old_u is not None and new_u is not None
                base_samples.append((old.model_dump(), row["outcome"]))
                candidate_samples.append((new.model_dump(), row["outcome"]))
                pairs.append(
                    {
                        "id": row["outcome"]["id"],
                        "base": old_u,
                        "candidate": new_u,
                        "state": state,
                        "expected": row["outcome"]["expected"],
                    }
                )
                if row["outcome"]["protected"] and new_u < old_u:
                    failures.append(row["outcome"]["id"])
            except ReflexError:
                failures.append(row["outcome"]["id"])
        for lock in locks:
            try:
                value = self.engine.evaluate(
                    proposed.pack,
                    lock["state"],
                    proposed.binding,
                    execution=execution_for(proposed),
                )
                if value.selected != lock["expected"]:
                    failures.append(lock["id"])
            except ReflexError:
                failures.append(lock["id"])
        spec = base.pack.evaluation
        old_mean = sum(p["base"] for p in pairs) / len(pairs) if pairs else 0
        new_mean = sum(p["candidate"] for p in pairs) / len(pairs) if pairs else 0
        enough = len(pairs) >= spec.min_replay
        passed = (
            enough
            and not failures
            and new_mean >= spec.min_utility
            and new_mean + spec.max_regression >= old_mean
        )
        report = {
            "passed": passed,
            "count": len(pairs),
            "base_utility": old_mean,
            "candidate_utility": new_mean,
            "regressions": failures,
            "enough": enough,
            "base_metrics": metrics(base_samples),
            "candidate_metrics": metrics(candidate_samples),
        }
        with self.store.transaction() as db:
            current = self.store.require("candidates", candidate_id, db)
            deployment = self._fresh(current, db)
            if current["status"] != candidate["status"]:
                raise ReflexError("transition", "candidate changed during replay")
            if guard != self.evidence.guard(replay_ids, candidate["key"], db):
                raise ReflexError(
                    "stale_evidence", "evidence changed during replay; rerun evaluation"
                )
            self.evidence.assert_authorized(replay_ids, "evaluatable", db)
            current.update(
                replay=report, replay_pairs=pairs, replay_ids=replay_ids, replay_guard=guard
            )
            if passed:
                current["status"] = "SHADOWING"
                deployment["status"] = "SHADOWING"
                self.store.event("shadow_started", current["key"], {"candidate": candidate_id}, db)
            elif enough or failures:
                self._reject(current, deployment, "replay failed", db)
            self.store.put("candidates", candidate_id, current, db)
            self.store.put("deployments", current["key"], deployment, db)
        return report

    def shadow(self, candidate_id: str, state: dict[str, Any]) -> dict[str, Any]:
        c = self.candidate(candidate_id)
        if c["status"] != "SHADOWING":
            raise ReflexError("transition", "candidate must pass replay before shadowing")
        base, proposed = (
            Deployment.model_validate(c["base"]),
            Deployment.model_validate(c["deployment"]),
        )
        old = self.engine.evaluate(
            base.pack, state, base.binding, execution=execution_for(base), record=True
        )
        new = self.engine.evaluate(
            proposed.pack, state, proposed.binding, execution=execution_for(proposed), record=True
        )
        pair: dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "candidate_id": candidate_id,
            "base": old.id,
            "candidate": new.id,
            "agreement": old.selected == new.selected,
            "state_hash": digest(state),
            "time": time.time(),
        }
        with self.store.transaction() as db:
            current = self.store.require("candidates", candidate_id, db)
            self._fresh(current, db)
            if current["status"] != "SHADOWING":
                raise ReflexError("transition", "candidate changed during shadow execution")
            self.store.put("shadow:" + candidate_id, pair["id"], pair, db)
        return pair

    def assess_shadow(self, candidate_id: str, *, scopes: list[str]) -> dict[str, Any]:
        c = self.candidate(candidate_id)
        if c["status"] not in {"SHADOWING", "PROMOTION_READY"}:
            raise ReflexError("transition", "candidate is not shadowing")
        base = Deployment.model_validate(c["base"])
        evidence = deduplicate(self.evidence.select(c["pack"], "evaluatable", scopes=scopes))
        by_decision = {
            r["outcome"]["decision_id"]: r for r in evidence if r["outcome"]["kind"] in REAL_KINDS
        }
        shadow_pairs = self.store.list("shadow:" + candidate_id)
        shadow_decisions = [p[side] for p in shadow_pairs for side in ["base", "candidate"]]
        with self.store.connect() as db:
            shadow_guard = self.evidence.guard([], c["key"], db, shadow_decisions=shadow_decisions)
        measured = []
        base_samples = []
        candidate_samples = []
        seen_states = set()
        for pair in shadow_pairs:
            if pair["state_hash"] in seen_states:
                continue
            old, new = by_decision.get(pair["base"]), by_decision.get(pair["candidate"])
            if not old or not new:
                continue
            old_u = utility(base.pack, old["decision"]["result"], old["outcome"], observed=True)
            new_u = utility(base.pack, new["decision"]["result"], new["outcome"], observed=True)
            if old_u is None or new_u is None:
                continue
            seen_states.add(pair["state_hash"])
            measured.append((old_u, new_u, pair))
            base_samples.append((old["decision"]["result"], old["outcome"]))
            candidate_samples.append((new["decision"]["result"], new["outcome"]))
        n = len(measured)
        old_mean = sum(p[0] for p in measured) / n if n else 0
        new_mean = sum(p[1] for p in measured) / n if n else 0
        spec = base.pack.evaluation
        ready = (
            n >= spec.min_shadow
            and new_mean >= spec.min_utility
            and new_mean + spec.max_regression >= old_mean
        )
        report = {
            "ready": ready,
            "real_pairs": n,
            "base_utility": old_mean,
            "candidate_utility": new_mean,
            "agreement": sum(p[2]["agreement"] for p in measured) / n if n else None,
            "base_metrics": metrics(base_samples),
            "candidate_metrics": metrics(candidate_samples),
        }
        with self.store.transaction() as db:
            current = self.store.require("candidates", candidate_id, db)
            deployment = self._fresh(current, db)
            if current["status"] != c["status"]:
                raise ReflexError("transition", "candidate changed during shadow assessment")
            if shadow_guard != self.evidence.guard(
                [], c["key"], db, shadow_decisions=shadow_decisions
            ):
                raise ReflexError("stale_evidence", "evidence changed during shadow assessment")
            current["shadow_report"] = report
            current.update(shadow_guard=shadow_guard, shadow_decisions=shadow_decisions)
            if ready:
                current["status"] = deployment["status"] = "PROMOTION_READY"
                self.store.event("promotion_ready", c["key"], {"candidate": candidate_id}, db)
            elif n >= spec.min_shadow:
                self._reject(current, deployment, "shadow utility failed", db)
            else:
                current["status"] = deployment["status"] = "SHADOWING"
            self.store.put("candidates", candidate_id, current, db)
            self.store.put("deployments", c["key"], deployment, db)
        return report

    def promote(self, candidate_id: str, *, approve: bool = False) -> dict[str, Any]:
        c = self.candidate(candidate_id)
        with self.store.transaction() as db:
            c = self.store.require("candidates", candidate_id, db)
            if c["status"] == "PROMOTED":
                return {"id": candidate_id, "status": "already promoted"}
            if not approve:
                raise ReflexError(
                    "approval_required",
                    "production promotion requires explicit approval",
                    f"reflex-axi promote --candidate {candidate_id} --approve",
                )
            row = self._fresh(c, db)
            if c["status"] != "PROMOTION_READY":
                raise ReflexError(
                    "transition", "candidate must pass replay and real shadow evidence first"
                )
            if c["replay_guard"] != self.evidence.guard(c["replay_ids"], c["key"], db) or c[
                "shadow_guard"
            ] != self.evidence.guard([], c["key"], db, shadow_decisions=c["shadow_decisions"]):
                raise ReflexError(
                    "stale_evidence",
                    "evidence or permissions changed; rerun replay and shadow assessment",
                )
            self.evidence.assert_authorized(c["evidence_ids"], "optimizable", db, mutate=True)
            row["history"].append(row["active"])
            row.update(active=c["deployment"], status="ACTIVE", cooldown_until=0.0)
            row.pop("candidate", None)
            c["status"] = "PROMOTED"
            for pair in c.get("replay_pairs", []):
                if pair["candidate"] == 1 and pair["base"] < 1:
                    lock = {
                        "id": digest({"state": pair["state"], "expected": pair["expected"]}),
                        "state": pair["state"],
                        "expected": pair["expected"],
                        "source": pair["id"],
                    }
                    self.store.put("regressions:" + c["key"], lock["id"], lock, db, immutable=True)
            self.store.put("deployments", c["key"], row, db)
            self.store.put("candidates", candidate_id, c, db)
            self.store.event("promoted", c["key"], {"candidate": candidate_id}, db)
        return {"id": candidate_id, "status": "ACTIVE"}

    def _reject(self, candidate: dict[str, Any], row: dict[str, Any], reason: str, db: Any) -> None:
        candidate.update(status="REJECTED", rejection=reason)
        row.update(status="ACTIVE", cooldown_until=time.time() + 60)
        row.pop("candidate", None)
        self.store.put(
            "rejections",
            candidate["signature"],
            {"candidate": candidate["id"], "reason": reason},
            db,
        )
        self.store.event(
            "candidate_rejected",
            candidate["key"],
            {"candidate": candidate["id"], "reason": reason},
            db,
        )

    def reject(self, candidate_id: str, reason: str) -> dict[str, Any]:
        candidate = self.candidate(candidate_id)
        with self.store.transaction() as db:
            candidate = self.store.require("candidates", candidate_id, db)
            if candidate["status"] == "REJECTED":
                return {"id": candidate_id, "status": "already rejected"}
            row = self._fresh(candidate, db)
            self._reject(candidate, row, reason, db)
            self.store.put("candidates", candidate_id, candidate, db)
            self.store.put("deployments", candidate["key"], row, db)
        return {"id": candidate_id, "status": "REJECTED"}

    def rollback(self, pack_id: str, *, target: str | None = None) -> dict[str, Any]:
        key = self.key(pack_id)
        with self.store.transaction() as db:
            row = self.store.require("deployments", key, db)
            if target and digest(row["active"]) == target:
                return {"pack": pack_id, "status": "already active"}
            history = row["history"]
            matches = [h for h in history if target is None or digest(h) == target]
            if not matches:
                raise ReflexError(
                    "rollback_missing", "no retained deployment matches the rollback target"
                )
            previous = matches[-1]
            if row.get("candidate"):
                candidate = self.store.require("candidates", row["candidate"], db)
                self._reject(candidate, row, "active deployment rolled back", db)
                self.store.put("candidates", candidate["id"], candidate, db)
            history.append(row["active"])
            row.update(active=previous, status="ACTIVE", cooldown_until=time.time() + 60)
            self.store.put("deployments", key, row, db)
            self.store.event("rolled_back", key, {"deployment_hash": digest(previous)}, db)
        return {"pack": pack_id, "status": "ACTIVE", "deployment_hash": digest(previous)}


def execution_for(deployment: Deployment) -> ExecutionPolicy:
    return ExecutionPolicy(workload=deployment.workload, scope=deployment.scope)
