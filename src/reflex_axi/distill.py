"""Teacher datasets and replaceable local trainers, using the same production gates."""

import time
from typing import Any, Protocol

from .errors import ReflexError
from .evidence import REAL_KINDS, deduplicate
from .identity import digest
from .lifecycle import Lifecycle, execution_for
from .models import Binding, Capabilities, Deployment, ProviderSpec
from .store import Store


class Trainer(Protocol):
    id: str
    version: str

    def train(
        self, dataset: list[dict[str, Any]], active: Deployment, job_id: str
    ) -> ProviderSpec: ...


class FixtureTrainer:
    """Deterministic table student for exercising the full lifecycle, not a learned model."""

    id = "fixture-table"
    version = "1"

    def train(self, dataset: list[dict[str, Any]], active: Deployment, job_id: str) -> ProviderSpec:
        examples = {digest(row["state"]): row["distribution"] for row in dataset}
        labels = active.pack.judgment.labels
        prior = {
            label: sum(row["distribution"][label] for row in dataset) / len(dataset)
            for label in labels
        }
        return ProviderSpec(
            adapter="local-student",
            id="student-" + active.pack.id,
            model="fixture-table",
            version=job_id,
            capabilities=Capabilities(
                local=True,
                native_batch=True,
                multi_question=True,
                max_batch=1024,
                concurrency=4,
                requests_per_second=10000.0,
                deterministic=True,
                latency_class="low",
                cost_class="free",
            ),
            settings={"examples": examples, "prior": prior, "trainer_version": self.version},
        )


class Distillation:
    def __init__(self, store: Store, lifecycle: Lifecycle | None = None) -> None:
        self.store = store
        self.lifecycle = lifecycle or Lifecycle(store)

    def run(
        self, pack_id: str, trainer: Trainer, *, scopes: list[str], approve: bool = False
    ) -> dict[str, Any]:
        if not approve:
            raise ReflexError(
                "approval_required", "teacher collection and training require explicit approval"
            )
        active = self.lifecycle.active(pack_id)
        evidence = [
            r
            for r in deduplicate(
                self.lifecycle.evidence.select(
                    pack_id,
                    "optimizable",
                    scopes=scopes,
                    mutate=True,
                    version=active.pack.version,
                    provider_hash=active.binding.provider.identity(),
                )
            )
            if r["outcome"]["kind"] in REAL_KINDS
        ]
        if len(evidence) < max(
            active.pack.evaluation.min_replay, active.pack.evaluation.min_shadow
        ):
            raise ReflexError(
                "distillation_not_ready",
                "mature-pack distillation requires sufficient authorized real outcomes",
            )
        identity = digest(
            {
                "active": active.model_dump(by_alias=True),
                "trainer": [trainer.id, trainer.version],
                "evidence": sorted(r["outcome"]["id"] for r in evidence),
            }
        )
        job_id = identity[:24]
        lease_key = "distill:" + job_id
        owner = self.store.claim(lease_key, ttl=900)
        if not owner:
            raise ReflexError("job_busy", "distillation job already has a worker")
        try:
            job = self.store.get("distill_jobs", job_id)
            if job and job["status"] == "CANDIDATE":
                return {"job": job_id, "candidate": job["candidate"], "status": "CANDIDATE"}
            job = job or {
                "id": job_id,
                "identity": identity,
                "status": "DATASET",
                "trainer": trainer.id,
                "trainer_version": trainer.version,
            }
            self.store.put("distill_jobs", job_id, job)
            dataset = self.store.get("datasets", job_id)
            if dataset is None:
                dataset = []
                for row in evidence:
                    state = row["decision"]["state"]
                    result = self.lifecycle.engine.evaluate(
                        active.pack, state, active.binding, execution=execution_for(active)
                    )
                    dataset.append(
                        {
                            "state": active.pack.state.build(state),
                            "distribution": result.distribution,
                            "expected": row["outcome"]["expected"],
                            "evidence_id": row["outcome"]["id"],
                            "teacher_hash": result.inference_hash,
                        }
                    )
                self.store.put("datasets", job_id, dataset, immutable=True)
            job.update(status="TRAINING", dataset_hash=digest(dataset), examples=len(dataset))
            self.store.put("distill_jobs", job_id, job)
            student = trainer.train(dataset, active, job_id)
            if not student.capabilities.local:
                raise ReflexError(
                    "trainer_contract", "distillation must produce a local student provider"
                )
            proposed = Deployment(
                pack=active.pack,
                binding=Binding(provider=student),
                workload=active.workload,
                scope=active.scope,
            )
            current_ids = {
                r["outcome"]["id"]
                for r in self.lifecycle.evidence.select(
                    pack_id,
                    "optimizable",
                    scopes=scopes,
                    mutate=True,
                    version=active.pack.version,
                    provider_hash=active.binding.provider.identity(),
                )
            }
            if not {r["outcome"]["id"] for r in evidence} <= current_ids:
                raise ReflexError(
                    "evidence_authorization", "training evidence permissions were revoked"
                )
            with self.store.transaction() as db:
                if not self.store.owns(lease_key, owner, db):
                    raise ReflexError("lease_lost", "training lease expired; resume the job")
                self.lifecycle.evidence.assert_authorized(
                    [r["outcome"]["id"] for r in evidence], "optimizable", db, mutate=True
                )
                key = self.lifecycle.key(pack_id)
                row = self.store.require("deployments", key, db)
                if (
                    row["status"] != "ACTIVE"
                    or row["active"] != active.model_dump(by_alias=True)
                    or row["cooldown_until"] > time.time()
                ):
                    raise ReflexError(
                        "transition",
                        "distillation requires an unchanged active deployment outside cooldown",
                    )
                candidate_result = self.lifecycle._candidate(
                    row,
                    active,
                    proposed,
                    evidence,
                    "local student; utility, calibration, latency and cost require replay/shadow review",
                    "distillation",
                    db,
                )
                job.update(
                    status="CANDIDATE",
                    candidate=candidate_result["id"],
                    student=student.model_dump(),
                )
                self.store.put("distill_jobs", job_id, job, db)
                self.store.event(
                    "distillation_candidate",
                    key,
                    {"job": job_id, "candidate": candidate_result["id"]},
                    db,
                )
            return {"job": job_id, "candidate": candidate_result["id"], "status": "CANDIDATE"}
        finally:
            self.store.release(lease_key, owner)
