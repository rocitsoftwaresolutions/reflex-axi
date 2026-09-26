"""Bounded chunks, durable per-cell checkpoints and deterministic result association."""

import time
from typing import Any

from .engine import Engine
from .errors import ReflexError
from .identity import digest
from .models import Binding, ExecutionPolicy, Pack
from .store import Store


class BatchRunner:
    def __init__(self, store: Store, engine: Engine | None = None) -> None:
        self.store = store
        self.engine = engine or Engine(store)

    def run(
        self,
        states: list[dict[str, Any]],
        packs: list[Pack],
        binding: Binding,
        *,
        job_id: str | None = None,
        execution: ExecutionPolicy | None = None,
        decide: bool = False,
        record: bool = False,
        retry_failed: bool = False,
        chunk_size: int = 128,
    ) -> dict[str, Any]:
        if not packs or not 1 <= chunk_size <= 10000:
            raise ReflexError("usage", "batch needs packs and chunk_size in [1,10000]")
        policy = execution or ExecutionPolicy()
        definition = {
            "project": self.store.project_key(),
            "states": states,
            "packs": [p.model_dump(by_alias=True) for p in packs],
            "binding": binding.model_dump(),
            "execution": policy.model_dump(),
            "decide": decide,
            "record": record,
        }
        identity = digest(definition)
        job_id = job_id or identity[:24]
        with self.store.transaction() as db:
            existing = self.store.get("jobs", job_id, db)
            if existing and existing["identity"] != identity:
                raise ReflexError(
                    "job_conflict", "job ID belongs to different inputs or versions; use a new ID"
                )
            if not existing:
                self.store.put(
                    "jobs",
                    job_id,
                    {
                        "id": job_id,
                        "identity": identity,
                        "definition": definition,
                        "status": "PENDING",
                        "total": len(states) * len(packs),
                    },
                    db,
                )
        return self.resume(job_id, retry_failed=retry_failed, chunk_size=chunk_size)

    def resume(
        self, job_id: str, *, retry_failed: bool = False, chunk_size: int = 128
    ) -> dict[str, Any]:
        if not 1 <= chunk_size <= 10000:
            raise ReflexError("usage", "chunk_size must be in [1,10000]")
        owner = self.store.claim("job:" + job_id, ttl=900)
        if not owner:
            raise ReflexError("job_busy", "job has an active worker; retry after its lease expires")
        try:
            job = self.store.require("jobs", job_id)
            definition = job["definition"]
            if definition["project"] != self.store.project_key():
                raise ReflexError("project_mismatch", "job belongs to another project directory")
            packs = [Pack.model_validate(p) for p in definition["packs"]]
            binding = Binding.model_validate(definition["binding"])
            execution = ExecutionPolicy.model_validate(definition["execution"])
            pending = []
            for state_index, state in enumerate(definition["states"]):
                for pack_index, pack in enumerate(packs):
                    cell = f"{state_index}:{pack_index}"
                    previous = self.store.get("job:" + job_id, cell)
                    if previous is None or (retry_failed and previous["status"] == "failed"):
                        pending.append((cell, state_index, pack_index, pack, state))
            job["status"] = "RUNNING"
            self.store.put("jobs", job_id, job)
            for offset in range(0, len(pending), chunk_size):
                chunk = pending[offset : offset + chunk_size]
                values = self.engine.evaluate_many(
                    [(c[3], c[4]) for c in chunk],
                    binding,
                    execution=execution,
                    record=definition["record"],
                    decide=definition["decide"],
                )
                with self.store.transaction() as db:
                    if not self.store.owns("job:" + job_id, owner, db):
                        raise ReflexError(
                            "lease_lost", "batch worker lease expired; resume the job"
                        )
                    for (cell, si, pi, _, _), value in zip(chunk, values, strict=True):
                        result = {
                            "cell": cell,
                            "state_index": si,
                            "pack_index": pi,
                            "status": "failed" if isinstance(value, ReflexError) else "complete",
                        }
                        result["error" if isinstance(value, ReflexError) else "result"] = (
                            {"code": value.code, "message": str(value)}
                            if isinstance(value, ReflexError)
                            else value.model_dump()
                        )
                        self.store.put("job:" + job_id, cell, result, db)
                    db.execute(
                        "UPDATE leases SET expires=? WHERE key=? AND owner=?",
                        (time.time() + 900, "job:" + job_id, owner),
                    )
            cells = self.results(job_id)
            failed = sum(c["status"] == "failed" for c in cells)
            job.update(
                status="PARTIAL" if failed else "COMPLETE",
                completed=len(cells) - failed,
                failed=failed,
            )
            with self.store.transaction() as db:
                if not self.store.owns("job:" + job_id, owner, db):
                    raise ReflexError("lease_lost", "batch worker lease expired")
                self.store.put("jobs", job_id, job, db)
            return {k: v for k, v in job.items() if k != "definition"}
        finally:
            self.store.release("job:" + job_id, owner)

    def results(self, job_id: str) -> list[dict[str, Any]]:
        return sorted(
            self.store.list("job:" + job_id), key=lambda c: (c["state_index"], c["pack_index"])
        )
