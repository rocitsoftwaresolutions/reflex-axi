import time
from typing import Any, Literal

from .errors import ReflexError
from .models import Outcome, Pack, Scope
from .store import Store

REAL_KINDS = {"proof", "test", "downstream", "correction", "metric", "delayed"}


class Evidence:
    def __init__(self, store: Store) -> None:
        self.store = store

    def scope(self, scope: Scope) -> None:
        # Grants are explicit local administration, never inferred from evidence quantity.
        with self.store.transaction() as db:
            self.store.put("scopes", scope.name, scope.model_dump(), db)
            self.store.event("scope_permissions", scope.name, scope.model_dump(), db)

    def feedback(self, outcome: Outcome) -> dict[str, Any]:
        with self.store.transaction() as db:
            self.store.require("scopes", outcome.scope, db)
            decision = self.store.require("decisions", outcome.decision_id, db)
            if decision.get("project") != self.store.project_key():
                raise ReflexError(
                    "project_mismatch", "decision belongs to another project directory"
                )
            pack = Pack.model_validate(decision["pack"])
            if outcome.expected is not None and outcome.expected not in pack.judgment.labels:
                raise ReflexError("outcome_invalid", "expected outcome is not a judgment label")
            existing = self.store.get("evidence", outcome.id, db)
            if existing:
                if existing["outcome"] != outcome.model_dump():
                    raise ReflexError(
                        "evidence_conflict", "outcome ID already exists with different content"
                    )
                return {"id": outcome.id, "status": "already recorded"}
            self.store.put(
                "evidence",
                outcome.id,
                {
                    "outcome": outcome.model_dump(),
                    "time": time.time(),
                    "pack": pack.id,
                    "version": pack.version,
                },
                db,
                immutable=True,
            )
        return {"id": outcome.id, "status": "recorded"}

    def select(
        self,
        pack_id: str,
        permission: Literal["readable", "evaluatable", "optimizable"],
        *,
        scopes: list[str] | None = None,
        mutate: bool = False,
        version: str | None = None,
        provider_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        result = []
        with self.store.transaction() as db:
            grants = {s["name"]: s for s in self.store.list("scopes", db)}
            for row in self.store.list("evidence", db):
                outcome = row["outcome"]
                grant = grants.get(outcome["scope"], {})
                if (
                    row["pack"] != pack_id
                    or (version and row["version"] != version)
                    or (scopes is not None and outcome["scope"] not in scopes)
                ):
                    continue
                if not grant.get(permission) or (mutate and not grant.get("allowed_to_mutate")):
                    continue
                if permission == "optimizable" and outcome["protected"]:
                    continue
                decision = self.store.require("decisions", outcome["decision_id"], db)
                if decision.get("project") != self.store.project_key():
                    continue
                if provider_hash and decision["result"]["provider"]["hash"] != provider_hash:
                    continue
                result.append({**row, "decision": decision})
        return result

    def assert_authorized(
        self, ids: list[str], permission: str, db: Any, *, mutate: bool = False
    ) -> None:
        for identity in ids:
            row = self.store.require("evidence", identity, db)
            outcome = row["outcome"]
            scope = self.store.require("scopes", outcome["scope"], db)
            if (
                not scope[permission]
                or (mutate and not scope["allowed_to_mutate"])
                or (permission == "optimizable" and outcome["protected"])
            ):
                raise ReflexError(
                    "stale_evidence", "evidence permissions changed; rerun authorized evaluation"
                )

    def guard(
        self, ids: list[str], pack_key: str, db: Any, *, shadow_decisions: list[str] | None = None
    ) -> str:
        selected = set(ids)
        decisions = set(shadow_decisions or [])
        pack_id = pack_key.split(":", 1)[1]
        rows = []
        for row in self.store.list("evidence", db):
            outcome = row["outcome"]
            scope = self.store.require("scopes", outcome["scope"], db)
            decision = self.store.require("decisions", outcome["decision_id"], db)
            if decision.get("project") != self.store.project_key():
                continue
            if (
                outcome["id"] in selected
                or outcome["decision_id"] in decisions
                or (row["pack"] == pack_id and outcome["protected"] and scope["evaluatable"])
            ):
                rows.append({"outcome": outcome, "scope": scope})
        from .identity import digest

        return digest({"evidence": rows, "locks": self.store.list("regressions:" + pack_key, db)})

    def readiness(
        self,
        pack_id: str,
        *,
        minimum: int = 3,
        version: str | None = None,
        provider_hash: str | None = None,
    ) -> dict[str, Any]:
        if minimum < 1:
            raise ReflexError("usage", "readiness minimum must be positive")
        rows = self.select(pack_id, "readable", version=version, provider_hash=provider_hash)
        gaps: dict[str, set[str]] = {}
        observed = set()
        for row in rows:
            o = row["outcome"]
            observed.add(o["decision_id"])
            result = row["decision"]["result"]
            failed = o["success"] is False or (
                o["expected"] is not None and o["expected"] != result["selected"]
            )
            if failed:
                gaps.setdefault(o["gap"], set()).add(o["decision_id"])
        ready = any(len(ids) >= minimum for ids in gaps.values())
        real = [r for r in deduplicate(rows) if r["outcome"]["kind"] in REAL_KINDS]
        real.sort(key=lambda r: r["time"])
        labeled = [r for r in real if r["outcome"]["expected"] is not None]
        drift = None
        if len(labeled) >= minimum * 2:
            earlier, recent = labeled[:-minimum], labeled[-minimum:]
            baseline = sum(
                r["decision"]["result"]["selected"] == r["outcome"]["expected"] for r in earlier
            ) / len(earlier)
            latest = sum(
                r["decision"]["result"]["selected"] == r["outcome"]["expected"] for r in recent
            ) / len(recent)
            drift = {
                "baseline_accuracy": baseline,
                "recent_accuracy": latest,
                "drop": baseline - latest,
                "detected": baseline - latest >= 0.2,
            }
        return {
            "ready": ready,
            "observed": len(observed),
            "gaps": {k: len(v) for k, v in gaps.items()},
            "minimum": minimum,
            "authorization": "separate",
            "drift": drift,
        }


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Corrections and deterministic proofs outrank opinion. Latest evidence wins within a class.
    rank = {
        "comparison": 0,
        "retrospective": 1,
        "override": 2,
        "metric": 3,
        "downstream": 4,
        "delayed": 4,
        "test": 5,
        "proof": 5,
        "correction": 6,
    }
    selected: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: (rank[r["outcome"]["kind"]], r["time"])):
        selected[row["outcome"]["decision_id"]] = row
    return list(selected.values())


def utility(
    pack: Pack, result: dict[str, Any], outcome: dict[str, Any], *, observed: bool = False
) -> float | None:
    if outcome["expected"] is not None:
        if pack.evaluation.utility == "accuracy":
            return float(result["selected"] == outcome["expected"])
        values = pack.judgment.values or {"false": 0.0, "true": 1.0}
        target = values[outcome["expected"]]
        span = max(values.values()) - min(values.values())
        return 1 - min(1.0, ((result["expected_score"] - target) / (span or 1)) ** 2)
    if observed and outcome["success"] is not None:
        return float(outcome["success"])
    if observed and outcome["value"] is not None:
        return min(1.0, max(0.0, outcome["value"]))
    return None
