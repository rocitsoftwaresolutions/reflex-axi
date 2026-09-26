import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import ValidationError

from .errors import ReflexError
from .identity import canonical, digest
from .models import Binding, DecisionResult, ExecutionPolicy, Pack, Prediction
from .providers import Provider, Request, provider_from_spec, validate_predictions
from .store import Store


class Engine:
    def __init__(self, store: Store, provider: Provider | None = None) -> None:
        self.store = store
        self.provider = provider

    def _check(
        self, pack: Pack, state: dict[str, Any], binding: Binding, policy: ExecutionPolicy
    ) -> Request:
        binding.check(pack, policy.workload, policy.scope)
        spec = binding.provider
        cap = spec.capabilities
        if (
            pack.judgment.primitive not in cap.primitives
            or (policy.local_only and not cap.local)
            or cap.max_batch < policy.min_batch
        ):
            raise ReflexError(
                "capability",
                "configured provider cannot satisfy primitive, privacy or batch requirements",
            )
        if ["free", "low", "medium", "high"].index(cap.cost_class) > [
            "free",
            "low",
            "medium",
            "high",
        ].index(policy.max_cost_class) or ["low", "medium", "high"].index(cap.latency_class) > [
            "low",
            "medium",
            "high",
        ].index(policy.max_latency_class):
            raise ReflexError(
                "capability", "configured provider exceeds execution cost or latency class"
            )
        normalized = pack.state.build(state)
        if len(canonical(normalized).encode()) > cap.max_state_bytes:
            raise ReflexError("state_too_large", "normalized state exceeds provider limit")
        identity = {
            "schema": 1,
            "state": normalized,
            "pack": pack.inference_definition(),
            "provider": spec.identity(),
            "calibration": binding.calibration.model_dump() if binding.calibration else None,
            "workload": policy.workload,
            "scope": policy.scope,
        }
        return Request(digest(identity), pack, normalized)

    def evaluate(
        self,
        pack: Pack,
        state: dict[str, Any],
        binding: Binding,
        *,
        execution: ExecutionPolicy | None = None,
        record: bool = False,
    ) -> DecisionResult:
        value = self.evaluate_many([(pack, state)], binding, execution=execution, record=record)[0]
        if isinstance(value, ReflexError):
            raise value
        return value

    def decide(
        self,
        pack: Pack,
        state: dict[str, Any],
        binding: Binding,
        *,
        execution: ExecutionPolicy | None = None,
        record: bool = True,
    ) -> DecisionResult:
        result = self.evaluate(pack, state, binding, execution=execution, record=False)
        result = apply_policy(pack, result)
        if record:
            self._record(pack, state, binding, result)
        return result

    def _record(
        self, pack: Pack, state: dict[str, Any], binding: Binding, result: DecisionResult
    ) -> None:
        self.store.put(
            "decisions",
            result.id,
            {
                "pack": pack.model_dump(by_alias=True),
                "state": state,
                "binding": binding.model_dump(),
                "result": result.model_dump(),
                "time": time.time(),
                "project": self.store.project_key(),
            },
            immutable=True,
        )

    def evaluate_many(
        self,
        items: list[tuple[Pack, dict[str, Any]]],
        binding: Binding,
        *,
        execution: ExecutionPolicy | None = None,
        record: bool = False,
        decide: bool = False,
        concurrency: int | None = None,
    ) -> list[DecisionResult | ReflexError]:
        policy = execution or ExecutionPolicy()
        provider = self.provider or provider_from_spec(binding.provider)
        if provider.spec.identity() != binding.provider.identity():
            raise ReflexError("provider_mismatch", "injected provider differs from pinned binding")
        cap = binding.provider.capabilities
        workers = min(concurrency or cap.concurrency, cap.concurrency)
        if workers < 1:
            raise ReflexError("usage", "concurrency must be positive")
        requests: list[Request | ReflexError] = []
        unique: dict[str, Request] = {}
        for pack, state in items:
            try:
                req = self._check(pack, state, binding, policy)
                requests.append(req)
                unique[req.id] = req
            except ReflexError as error:
                requests.append(error)
        results: dict[str, tuple[dict[str, Any], bool] | ReflexError] = {}
        pending: list[Request] = []
        for req in unique.values():
            cached = self.store.get("cache", req.id) if policy.cache else None
            if cached:
                results[req.id] = (cached, True)
            else:
                pending.append(req)
        # Native batches may span packs only when multi-question is advertised.
        groups: dict[str, list[Request]] = {}
        for req in pending:
            key = "all" if cap.multi_question else digest(req.pack.inference_definition())
            groups.setdefault(key, []).append(req)
        size = cap.max_batch if cap.native_batch or cap.multi_question else 1
        chunks = [
            group[i : i + size] for group in groups.values() for i in range(0, len(group), size)
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._execute, chunk, provider, binding, policy) for chunk in chunks
            ]
            for future in as_completed(futures):
                results.update(future.result())
        output: list[DecisionResult | ReflexError] = []
        for (pack, state), item_request in zip(items, requests, strict=True):
            if isinstance(item_request, ReflexError):
                output.append(item_request)
                continue
            found = results[item_request.id]
            if isinstance(found, ReflexError):
                output.append(found)
                continue
            payload, hit = found
            result = DecisionResult.model_validate(
                {
                    **payload,
                    "id": uuid.uuid4().hex,
                    "execution": {
                        "cache_hit": hit,
                        "source_latency_ms": payload["latency_ms"],
                        "source_cost": payload["cost"],
                        "workload": policy.workload,
                        "scope": policy.scope,
                    },
                    "latency_ms": 0.0 if hit else payload["latency_ms"],
                    "cost": 0.0 if hit else payload["cost"],
                    "usage": {} if hit else payload["usage"],
                }
            )
            if decide:
                result = apply_policy(pack, result)
            if record:
                self._record(pack, state, binding, result)
            output.append(result)
        return output

    def _execute(
        self, requests: list[Request], provider: Provider, binding: Binding, policy: ExecutionPolicy
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        owned: dict[str, str] = {}
        busy: list[Request] = []
        claimed: list[Request] = []
        slot: tuple[str, str] | None = None
        try:
            for req in requests:
                if policy.cache:
                    owner = self.store.claim("inference:" + req.id)
                    if owner is None:
                        busy.append(req)
                        continue
                    owned[req.id] = owner
                    cached = self.store.get("cache", req.id)
                    if cached:
                        results[req.id] = (cached, True)
                        continue
                claimed.append(req)
            if claimed:
                cap = provider.spec.capabilities
                deadline = time.monotonic() + 330
                while slot is None:
                    for index in range(cap.concurrency):
                        key = f"provider:{provider.spec.identity()}:{index}"
                        owner = self.store.claim(key)
                        if owner:
                            slot = key, owner
                            break
                    if slot is None:
                        if time.monotonic() > deadline:
                            raise ReflexError(
                                "provider_busy",
                                "provider concurrency slots are busy; resume the job",
                            )
                        time.sleep(0.02)
                delay = self.store.reserve_rate(provider.spec.identity(), cap.requests_per_second)
                if delay > 30:
                    raise ReflexError(
                        "rate_limited",
                        "provider rate reservation exceeds 30 seconds; retry or lower concurrency",
                    )
                time.sleep(delay)
                start = time.monotonic()
                predictions = provider.infer(claimed)
                validate_predictions(claimed, predictions)
                elapsed = (time.monotonic() - start) * 1000
                for req in claimed:
                    payload = make_result(req, predictions[req.id], binding, elapsed).model_dump()
                    if policy.cache:
                        with self.store.transaction() as db:
                            if not self.store.owns("inference:" + req.id, owned[req.id], db):
                                raise ReflexError(
                                    "lease_lost",
                                    "inference lease expired; retry without trusting stale output",
                                )
                            self.store.put("cache", req.id, payload, db)
                    results[req.id] = (payload, False)
        except (ReflexError, ValidationError) as error:
            safe = (
                error
                if isinstance(error, ReflexError)
                else ReflexError("provider_response", "provider response violates its schema")
            )
            for req in claimed:
                results.setdefault(req.id, safe)
        finally:
            for key, owner in owned.items():
                self.store.release("inference:" + key, owner)
            if slot:
                self.store.release(*slot)
        # Never wait while holding inference locks: overlapping bundles cannot deadlock.
        for req in busy:
            deadline = time.monotonic() + 370
            while True:
                cached = self.store.get("cache", req.id)
                if cached:
                    results[req.id] = (cached, True)
                    break
                with self.store.connect() as db:
                    lease = db.execute(
                        "SELECT expires FROM leases WHERE key=?", ("inference:" + req.id,)
                    ).fetchone()
                if not lease or lease[0] <= time.time():
                    results.update(self._execute([req], provider, binding, policy))
                    break
                if time.monotonic() > deadline:
                    results[req.id] = ReflexError(
                        "inference_busy", "inference is still running; resume later"
                    )
                    break
                time.sleep(0.02)
        return results

    def cascade(
        self,
        pack: Pack,
        state: dict[str, Any],
        bindings: list[Binding],
        *,
        threshold: float = 0.8,
        execution: ExecutionPolicy | None = None,
        fallback_errors: bool = False,
    ) -> DecisionResult:
        if not bindings or not 0 <= threshold <= 1:
            raise ReflexError("usage", "cascade needs bindings and a threshold in [0,1]")
        attempts: list[dict[str, Any]] = []
        final: DecisionResult | None = None
        for binding in bindings:
            try:
                result = self.evaluate(pack, state, binding, execution=execution)
                attempts.append(
                    {
                        "provider": result.provider,
                        "selected": result.selected,
                        "confidence": result.confidence,
                        "cost": result.cost,
                    }
                )
                final = result
                if result.confidence >= threshold:
                    break
            except ReflexError as error:
                if not fallback_errors or error.code not in {
                    "provider_http",
                    "provider_transport",
                    "provider_unavailable",
                }:
                    raise
                attempts.append({"provider": binding.provider.id, "error": error.code})
        if final is None:
            raise ReflexError(
                "cascade_exhausted", "all explicitly configured cascade providers failed"
            )
        return final.model_copy(
            update={
                "execution": {
                    **final.execution,
                    "attempts": attempts,
                    "system2_required": final.confidence < threshold,
                }
            }
        )


def make_result(
    request: Request, prediction: Prediction, binding: Binding, elapsed: float
) -> DecisionResult:
    raw = prediction.distribution
    calibration = binding.calibration
    distribution = dict(raw)
    if calibration:
        # Log-space temperature scaling avoids underflow for small temperatures.
        logs = {
            k: math.log(p) / calibration.temperature if p else -math.inf for k, p in raw.items()
        }
        top = max(logs.values())
        weights = {k: math.exp(v - top) for k, v in logs.items()}
        distribution = {k: v / sum(weights.values()) for k, v in weights.items()}
    selected = max(request.pack.judgment.labels, key=lambda x: distribution[x])
    values = request.pack.judgment.values
    if request.pack.judgment.primitive == "binary":
        values = {"false": 0.0, "true": 1.0}
    entropy = -sum(p * math.log(p) for p in distribution.values() if p) / math.log(
        len(distribution)
    )
    spec = binding.provider
    return DecisionResult(
        id="inference",
        pack=request.pack.id,
        version=request.pack.version,
        primitive=request.pack.judgment.primitive,
        raw_distribution=raw,
        distribution=distribution,
        selected=selected,
        expected_score=sum(distribution[k] * v for k, v in values.items()) if values else None,
        confidence=distribution[selected],
        uncertainty=entropy,
        provider={
            "id": spec.id,
            "model": spec.model,
            "version": spec.version,
            "adapter_version": spec.adapter_version,
            "hash": spec.identity(),
        },
        calibration={"id": calibration.id, "version": calibration.version} if calibration else None,
        state_hash=digest(request.state),
        inference_hash=request.id,
        latency_ms=elapsed,
        cost=prediction.cost,
        usage=prediction.usage,
        execution={},
    )


def apply_policy(pack: Pack, result: DecisionResult) -> DecisionResult:
    policy = pack.policy
    if policy.kind == "none":
        return result
    confident = result.confidence >= policy.min_confidence
    action = policy.actions.get(result.selected, result.selected) if confident else policy.uncertain
    return result.model_copy(
        update={
            "policy_result": {
                "version": policy.version,
                "recommendation": action,
                "uncertain": not confident,
                "executed": False,
                "authorization": "external",
            }
        }
    )
