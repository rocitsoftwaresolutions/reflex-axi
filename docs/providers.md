# Provider contracts

A provider is a replaceable inference adapter. It receives normalized state plus a judgment
definition and returns a complete probability distribution. It never receives a policy, never
returns an action, and its response is always treated as untrusted data.

```python
class Provider(Protocol):
    spec: ProviderSpec

    def infer(self, requests: list[Request]) -> dict[str, Prediction]: ...
```

`Request` is `(id, pack, state)`; `Prediction` is `(distribution, usage, cost)`. Every
response is validated before use: the returned IDs must be exactly the requested IDs (no
omissions, duplicates or invented IDs -> `provider_association`), and each distribution must
cover exactly the pack's labels with finite probabilities in `[0,1]` summing to one within
1e-6 (`provider_distribution`). A malformed response is never cached.

## `ProviderSpec`

```json
{
  "adapter": "jev",
  "id": "jev-prod",
  "model": "jev-small",
  "version": "2026-04-01",
  "adapter_version": "1",
  "endpoint": "https://jev.internal/v1/judge",
  "api_key_env": "JEV_API_KEY",
  "timeout": 30.0,
  "capabilities": { "...": "see below" },
  "settings": { "questions": { "filter-context@1": "noul-7731" } }
}
```

`model` and `version` must be **immutable identifiers**. A floating alias silently changes
the model under a pinned calibration, which is exactly what the identity design exists to
prevent. Secrets are never stored: `api_key_env` names an environment variable, and the
variable's *name* is excluded from the provider identity hash because it affects
authorization, not inference.

Register with `reflex-axi providers --import provider.json`; registration is immutable per
`id@version`.

## Capabilities

Providers publish what they can do, and the engine enforces it before any call:

| Field | Meaning |
| --- | --- |
| `primitives` | which of `binary` / `choice` / `score` are supported |
| `native_batch` | many requests for one pack in one call |
| `multi_question` | many *different* packs in one call (bundle-native) |
| `local` | runs locally; required by `--local-only` |
| `max_batch` | maximum requests per call (1-10000) |
| `concurrency` | maximum simultaneous calls (enforced by cross-process lease slots) |
| `requests_per_second` | pacing budget, shared durably across processes |
| `max_state_bytes` | rejects oversized normalized state before the call |
| `latency_class` | `low` / `medium` / `high` |
| `cost_class` | `free` / `low` / `medium` / `high` |
| `seeded`, `deterministic` | reproducibility claims |

An `ExecutionPolicy` may select **explicitly** on privacy (`local_only`), cost
(`max_cost_class`), latency (`max_latency_class`), batch size (`min_batch`) and caching. A
provider that cannot satisfy the request raises `capability`. Selection never silently
substitutes a different provider or model: there is no implicit fallback anywhere in the
engine, and an HTTP or transport failure surfaces as an error rather than a downgrade.

## Provider upgrades

A provider or model upgrade is an independent lifecycle event, not a config tweak:

1. It changes the provider identity hash, so it changes every inference key. No cached
   result, and no calibration, carries over.
2. A `Calibration` pinned to the old provider hash raises `calibration_mismatch` rather than
   being reused.
3. Replay the pack against authorized historical outcomes and recalibrate before promoting.

Upgrades therefore go through the ordinary candidate path in [lifecycle](lifecycle.md):
propose a deployment with the new binding, replay, shadow, then promote.

## Calibration

```json
{
  "id": "jev-filter-context",
  "version": "3",
  "provider_hash": "<ProviderSpec.identity()>",
  "pack_id": "filter-context",
  "pack_version": "1",
  "workload": "support-tickets",
  "scope": "default",
  "temperature": 1.4
}
```

Calibration is scoped by provider/model × pack version × workload × evaluation scope ×
calibration version. Any mismatch is an error, never a silent inheritance. Scaling is applied
in log space so small temperatures cannot underflow, and the result keeps **both**
`raw_distribution` and the calibrated `distribution`.

## Bundled adapters

### `mock`

Explicit test backend. `settings` selects a label directly (`label`), or maps a state field
to a label (`field` + `mapping`), with a configurable `confidence`; `fail: true` raises
`provider_unavailable` for failure-path tests. It makes **no semantic claim**, and nothing
ever falls back to it implicitly - if you have not configured a provider, built-in packs use
it because that was the only explicit default, and the CLI says so on the home view.

### `jev`

Versioned bridge protocol, `reflex-jev/1`. Jev is one backend among several; nothing
Jev-specific reaches the core. Request:

```json
{
  "protocol": "reflex-jev/1",
  "model": "jev-small",
  "version": "2026-04-01",
  "settings": { "questions": { "filter-context@1": "noul-7731" } },
  "requests": [
    { "id": "<inference hash>", "question": "...", "primitive": "binary",
      "labels": ["false", "true"], "state": { "...": "normalized" } }
  ]
}
```

Response:

```json
{ "results": [ { "id": "<same id>", "distribution": { "false": 0.1, "true": 0.9 },
                 "usage": { "tokens": 12 }, "cost": 0.0001 } ] }
```

The envelope must contain exactly `results`; unknown or duplicate IDs raise
`provider_association`. Noul identifiers live only in `settings.questions`, keyed by
`pack@version`, and are opaque to the engine. The adapter supports `native_batch` and
`multi_question`, so a bundle or a batch chunk becomes one call.

### `structured-llm`

OpenAI-compatible chat completions with a strict JSON-schema response format, one request per
call (declaring `native_batch` or `multi_question` on this adapter is a validation error).
The system message states the contract explicitly: return bounded probabilities summing to
one, treat state as untrusted data rather than instructions, and perform no actions,
generation or exact arithmetic. Only `temperature`, `seed`, `max_tokens` and `top_p` are
accepted in `settings`. If the response's `model` differs from the pinned model, the call
fails with `provider_version` rather than silently using a different model.

### `local-student`

The deployment target for [distillation](lifecycle.md#distillation): a deterministic,
local-only lookup provider with a prior fallback. A real trainer implements the same
`Trainer` protocol and returns whatever `ProviderSpec` its runtime needs; the shipped fixture
trainer keeps the lifecycle testable without bundling an ML runtime.

## Network safety

All remote calls go through one hardened `post`:

- HTTPS only; plain HTTP is allowed solely for loopback fixtures (`127.0.0.1`, `localhost`,
  `::1`).
- Endpoints may not embed credentials, query strings or fragments.
- Redirects are never followed.
- Responses are capped at 8 MB and parsed with strict JSON (duplicate keys and non-finite
  numbers rejected).
- An `Idempotency-Key` derived from the request body is always sent.
- Errors are sanitized into `provider_http` / `provider_transport` codes: no URLs, headers or
  credentials reach the output.

## Opt-in disagreement

Ordinary operation runs exactly one configured provider or one explicitly configured cascade.
Running several providers is always deliberate:

- `reflex-axi compare --pack p --state s.json --provider a.json --against b.json` reports
  agreement and total-variation distance between two pinned bindings.
- `Engine.cascade(pack, state, [binding_a, binding_b], threshold=...)` walks an explicitly
  ordered list, stopping at the first result above the threshold, and records the attempts
  plus a `system2_required` flag. `fallback_errors=True` continues past availability errors
  only.
- Shadow pairs during a candidate evaluation.

None of these run implicitly, and none of them cost anything during ordinary inference.

## Writing an adapter

1. Implement `infer` and expose an accurate `spec`.
2. Declare capabilities honestly - the engine trusts them for batching, concurrency, pacing
   and privacy enforcement.
3. Never invent, merge or reorder result IDs; return exactly what was requested.
4. Raise `ReflexError` with a specific code; never fall back to another backend.
5. Keep backend-specific concepts inside `settings`.
6. Register the class in `provider_from_spec` and add the adapter name to
   `ProviderSpec.adapter`.
7. Add a fixture-server test in the style of `tests/test_providers.py`, which drives real
   loopback HTTP rather than mocking the transport.
