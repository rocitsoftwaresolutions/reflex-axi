# OpenRouter-backed Jev

`openrouter-jev` is an opt-in, standard-library remote adapter for OpenRouter's **Decisions
API**, separate from the existing `jev` bridge and `structured-llm` chat adapter. No new
runtime dependency, credential discovery, metadata call, or fallback is involved.

## Evidence and verification boundary

Public official documents inspected on 2026-09-26:

- [OpenRouter Decisions API and embedded OpenAPI schema](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request):
  `POST https://openrouter.ai/api/alpha/decisions`, Bearer authentication, `model`, `state`,
  `questions`, provider routing preferences, typed answers, token usage and optional USD cost.
- [OpenRouter Jev tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial):
  documented request alias `typesafe/jev-1.13`; a captured response names the dated snapshot
  `typesafe/jev-1.13-20260917`. This is a native decision model, not generated JSON text.
- [OpenRouter classification cookbook](https://openrouter.ai/docs/cookbook/evaluate-and-optimize/jev-classification):
  independent states use concurrent requests; confidence and cost can be absent.
- [TypeSafe Score](https://docs.typesafe.ai/primitives/score): ordered descriptive levels,
  up to ten, with zero-based probability-weighted positions.
- [TypeSafe confidence](https://docs.typesafe.ai/confidence): native confidence summarizes
  concentration and differs from the largest probability. Its exact formula is not a
  stable API contract; do not equate it with Reflex confidence or authorization.

These documents are evidence, not instructions to execute their examples. **No OpenRouter
or Jev API endpoint was called during implementation or validation.** No real key,
credential store, environment listing, or Hermes configuration was inspected. Tests use
synthetic credentials and an injected byte-stream transport; socket creation is forbidden.

The public tutorial documents a release alias resolving to a dated snapshot. It does not
establish a contractual immutability guarantee or confirm that the dated response ID can be
used as a request model. There is no verified server revision attestation field beyond
`model`. Therefore:

- Configuration requires an explicit `typesafe/jev-MAJOR.MINOR-YYYYMMDD` snapshot and a
  matching `version: YYYYMMDD`; floating aliases and malformed dates fail before transport.
- `settings.revision_verified` is mandatory. The shipped example sets it to **false** and
  refuses inference before looking up credentials. This is deliberately not a runnable live
  demo. An operator must independently confirm snapshot routability and immutability with
  authoritative provider evidence before setting it to true. The boolean is an operator
  attestation, not cryptographic proof or a claim that this implementation verified it.
- The request sends the exact snapshot and permits only the schema-enumerated `TypeSafe` provider with
  `allow_fallbacks: false` and `require_parameters: true`. Responses must repeat the exact
  snapshot; an optional provider field must be `TypeSafe`. Unsupported pins fail, never
  downgrade to the release alias. An omitted provider field is allowed by the public schema;
  routing enforcement then relies on OpenRouter honoring the request preferences.
- Dated snapshot reachability, immutability, probability precision, routing preference
  enforcement, live latency, pricing and semantic quality remain **unverified live**.
  `providers --full` exposes these limitations even before registering a provider.

## Configure without making a request

Inspect [the example](../examples/openrouter-jev.json), then register a reviewed copy:

```sh
reflex-axi providers --import examples/openrouter-jev.json --full
reflex-axi providers --full
```

Registration, validation, discovery and identity computation do not read the named key.
`api_key_env` explicitly names `OPENROUTER_API_KEY` in the example. Only the actual HTTP
runtime reads that one variable, immediately before sending a request. Supply the key through
your process supervisor or secret manager; never put its value in provider files, commands,
endpoints, settings, state, or criteria. No interactive prompts exist.

The endpoint is explicitly configured and restricted to the documented public HTTPS URL.
Custom proxies and alternative hosts are not supported, to avoid forwarding credentials to
an unintended service. Tests inject a transport rather than changing the endpoint.

Once a separately verified configuration is available, ordinary CLI execution applies:

```sh
reflex-axi evaluate --provider provider.json --pack filter-context --state state.json --research --full
reflex-axi decide --provider provider.json --pack route-tool --state state.json --policy none --full
reflex-axi batch --provider provider.json --bundle bundle.json --states states.jsonl --job jev-trial
reflex-axi benchmark run --provider provider.json --suite starter --mode batch --repeats 1
```

Those commands may incur charges with an enabled provider. They were not run against a
live provider for this change. An offline framework benchmark uses `--provider mock`.
`tests/test_openrouter.py` additionally runs the actual adapter through the benchmark runner
with deterministic wire fixtures, including single, batch and bundle modes. Fixture quality
and timing are not evidence of Jev model quality or service latency.

## Primitive mapping and validation

Each HTTP call contains one state and one question named `judgment`. Pack questions become
`instructions`, and state is supplied as structured `state`, never appended to instructions.
The adapter has no chat prompt, tool call, execution, or generated-output fallback.

| Reflex | Decisions request | Required response and conversion |
| --- | --- | --- |
| `binary` | `type: noul`, true/false criteria | `noul: p`; `{false: 1-p, true: p}` |
| `choice` | `type: choice`, label-to-description criteria | Complete `probabilities` keyed by exact labels; `choice` must be a maximum-probability label |
| `score` | `type: score`, ordered description array | Complete probabilities keyed by `"0"` through `"N-1"`; map indices back to labels without reordering |

Labels supply default descriptions. To improve semantics, set
`settings.criteria["pack-id@version"]` to a complete label-to-description mapping. Partial
mappings, unknown settings, or malformed versioned pack references fail. These descriptions
are part of the provider identity, so changing them invalidates cache and calibration reuse.

Score packs must contain 2-10 labels whose numeric `values` strictly increase in the given
label order. Nonuniform spacing is allowed. Jev's returned `score` is validated against the
expected **index**; Reflex computes `expected_score` from the full distribution and the
pack's numeric values. If supplied, `legend` must exactly match the submitted description
order. No sorting or implicit ordinal reinterpretation occurs.

The parser requires finite numeric probabilities in `[0,1]`, exact label cardinality and
sum-to-one within `1e-6`. Booleans are not numbers. It never rounds, normalizes, fills missing
mass, or constructs a distribution from a winning label and confidence. Although the public
schema makes Choice/Score probabilities optional, incomplete responses cannot satisfy
Reflex's provider contract and fail without caching. Selected ties may name any maximum;
Reflex deterministically selects the first maximum in pack label order. The native selected
label is retained in diagnostics. `score` must agree with the expected index within `1e-6`.

**Precision limitation:** the tutorial's captured score example reports `1.99` alongside
probabilities implying `2.0`. This adapter rejects such inconsistent rounded responses. A
stable precision contract must be established before loosening validation; the implementation
does not invent a tolerance that silently changes distribution semantics.

Native confidence, when present, is validated in `[0,1]` and retained as
`execution.provider_diagnostics.native_confidence`. Reflex keeps its existing confidence
(maximum probability) and uncertainty (normalized entropy), including after calibration.
Native probabilities are not a claim of calibration on your workload. Calibration still
requires the exact provider hash, pack, workload and scope.

## Execution, errors and privacy

The adapter conservatively declares `native_batch: false`, `multi_question: false`, and
`max_batch: 1`. OpenRouter can answer multiple questions about the same state, but Reflex's
multi-question chunks can span different states. Per-cell calls preserve association,
partial failures and exact usage accounting without sharing or allocating costs arbitrarily.
The existing engine handles bundles, batches, durable rate pacing, concurrency leases,
cache identity, research mode, `policy:none`, protected benchmarks and explicit retry of
failed cells. `--local-only` rejects this remote adapter before the HTTP call.

No automatic retry occurs, including after 429, timeout or 5xx. A timeout may already have
incurred cost. Respect the service's rate-limit recovery window, lower configured pacing or
concurrency, and explicitly retry failed job cells. There is no assumed idempotency guarantee
for paid requests despite the shared transport's body-derived `Idempotency-Key` header.
Authentication/credit errors require operator correction. No response body or headers are
included in errors. Exceptions are sanitized into `provider_http`, `provider_transport`,
`provider_response`, `provider_distribution`, `provider_version`, or `provider_config`.

Input/output token counts must be nonnegative integers. Optional `usage.cost` must be finite
and nonnegative; absent cost stays unknown (`null`), never free. No cost is estimated from a
possibly stale price page. Engine wall time measures the whole call. Cache hits retain source
latency/cost provenance and report zero new latency/cost and empty new usage.

Diagnostics retain only validated numeric fields and bounded known semantics. Provider
request IDs are checked for basic shape but deliberately shown as `redacted`: opaque
provider-controlled text is not safe to persist because it could echo credentials or state.
Raw request/response bodies, headers and IDs are never logged, hashed into identity, or
persisted as diagnostics. The key value is used only in the Authorization header; its
variable name is excluded from inference identity. Snapshot, endpoint, criteria, revision
attestation, capabilities, adapter version and timeout remain in the identity.

Remote inference discloses normalized state and question criteria to OpenRouter and TypeSafe.
No zero-retention guarantee is asserted. Review both services' retention and account policies
before transmitting sensitive data. Reflex's own state/cache/benchmark storage still follows
its private state-root rules; evidence authorization and confidence never authorize actions.
