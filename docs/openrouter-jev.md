# OpenRouter-backed Jev

`openrouter-jev` is an opt-in, standard-library remote adapter for OpenRouter's **Decisions
API**, separate from the existing `jev` bridge and `structured-llm` chat adapter. No new
runtime dependency, credential discovery, metadata call, or fallback is involved.

## Evidence and verification boundary

Public official documents inspected on 2026-09-26; TypeSafe Score and the OpenRouter
tutorial rechecked on 2026-09-27 for continuous-score semantics:

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

These documents are evidence, not instructions to execute their examples. The automated
suite uses synthetic credentials and injected byte-stream transports; socket creation is
forbidden. A separately authorized synthetic live check on 2026-09-27 made six focused
score calls and one 19-case starter run against `typesafe/jev-1.13-20260917`. Only sanitized
numeric diagnostics were retained outside the repository; no live payloads are committed.
See the limited [live evidence](#live-score-check-2026-09-27) below.

The public tutorial documents a release alias resolving to a dated snapshot. It does not
establish a contractual immutability guarantee. The dated snapshot was reachable during
the limited live check. There is no verified server revision attestation field beyond
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
- Live reachability, latency, usage and semantic observations are limited to the synthetic
  sample below. Snapshot immutability, precision guarantees, routing enforcement beyond
  returned identity, and production quality remain unverified. `providers --full` exposes
  these limitations even before registering a provider. The diagnostic
  `live_verification: unverified` denotes the absence of independent server attestation.

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

Those commands may incur charges with an enabled provider. The live check used single mode;
it did not qualify native batching or higher concurrency. An offline framework benchmark uses `--provider mock`.
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
label order. Nonuniform spacing is allowed. Jev's returned `score` is a **continuous expected
index**, not an integer class or a pack value. It must be a finite number in `[0,N-1]`.
The returned distribution is canonical provider evidence. Universal core logic computes
Reflex `expected_score = sum(probability(label) * pack_value(label))` from that distribution
(after calibration, if configured). If supplied, `legend` must exactly match the submitted description
order. No sorting or implicit ordinal reinterpretation occurs.

The parser requires finite numeric probabilities in `[0,1]`, exact label cardinality and
sum-to-one within `1e-6`. Booleans are not numbers. It never rounds, normalizes, fills missing
mass, or constructs a distribution from a winning label and confidence. Although the public
schema makes Choice/Score probabilities optional, incomplete responses cannot satisfy
Reflex's provider contract and fail without caching. Selected ties may name any maximum;
Reflex deterministically selects the first maximum in pack label order. The native selected
label is retained in diagnostics. Score selection uses that same provider-neutral rule,
never rounding or casting the native score to a class.

**Native score comparison is diagnostic, not distribution validation.** TypeSafe describes
the score as a probability-weighted mean of level numbers. The OpenRouter tutorial's
captured response reports `1.99` alongside probabilities implying `2.0`; the public evidence
does not establish exact equality at the returned precision. The adapter therefore preserves
`native_score`, recomputes `native_expected_index = sum(p_i * i)`, and records the absolute
`native_score_disagreement` in provider diagnostics. Its warning is `none` for disagreement
at most `1e-6`, `small_disagreement` up to `0.02` index units, otherwise `large_disagreement`.
The `0.02` threshold is a local diagnostic severity choice, not a provider precision promise.
Even large disagreement cannot invalidate otherwise valid evidence. For example,
`{"0":0.01,"1":0,"2":0.99}` implies native index `1.98`; with pack values `0,0.5,1`,
the Reflex semantic score is `0.99`, regardless of a valid native score of `1.98` or `1.99`.
Neither distribution probabilities nor the universal sum tolerance change.

Adapter version remains `1`: request inputs, canonical distributions of previously accepted
responses, selection, calibration and identity semantics are unchanged. This fixes response
acceptance and adds diagnostics, not an inference input or model change. Existing cached
accepted results remain valid (and may lack the new diagnostic fields); rejected responses
were never cached. Benchmark implementation hashes still distinguish before/after runs.

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

Typed response validation keeps the `provider_response` code and includes a safe local
reason in its message, for example `invalid Decisions response (probability_sum);
Probabilities must sum to one within 1e-6`. Reasons distinguish schema, probability keys,
values or sums, native choice/confidence, score/legend, request ID shape, and usage failures.
They come from a fixed allowlist, never provider-controlled text, and survive in saved failed
job cells. This adds no rejected-response capture and does not change validation, caching,
or retry behavior. A `provider_response` error alone does not establish missing probabilities;
inspect its reason to identify the failed check.

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

## Opt-in score failure diagnostics

Use `benchmark run ... --score-diagnostics` (or explicitly opt in again on an unfinished
`benchmark resume`) to capture score response validation evidence. Ordinary inference and
benchmarks without this flag emit no rejected response diagnostics or console warnings.
This option is restricted to `openrouter-jev`; it does not change provider identity,
calibration, benchmark quality denominators or the sealed result. `benchmark show --run <id>
--full --json` includes the separate records as `score_diagnostics` when present.

Before typed validation, the adapter projects the decoded JSON onto a fixed allowlist:
known answer kind; numeric native score/confidence; exact numeric probabilities under at
most ten locally generated index keys; probability, legend and label counts; recomputed
expected index when the complete index map is numeric; matching configured model/provider;
numeric usage/cost. Unknown model/provider strings become `mismatch`, never copied text.
Non-numeric/nonfinite fields become null or are omitted; `probabilities_complete` exposes
whether every returned entry was retained. Malformed maps with unknown or oversized keys
retain cardinality, not provider-controlled keys. Legend text, state, instructions, IDs,
headers, arbitrary text and credentials are never persisted. Strict JSON/size/transport
failures produce only a local category, since there is no safe decoded answer to inspect.

After validation, the record adds a stable Reflex-authored category/message and, for valid
answers, native disagreement diagnostics. Records live in the state-root SQLite bucket
`openrouter_score_diagnostics:<run-id>` (0700 directory, 0600 database), outside the source
tree. A session is capped at 256 records of at most 4096 bytes plus one limit marker;
transactions enforce the cap across concurrent writers. These are supplemental observations,
not accepted results, calibration evidence, cost totals, or learnable data. Full exports
remain an explicit operation. Storage failures surface rather than claiming capture succeeded.

For controlled Python diagnostic calls, construct `ScoreDiagnostics(store)` from
`reflex_axi.openrouter_diagnostics` and pass it as `score_diagnostics=` to
`OpenRouterJevProvider`. Only score calls are captured. The optional recorder is runtime
instrumentation, never a provider setting, and must use a private state root.

## Live score check, 2026-09-27

Two passes over the synthetic sufficient/partial/empty-result shapes accepted 6/6 calls
and selected every authored expected label. Native indexes included fractional values;
all six matched their returned distribution-derived indexes. The zero controls both
returned zero. No raw-response comparison or extra sampling was needed.

A new immutable run used the original provider binding and corpus with `starter`, `single`,
one repeat and concurrency one. Original sealed records remained unchanged; both runs'
definition/results hashes and independently recomputed summaries verified.

| Measurement | Original live run | New live run |
| --- | --- | --- |
| Accepted and correct / attempted | 17/19 | 19/19 |
| Accepted-label accuracy | 17/17 | 19/19 |
| Score cases accepted | 1/3 | 3/3 |
| Brier / ECE / log loss | 0.014494 / 0.058824 / 0.063099 | 0.019484 / 0.071053 / 0.077635 |
| Semantic score MAE | 0, only one accepted score | 0.033333, all three scores |
| Coverage at 0.8, accepted denominator | 17/17 | 17/19 |
| Coverage at 0.95, accepted denominator | 10/17 | 9/19 |
| Mean / p50 / p95 call latency, ms | 277.48 / 267.49 / 355.99 | 289.77 / 284.98 / 443.60 |
| Input / output tokens | 6202 / 583, incomplete | 6921 / 621, complete |
| Reported cost, USD | 0.000260484 subtotal; total unknown | 0.000290682 complete |

There was no calibration configured. These are descriptive calibration metrics, not a
calibration guarantee. Coverage at 0.95 over **all attempts** was 10/19 before and 9/19 after.
No selected errors were observed, but nineteen authored examples do not estimate production
risk. Quality denominators differ because rejected calls are excluded. Pilot reported cost
was USD 0.000089964; all 25 authorized new calls together reported USD 0.000380646.

The before/after observation does **not** establish causality for the historical failures:
original rejected responses were not retained, and every newly observed score also satisfies
the former equality check. The regression fixtures and public tutorial establish why strict
displayed equality is an invalid acceptance requirement; the live sample establishes current
fractional-score operation. Probabilities and latency varied between calls. No further paid
reruns were made, and no production, immutable-revision or service-reliability claim follows.
