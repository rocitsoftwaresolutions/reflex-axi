# Architecture

Reflex is a **judgment engine**, not an action engine. It converts a normalized state into a
bounded probability distribution over a small label set, optionally attaches a policy
*recommendation*, and stops there. Nothing in Reflex executes an action, and no confidence
value authorizes a destructive, sensitive or external side effect.

## The hierarchy

1. **Deterministic code** for anything exact: arithmetic, parsing, transformations, lookups.
   Reflex must never be asked to compute these.
2. **Reflex (System 1)** for bounded fuzzy judgment: is this context relevant, which skill
   fits, is this result sufficient, how risky is this.
3. **System 2** only for genuine reasoning or generation.

`route-model` exists to make step 3 an explicit, observable routing decision rather than a
default.

## Layers

```
CLI (cli.py)  ──►  Engine (engine.py)  ──►  Provider adapter (providers.py)
     │                   │
     │                   ├─► Store (store.py): SQLite records, leases, events, rate slots
     │                   └─► identity.py: canonical JSON + SHA-256 inference identity
     │
     ├─► BatchRunner (jobs.py): four execution shapes, per-cell checkpoints, resume
     ├─► Evidence (evidence.py) + Outcome adapters (outcomes.py)
     ├─► Lifecycle (lifecycle.py): candidate → replay → shadow → promotion → rollback
     ├─► Distillation (distill.py): teacher dataset → local student → same gates
     └─► TOON encoder (toon.py): compact default output
```

`models.py` holds every contract. Each model is strict, frozen, and forbids unknown keys, so
a typo in a pack, provider or outcome is a validation error rather than a silent default.

## Core primitives

Three provider-neutral primitives, and only three:

| Primitive | Labels | Result |
| --- | --- | --- |
| `binary` | exactly `["false", "true"]` | distribution plus expected score in `[0,1]` |
| `choice` | 2-100 distinct labels | distribution over the labels |
| `score` | 2-100 labels, each with a numeric `values` entry | distribution plus expected score |

Backend-specific concepts stay in the adapter. Jev's Noul identifiers, for example, live in
`ProviderSpec.settings.questions` and never appear in a pack, a result or the cache identity
as anything but an opaque provider setting. See [providers](providers.md).

## Decision Packs

A [pack](packs.md) is the reusable, versioned unit. It composes four independently versioned
components plus metadata:

- `state` - a `StateBuilder`: a JSON Schema contract plus an optional projection field list.
- `judgment` - the primitive, question and labels.
- `policy` - optional; `kind: "none"` is first-class and the default.
- `evaluation` - the gates a candidate must clear before promotion.

Purpose words such as routing, filtering, gating, scoring, feature extraction and monitoring
are `metadata`, not types. The engine treats every pack identically.

A **Pack Bundle** (`Bundle`) names several `id@version` packs that share one StateBuilder
contract. The engine evaluates them independently against one normalized state and returns
one result per pack; it never collapses them into a single opaque decision. When the provider
advertises `multi_question`, the bundle becomes one provider-native call.

## Execution shapes

All four are first-class:

| Shape | Entry point |
| --- | --- |
| one state × one pack | `evaluate` / `decide` |
| one state × bundle | `evaluate --bundle` |
| many states × one pack | `batch --pack --states` |
| many states × bundle | `batch --bundle --states` |

`BatchRunner` fans out in bounded chunks (`--chunk-size`, default 128), checkpoints every
cell (`state_index:pack_index`) durably before moving on, and associates results by index
rather than by arrival order. A job is identified by the digest of its complete definition -
states, packs, binding, execution policy - so `--resume` can never silently run different
inputs, and reusing a job ID with different inputs is a `job_conflict` error. Failures are
per-cell: a job ends `PARTIAL` (exit status 1) with the successful cells intact, and
`--retry-failed` retries only those.

## Inference identity and caching

The cache key is a SHA-256 digest over canonical JSON of:

```
{schema, normalized state, pack inference definition (id, version, StateBuilder, judgment),
 provider identity hash, calibration, workload, evaluation scope}
```

The provider identity hash covers adapter, id, model, version, adapter version, endpoint,
timeout, capabilities and settings - everything except the credential environment variable
name, which affects authorization and not inference. A provider or model upgrade therefore
produces a completely disjoint key space: **reuse across incompatible versions is impossible
by construction**, not by policy. The same property gives deterministic replay, research
reuse, debugging and provider comparison.

Cache writes are fenced. A worker claims an `inference:<hash>` lease, and the write is
rejected if the lease expired in between (`lease_lost`). Other workers waiting on the same
key never hold a lease while waiting, so overlapping bundles cannot deadlock.

## Concurrency and rate control

Concurrency is bounded by the provider's advertised `concurrency`, enforced through
cross-process lease slots keyed on the provider identity. Request pacing uses a durable
reservation table (`reserve_rate`), so parallel processes on one machine share one rate
budget rather than each assuming the whole thing.

## Storage

One SQLite database under the state root, opened with WAL and `synchronous=FULL`, with four
tables: `records` (bucketed JSON documents), `leases`, `events` and `rates`. Every mutation
runs inside `BEGIN IMMEDIATE`. Registered assets (packs, providers, components, calibrations,
decisions, datasets, regression locks) are written `immutable=True`: rewriting a key with
different content raises `version_conflict` rather than mutating history.

The state root is `$REFLEX_STATE_ROOT`, else `$XDG_STATE_HOME/reflex-axi`, else
`~/.local/state/reflex-axi`; directories are `0700` and the database is `0600`. A symlinked
state root is refused. Nothing runtime - caches, evidence, jobs, candidates, learned
calibration, datasets - is ever written into the source tree. Explicit exports
(`--output`, `setup --output`) are written atomically under a lock file with an fsync of both
the file and its parent directory.

Records are namespaced by `project_key()` (the digest of the working directory) so that
deployments, decisions, evidence and jobs from different projects sharing a state root cannot
contaminate each other.

## Output

Default output is compact TOON (encoder in `src/reflex_axi/toon.py`) - a small tabular
encoding that costs far fewer tokens than JSON for uniform rows. Defaults are deliberately
small: `evaluate` prints id, pack, selected and confidence. Full detail is opt-in through
`--full`, `--json`, `--fields` and `--output`. Every error is a structured
`{error, code, help}` record; exit status is 0 on success, 1 on runtime/provider failure or a
`PARTIAL` batch, and 2 on usage/validation errors.

## Evidence, optimization and lifecycle

Improvement is Backpass-shaped: collect cheap evidence silently, spend nothing per decision,
and periodically run one strong sweep only when recurring gaps justify it. Two things are
kept strictly separate:

- **Readiness** is deterministic: are there `minimum` distinct failing decisions in one gap
  bucket, and is accuracy drifting?
- **Authorization** is administrative: a `Scope` independently grants `readable`,
  `evaluatable`, `optimizable` and `allowed_to_mutate`.

Evidence can be observable without ever being learnable. Protected evidence is excluded from
optimization permanently and no quantity of it changes that. See
[lifecycle and evidence](lifecycle.md).

## Integration

Reflex runs as a process and publishes lifecycle events to a durable, monotonically numbered
event log read with a cursor (`events --after N`). Supervisors such as FirstMate consume that
log externally; Reflex requires no FirstMate fork, plugin or modification, and routine
evidence collection emits nothing. See `examples/event_watch.py`.

## Deliberate non-goals

No distributed queue, broker or server. No domain logic: finance, research, trading,
portfolio, market data, feature engineering and model-quality metrics such as IC or Sharpe
belong in the consuming system. No heavyweight ML runtime in the core - distillation defines
a `Trainer` protocol and ships a deterministic fixture implementation. No provider lock-in:
Jev is one adapter among four.

## Performance evidence

`scripts/benchmark.py` runs as part of `scripts/check.py` and writes a machine-readable
record to `.scratch/performance.json`. It asserts two properties rather than chasing absolute
numbers, which are machine-dependent:

- **Version fast path.** `reflex-axi --version` must stay within `3x + 5ms` of bare
  interpreter startup. `entry.py` answers `-v` / `-V` / `--version` before importing the
  command graph, Pydantic or jsonschema. Measured on CPython 3.11.15 / macOS arm64: 10.2 ms
  median against a 9.2 ms interpreter floor, a ratio of 1.11.
- **Deterministic cache reuse.** 256 states through one pack, run twice: the second pass must
  make **zero** additional provider calls, must reproduce every `inference_hash` exactly, and
  every result must report `cache_hit`. Measured: 283 ms cold (2 native batch calls) against
  146 ms warm (0 calls).

These are mock-provider timings, so they measure framework overhead, not live inference
latency or semantic quality. The same reuse is observable through the installed CLI: a
repeated `evaluate` reports `cache_hit: true` with `latency_ms: 0.0` and `cost: 0.0`.
