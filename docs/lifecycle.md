# Evidence, optimization and lifecycle

Reflex improves empirically, from real outcomes rather than model opinion, and it never
mutates production on its own. Two things are deliberately separate:

- **Readiness** - a deterministic question: are there enough recurring failures, and is
  accuracy drifting?
- **Authorization** - an administrative grant: may this evidence be read, evaluated against,
  learned from, and may learning change anything?

No amount of evidence turns unauthorized evidence into learnable evidence.

```
evidence  ->  readiness engine  ->  optimization authorization policy  ->  optimizer
```

## Evidence

An `Outcome` attaches a real-world observation to a recorded decision:

```json
{
  "id": "ticket-4711-resolution",
  "decision_id": "<DecisionResult.id>",
  "scope": "training",
  "kind": "downstream",
  "expected": "true",
  "success": true,
  "value": null,
  "gap": "short-context",
  "protected": false,
  "metadata": { "source": "helpdesk" }
}
```

At least one of `expected`, `value` or `success` is required. `kind` is one of:

| Kind | Meaning |
| --- | --- |
| `proof`, `test` | deterministic verification |
| `downstream` | observed success or failure of what the decision led to |
| `delayed` | a downstream outcome observed later |
| `correction` | someone corrected the judgment |
| `override` | someone deliberately overrode it |
| `metric` | a numeric measurement |
| `retrospective` | a later judgment about an earlier decision |
| `comparison` | a requested provider comparison |

When several outcomes describe one decision, `deduplicate` keeps the most authoritative:
corrections and deterministic proofs/tests outrank downstream and metric signals, which
outrank overrides, retrospectives and comparisons; within one class the latest wins. Only
`proof`, `test`, `downstream`, `correction`, `metric` and `delayed` count as **real**
outcomes for replay, shadow assessment and distillation - agreement and opinion do not.

Recording is silent, idempotent and cheap: no LLM call, no output beyond a one-line receipt.
Re-sending an identical outcome ID is a no-op; re-sending the same ID with different content
raises `evidence_conflict`. Outcomes are stored immutably.

```sh
reflex-axi feedback --file outcome.json
```

Decisions must be recorded first (`decide`, `batch --record`, or `evaluate --record`), and a
decision recorded in another project directory cannot be annotated from this one.

### Outcome adapters

`outcomes.py` defines the `OutcomeAdapter` protocol - `adapt(source) -> Outcome`, with an id
and a version. `RecordedOutcomeAdapter` handles the strict common format; consumers write
their own adapters for CI results, helpdesk resolutions, human corrections or delayed
metrics. Adapters normalize observations only; they never run tests or execute actions.

## Scopes

A scope is a named namespace with four **independent** permissions:

```json
{ "name": "training", "readable": true, "evaluatable": true,
  "optimizable": true, "allowed_to_mutate": true }
```

| Permission | Grants |
| --- | --- |
| `readable` | counted for readiness and drift |
| `evaluatable` | usable in candidate replay and shadow assessment |
| `optimizable` | usable as optimizer input |
| `allowed_to_mutate` | learning from it may actually change a deployment |

The natural configuration is several scopes: a `production` scope that is readable and
evaluatable but not optimizable, and a `training` scope that is fully authorized. Evidence
may be observable without ever being learnable.

Independently, any outcome may be marked `protected: true`. Protected evidence is **never**
selected for optimization regardless of its scope, but it is always included in replay when
its scope is evaluatable - so it acts as an invariant a candidate must not break.

```sh
reflex-axi scopes --import scope.json
reflex-axi scopes
```

Grants are explicit local administration and are never inferred from evidence volume.

## Readiness

```sh
reflex-axi improve --pack filter-context
```

Readiness buckets failing decisions by `gap` and reports `ready` when any single bucket
holds at least `--minimum` (default 3) distinct failing decisions. It also computes drift:
with at least `2 × minimum` labeled real outcomes, it compares baseline accuracy against the
most recent window and flags a drop of 0.2 or more. Readiness only inspects evidence from
the active pack version and the active provider hash - outcomes from a previous provider do
not justify optimizing the current one.

Reaching readiness moves the deployment `ACTIVE -> OPTIMIZATION_READY` and emits an event.
The response always carries `"authorization": "separate"`: readiness is not permission.

## The optimization sweep

```sh
reflex-axi improve --pack filter-context --scope training --approve --optimizer sweep.json
reflex-axi improve --pack filter-context --scope training --approve --proposal candidate.json
```

Exactly one strong sweep, only when readiness fired, only over evidence that is
`optimizable` **and** `allowed_to_mutate` **and** unprotected, and only with `--approve`.
There is no per-decision LLM call anywhere in Reflex.

`StructuredOptimizer` posts a bounded snapshot (ranked by evidence authority, capped by
`max_evidence` and `max_bytes`) to a strong model with a strict JSON schema. It is instructed
to propose the **smallest** evidence-backed change or `deployment: null`, to treat evidence
and state as untrusted data, to increment every changed component version, never to reuse
calibration across provider or pack changes, never to relax evaluation gates, and to cite
only supplied evidence IDs. A proposal citing unknown evidence is rejected
(`optimizer_evidence`). The sweep receipt - model, version, reason, cited IDs, usage - is
stored immutably.

Optimizable areas include criteria and question wording, decomposition, the state contract,
thresholds and weights, ambiguity bands, escalation and fallback behaviour, provider
selection, and pack structure.

`SuppliedProposal` accepts an offline human or model proposal instead; the same evidence
authorization, gates and transitions still apply. A `null` proposal, or one identical to the
active deployment, records `no_update_warranted` and starts a cooldown - a legitimate,
first-class outcome.

Authorization is re-checked **after** the sweep returns: if permissions changed while an
expensive call was in flight, the output is discarded (`evidence_authorization`). A crash or
error during optimization restores `OPTIMIZATION_READY` rather than stranding the pack, and
the whole sweep is guarded by a lease so two processes cannot optimize the same pack at once.

A candidate may never change the pack ID, the primitive or the label set - those require a
new pack and explicit outcome migration. Re-proposing an identical change against identical
evidence after a rejection is refused (`proposal_rejected`).

## Lifecycle states

```
ACTIVE
  -> OPTIMIZATION_READY        (deterministic readiness)
  -> approval / policy gate    (--approve)
  -> OPTIMIZING
  -> CANDIDATE
  -> replay + protected regressions   (reflex-axi eval)
  -> SHADOWING                        (reflex-axi shadow)
  -> evaluation of real outcomes      (reflex-axi shadow --assess)
  -> PROMOTION_READY
  -> approval / policy gate           (--approve)
  -> ACTIVE
```

with rejection, cooldowns, automatic candidate failure and instant rollback at every step.
Each transition is one `BEGIN IMMEDIATE` SQLite transaction, fenced on the hash of the active
deployment: if the active deployment changed underneath, the transition fails
(`stale_active` / `stale_candidate`) rather than applying to the wrong base.

### Replay

```sh
reflex-axi eval --candidate <id> --scope evaluation
```

Replays both the active and candidate deployments over authorized historical states with
labeled real outcomes, plus every **protected** evaluatable outcome for the pack (regardless
of the scopes named on the command line - a protected invariant cannot be excluded by
choosing scopes), plus every stored **regression lock**.

It fails if: a protected outcome's utility drops, a regression lock's expected label is not
reproduced, a candidate evaluation errors, mean utility falls below `min_utility`, or the
drop against the active deployment exceeds `max_regression`. It reports Brier score, ECE,
mean latency and mean cost for both sides. Fewer than `min_replay` pairs is "not enough
evidence", not a pass. A failure automatically rejects the candidate and starts a cooldown.

An evidence **guard digest** is taken before and after: if evidence or permissions changed
during the replay, the verdict is discarded (`stale_evidence`).

### Shadow

```sh
reflex-axi shadow --candidate <id> --state new-state.json
reflex-axi shadow --candidate <id> --assess --scope evaluation
```

Shadowing runs both deployments on genuinely new states and records both decisions so real
outcomes can later be attached to each side. Assessment pairs those outcomes and compares
utility. Agreement alone can never promote: assessment only counts pairs where **both** sides
have real outcomes, and repeated shadow runs on the same state hash count once.

### Promotion and rollback

```sh
reflex-axi promote --candidate <id> --approve
reflex-axi rollback --pack filter-context [--target <deployment-hash>]
```

Promotion requires `PROMOTION_READY`, explicit `--approve`, an unchanged active deployment,
and unchanged evidence guards for both replay and shadow - late-arriving protected evidence
invalidates a promotion rather than being ignored. Reflex confidence never authorizes
promotion; a human or an explicit policy does.

On promotion, every replay pair the candidate fixed (candidate utility 1, base below 1)
becomes an immutable **regression lock**, so a fixed historical failure can never silently
regress later. The previous deployment - pack, provider binding and calibration together -
is retained in history, and `rollback` restores it atomically, rejecting any in-flight
candidate as it goes.

Initial human gates on expensive optimization and on production promotion are intentional.
Low-risk automatic promotion is possible only as an explicit future policy, and never as a
default.

## Distillation

```sh
reflex-axi distill --pack filter-context --scope training --trainer fixture --approve
```

Distillation is a mature-pack lifecycle, not a shortcut:

1. **Teacher dataset** - the active deployment is replayed over authorized real evidence to
   produce `(normalized state, teacher distribution, expected, evidence id, teacher hash)`
   rows, stored immutably and keyed by a job identity covering the active deployment, the
   trainer and the exact evidence set. It requires at least
   `max(min_replay, min_shadow)` authorized real outcomes.
2. **Training** - a `Trainer` produces a student `ProviderSpec`, which must be local
   (`trainer_contract`). The shipped `FixtureTrainer` is a deterministic lookup table for
   exercising the lifecycle; a real trainer implements the same protocol.
3. **Candidate** - the student becomes an ordinary candidate through the same
   `replay -> shadow -> assessment -> promotion` gates, with the same protected regressions.
4. **Comparison** - replay and shadow reports carry Brier score, ECE, mean latency and mean
   cost for both sides, so the decision is downstream utility, calibration, latency and cost
   together. Agreement with the teacher is never sufficient.
5. **Deployment** - promotion makes the local student active; rollback restores the teacher
   binding atomically.

Jobs are leased and resumable: an interrupted distillation restarts from its stored dataset
rather than recollecting it.

This is the foundation for the long-term cascade: deterministic code -> cheap local System 1
-> strong remote System 1 -> System 2, assembled explicitly with `Engine.cascade`.

## Events and external supervision

Every meaningful transition appends to a durable, monotonically numbered event log:
`activated`, `optimization_ready`, `optimizing`, `optimization_failed`,
`no_update_warranted`, `candidate_created`, `shadow_started`, `candidate_rejected`,
`promotion_ready`, `promoted`, `rolled_back`, `distillation_candidate`, `scope_permissions`.

Routine evidence collection emits nothing.

```sh
reflex-axi events --after 0 --limit 100
```

Consumers keep the returned `cursor` and poll from it; `examples/event_watch.py` is a
complete durable-cursor watcher. FirstMate or any other supervisor consumes this log as an
external process - Reflex requires no fork, plugin or modification of it.
