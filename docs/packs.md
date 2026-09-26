# Pack authoring

A **Decision Pack** is the reusable unit of judgment: a versioned state contract, a versioned
judgment definition, an optional versioned policy, versioned evaluation gates, and metadata.
Each component carries its own version, and each is registered immutably, so components can
evolve independently while remaining reproducible.

Packs are strict. Unknown keys are rejected, values are type-strict (no string-to-number
coercion), and every model is frozen after validation.

## Minimal pack

```json
{
  "id": "my-pack",
  "version": "1",
  "state": {
    "version": "1",
    "schema": {
      "type": "object",
      "properties": {
        "task": { "type": "string", "maxLength": 16000 },
        "context": { "type": "string", "maxLength": 32000 }
      },
      "required": ["task", "context"],
      "additionalProperties": false
    }
  },
  "judgment": {
    "version": "1",
    "primitive": "binary",
    "labels": ["false", "true"],
    "question": "Is this context materially useful for the task?"
  }
}
```

`policy` defaults to `{"kind": "none"}` and `evaluation` to its defaults, so the smallest
useful pack is judgment only.

Register it:

```sh
reflex-axi packs --import my-pack.json
reflex-axi packs --full
```

Registration is immutable per `id@version`. Changing content without bumping the version is
a `version_conflict` error.

## `state` - the StateBuilder

```json
{ "version": "1", "schema": { ... }, "fields": ["task", "context"] }
```

- `schema` must be a valid Draft 2020-12 schema, must have `"type": "object"`, and must set
  `"additionalProperties": false`. External `$ref` / `$dynamicRef` targets are forbidden: no
  network or filesystem resolution happens while validating untrusted state.
- `fields` is an optional projection. Validation always runs **before** projection, so a
  misspelled input fails loudly instead of silently disappearing.
- Non-finite numbers are rejected anywhere in the state, including nested objects.

The StateBuilder version is part of the inference identity. Changing the schema or the
projection changes every cache key for the pack, which is intended: old results were produced
from a different normalization.

Keep state small and stable. Everything the provider sees goes through here, so exclude
volatile fields (timestamps, request IDs, counters) that would defeat caching without
changing the judgment.

## `judgment`

```json
{
  "version": "1",
  "primitive": "choice",
  "labels": ["cheap", "strong", "system2"],
  "question": "Judge required semantic depth ..."
}
```

- `binary` labels must be exactly `["false", "true"]`; the expected score is the probability
  of `true`.
- `choice` takes 2-100 distinct non-empty labels.
- `score` additionally requires `values`, a number per label; the result's `expected_score`
  is the probability-weighted value. Use it when downstream code wants a magnitude rather
  than a winner.

Write the question as a **judgment criterion**, not a task. Good questions are bounded,
answerable from the state alone, and free of arithmetic, generation or tool use. If the
question needs the model to compute, look something up, or produce prose, it belongs in
deterministic code or System 2 instead.

## `policy` (optional)

```json
{
  "version": "1",
  "kind": "recommend",
  "min_confidence": 0.8,
  "actions": { "cheap": "cheap-system1", "strong": "strong-system1" },
  "uncertain": "escalate"
}
```

`kind: "none"` is the default and is first-class: `evaluate` never attaches a policy, and
`decide --policy none` suppresses it even for packs that define one. When `kind` is
`recommend`, `decide` attaches:

```json
{ "version": "1", "recommendation": "...", "uncertain": false,
  "executed": false, "authorization": "external" }
```

`executed` is always `false`. A recommendation is advice to the caller; the caller owns
authorization. Below `min_confidence` the recommendation becomes `uncertain`, which should
route to a stronger tier rather than to a guess. Every key in `actions` must be a declared
label.

## `evaluation`

Gates a candidate must clear before it can be promoted:

| Field | Default | Meaning |
| --- | --- | --- |
| `min_replay` | 3 | labeled historical outcomes required for a replay verdict |
| `min_shadow` | 3 | real paired shadow outcomes required for a promotion verdict |
| `max_regression` | 0.0 | tolerated utility drop against the active deployment |
| `min_utility` | 0.0 | absolute utility floor for the candidate |
| `utility` | `accuracy` | `accuracy`, or `squared_error` against `values` for score packs |

Raise `min_replay` / `min_shadow` for packs whose mistakes are expensive. These gates cannot
be relaxed by an optimizer proposal.

## `metadata`

Free-form string map. Purpose labels live here and are never primitives:

```json
{ "purpose": "routing", "status": "starter-unvalidated", "owner": "platform" }
```

## Bundles

A bundle evaluates several versioned packs against one normalized state, independently:

```json
{
  "id": "result-triage",
  "version": "1",
  "packs": ["result-sufficiency@1", "retry-worthiness@1", "escalation-need@1"]
}
```

Every reference must be `id@version` and unique, and all packs must share **one identical
StateBuilder** - otherwise `bundle_state` is raised. Bundles never merge results; you get one
`DecisionResult` per pack. On a provider advertising `multi_question`, the whole bundle is a
single call.

## Starter packs

Shipped inside the wheel (`src/reflex_axi/packs/`), all domain-agnostic:

| Pack | Primitive | Purpose |
| --- | --- | --- |
| `route-skill` | choice | `none` / `existing` / `discover`: does a described specialist skill fit |
| `route-model` | choice | `cheap` / `strong` / `system2`: required semantic depth |
| `route-tool` | choice | `none` / `read` / `write` / `reason`: which tool class fits |
| `filter-context` | binary | is this context relevant to the task |
| `result-sufficiency` | score | `insufficient` / `partial` / `sufficient`, with numeric values |
| `retry-worthiness` | binary | would another attempt plausibly resolve a transient failure |
| `escalation-need` | binary | does this exceed bounded System-1 judgment |

The last three form a deliberate **decomposition** rather than one overloaded "what now"
pack: each is separately versionable, separately calibratable, separately replaceable, and
the bundle runs them as one provider call. Compose them with `examples/bundle.json`.

All starter packs carry `"status": "starter-unvalidated"`. They are useful definitions, not
calibrated production models. Validate and calibrate them on your workload first.

## Authoring checklist

1. Can deterministic code answer this? If yes, do not write a pack.
2. Is the output a small bounded label set? If it needs prose, it is System 2 work.
3. Is the state minimal, stable and schema-closed?
4. Is the question answerable from the state alone, with no arithmetic?
5. Would one overloaded pack be better as a decomposed bundle?
6. Start with `policy: none`; add a policy only once the caller genuinely wants a
   recommendation.
7. Set `evaluation` gates proportional to the cost of being wrong.
8. Version every component you change, and never reuse a version.
