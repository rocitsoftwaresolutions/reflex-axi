# Migrating from `decision-axi`

## There is nothing to migrate

`decision-axi` was a **name for a concept**, not a shipped system. No implementation, no
package, no CLI, no stored state, no evidence, no calibration and no packs ever existed under
that name. `reflex-axi` is a clean, from-scratch build - not a rename, a fork, or an
incremental cleanup of an earlier codebase.

Concretely:

- There is no `decision-axi` package to uninstall, and no version to pin.
- There is no `decision-axi` state root, database, cache or evidence store to convert.
  Reflex's first run creates its own state at `$REFLEX_STATE_ROOT` /
  `$XDG_STATE_HOME/reflex-axi` / `~/.local/state/reflex-axi`.
- There are no old pack, provider or calibration files to translate. Author packs against
  the current contract in [pack authoring](packs.md).
- There is no data-migration command, and none will be added. If one ever becomes necessary
  for a real predecessor, it will arrive as an explicit versioned command rather than as
  implicit conversion at startup.

If you find a document, ticket or comment that mentions `decision-axi`, read it as an early
sketch of the ideas below and treat this repository as the authority.

## What changed conceptually

The concept was framed around "decisions". Reflex is deliberately narrower and more honest
about what it does:

| Earlier framing | What Reflex ships |
| --- | --- |
| a decision system | a **judgment engine**: `evaluate` returns probabilities; `decide` adds an optional *recommendation*; nothing is ever executed |
| decisions imply actions | `policy: none` is the default and first-class; a policy result always carries `executed: false, authorization: external` |
| a provider integration | a provider-neutral core with four adapters; Jev is one backend, and no backend-specific concept reaches the core |
| one decision definition | separately versioned StateBuilder, judgment, policy, evaluation and metadata inside a versioned Pack |
| decision types (routing, filtering, gating, scoring) | three primitives - `binary`, `choice`, `score` - with purpose words demoted to `metadata` |
| a single call | four first-class execution shapes, including bundles and resumable batches |
| learning from feedback | evidence with four independent permissions, deterministic readiness kept strictly separate from authorization, and gated candidate -> replay -> shadow -> promotion |

The naming reflects that: **Reflex** is System-1 judgment - fast, bounded, cheap, and always
subordinate to deterministic code and to explicit authorization for anything that acts.

## Adopting Reflex

1. Install it and confirm the version: `uv tool install .` then `reflex-axi --version`.
2. Try a starter pack against the mock fixture:
   `reflex-axi evaluate --pack filter-context --state examples/state.json`.
3. Configure a real provider ([provider contracts](providers.md)) and register it with
   `reflex-axi providers --import`.
4. Write or adapt a pack ([pack authoring](packs.md)) and `activate` a deployment.
5. Record outcomes and configure scopes before enabling any optimization
   ([lifecycle](lifecycle.md)).

No FirstMate fork, plugin or modification is required at any step; supervision is external
and event-driven.
