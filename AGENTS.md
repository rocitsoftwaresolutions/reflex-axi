# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Verification

`uv run python scripts/check.py` is the single gate and is what CI runs. It covers ruff
lint + format, mypy, generated-skill freshness, pytest, wheel/sdist build, an isolated
installed-wheel smoke test, and the performance benchmark. Run it before any commit; do not
run the steps piecemeal and assume the gate passes.

`scripts/generate_skill.py` writes `skills/reflex-axi/SKILL.md` from `catalog.py`. Never
edit that file by hand - the check fails on staleness.

## Invariants that must survive every change

These are the product, not implementation details. Tests enforce them; read `docs/` before
changing anything here.

- **No action execution.** `decide` returns a recommendation with `executed: false`. No
  confidence value ever authorizes a side effect.
- **`policy: none` is first-class**, and is the default for a pack that omits a policy.
- **Provider neutrality.** Backend-specific concepts (Jev's Noul IDs, for example) live in
  `ProviderSpec.settings` and never reach a pack, a result, or the core. No implicit
  fallback between providers, ever.
- **Inference identity is the safety mechanism.** The cache key covers normalized state,
  StateBuilder version, pack version, provider identity hash, calibration and workload/scope,
  so reuse across incompatible versions is impossible by construction. Anything added to
  inference must be added to the identity.
- **Readiness is not authorization.** Deterministic readiness and scope permissions are
  separate; protected evidence is never learnable regardless of quantity.
- **Every lifecycle transition is one fenced SQLite transaction** guarded on the active
  deployment hash, with leases for cross-process work. Writes are atomic and crash-safe.
- **Nothing runtime is written into the source tree.** State, caches, evidence, jobs,
  candidates, datasets and secrets live under the state root only.

## Conventions

- Reproduce a bug through the console entry point or the provider boundary before fixing it;
  provider tests exercise serialized HTTP through injected byte-stream transports; the suite
  forbids network sockets. OpenRouter/Jev snapshot gates and limitations are in
  `docs/openrouter-jev.md`.
- Keep default CLI output small and add detail behind `--full` / `--json` / `--fields` /
  `--output`.
- Add adapters rather than core domain logic. Finance, research and other domain concerns
  belong in consuming systems, not here.
- `docs/` is the durable reference: architecture, packs, providers, lifecycle, cli, migration.
  Provider benchmark isolation and reproducibility are in `docs/benchmarks.md`; optional
  GLiNER storage gates and offline runtime pins are in `docs/gliner.md`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
