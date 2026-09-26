# reflex-axi

**Fast judgment before expensive thought.** Reflex is a provider-neutral System-1 judgment
engine, independently usable as a Python library or a compact TOON CLI. It never executes
an action. Run deterministic code first, bounded fuzzy judgment second, and System-2
reasoning or generation only when needed.

Requires Python 3.11+ on macOS or Linux. Install from the repository:

```sh
uv tool install .
reflex-axi --version
reflex-axi evaluate --pack filter-context --state examples/state.json
reflex-axi decide --pack filter-context --state examples/state.json --policy none --full
reflex-axi batch --bundle examples/bundle.json --states examples/states.jsonl --results
```

Without an active deployment or explicit provider, built-in packs use the **mock fixture**.
It does not perform semantic judgment. Configure a provider binding before real use.
Starter packs are useful definitions, not prevalidated or calibrated production models.

`evaluate` returns probabilities. `decide` optionally attaches a policy recommendation.
`policy:none` and `--research` are first-class; research never records evidence, optimizes,
promotes, or changes active deployments. Cache and batch checkpoints remain available.
Use `--full`, `--json`, `--fields`, and `--output` for explicit detail.

Runtime state defaults to `$XDG_STATE_HOME/reflex-axi` (or `~/.local/state/reflex-axi`),
outside source control. Override with `REFLEX_STATE_ROOT` or `--state-root`.
State directories are mode 0700; the SQLite database and exports are mode 0600.
Provider secrets come only from configured environment variables.

Read [architecture](docs/architecture.md), [pack authoring](docs/packs.md),
[provider contracts](docs/providers.md), [lifecycle and evidence](docs/lifecycle.md),
[CLI and integration](docs/cli.md), and [development](CONTRIBUTING.md).

Two runnable examples cover the independent paths. `examples/library_usage.py` embeds Reflex
without the CLI: judgment, `policy:none`, recommendation, bundles, deterministic cache reuse,
resumable batch, explicit cascade and research mode. `examples/event_watch.py` is the optional
external integration: a durable-cursor watcher over `reflex-axi events` that notifies only
meaningful lifecycle transitions.

```sh
REFLEX_STATE_ROOT=$(mktemp -d) python examples/library_usage.py
REFLEX_STATE_ROOT=$(mktemp -d) python examples/event_watch.py --once
```

For ambient session discovery, `setup --app codex|claude|opencode --output <file>`
generates a reviewable integration snippet. Merge it into the harness's project config;
ordinary commands never install hooks. Alternatively install the on-demand
[agent skill](skills/reflex-axi/SKILL.md). Either discovery path works independently.

`decision-axi` was a concept, with no implementation or stored data. See
[conceptual migration](docs/migration.md). No FirstMate modification or fork is required.

Development:

```sh
uv venv
uv pip install -e '.[dev]'
.venv/bin/python scripts/check.py
```


The optional [OpenRouter/Jev adapter](docs/openrouter-jev.md) uses native Decisions API
probabilities. Its example configuration is disabled pending independent verification of the
dated model snapshot. Implementation and benchmarks are offline fixture verified only.

## Provider fitness and optional GLiNER

The generated [agent skill](skills/reflex-axi/SKILL.md) now includes configuration, examples,
command selection and the evidence/lifecycle boundaries. Generate it from `catalog.py`;
CI checks drift. Upgrade an existing core tool install with `uv tool install --reinstall .`
from a reviewed checkout. This replaces the executable without rewriting provider or harness
configuration. Setup continues to export opt-in snippets for manual merging.

```sh
reflex-axi benchmark list
reflex-axi benchmark run --provider mock --suite starter
reflex-axi benchmark runs
reflex-axi benchmark show --run <id> --full --json --output /private/path/run.json
```

[Provider benchmarks](docs/benchmarks.md) separate conformance, quality/calibration,
performance and resilience. Runs are protected, resumable and cannot train or promote a
provider. Mock runs test the harness, not semantic fitness.

[GLiNER2.5-Decide setup](docs/gliner.md) documents the optional `gliner` extra, offline
snapshot manifest, runtime pins and tested adapter contract. Real inference is unvalidated:
the inspected machine had only 2.64 GiB free, insufficient for the 1.95 GB weights plus runtime,
caches and operating margin. No weights were downloaded. Do not install the optional runtime
until the documented capacity check passes.
