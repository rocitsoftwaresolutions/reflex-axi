# CLI and integration

`reflex-axi` is an AXI-style agent interface: content-first, compact by default, never
interactive, strict about unknown input, and structured on failure.

## Conventions

- **Compact TOON by default.** Uniform row collections become a header plus one line per
  row, which costs far fewer tokens than JSON. `--json` switches to strict JSON.
- **Small defaults, explicit detail.** `evaluate` prints four fields. `--full` adds the whole
  record, `--fields a,b` projects top-level fields, `--output FILE` writes the rendered
  result atomically and prints a one-line receipt instead.
- **No prompts, ever.** Nothing waits on stdin except `--state -`.
- **Strict unknown input.** Abbreviated and unknown flags are rejected before any state,
  storage or network access, and the error lists the command's valid flags.
- **Structured errors.** Every failure is `{error, code, help}` in the active format.
- **Exit status.** `0` success, `1` runtime/provider failure or a `PARTIAL` batch, `2` usage
  or validation error.
- **Fast version path.** `-v`, `-V` and `--version` print the version without importing the
  command graph, Pydantic or jsonschema (measured at ~1.06× bare interpreter startup; see
  `scripts/benchmark.py`).

Every command has full `--help` with a description, per-flag help and examples, and help
never touches storage.

## Home view

```
$ reflex-axi
bin: ~/.local/bin/reflex-axi
description: "Fast judgment before expensive thought; bounded probabilities, optional policy, no actions."
active: []
count: 0
status: 0 active deployments in this directory; built-in packs use the mock fixture until configured
help[3]: reflex-axi evaluate --pack filter-context --state state.json,...
```

The bare invocation is content, not a banner: the packs active in this directory, a
deterministic empty state when there are none, and contextual next steps. `--full` adds full
deployment records and this directory's jobs.

## Global flags

| Flag | Effect |
| --- | --- |
| `--state-root PATH` | private runtime directory (else `$REFLEX_STATE_ROOT`, else XDG state) |
| `--json` | strict JSON instead of TOON |
| `--full` | complete records |
| `--fields a,b` | project top-level output fields |
| `--output FILE` | write the rendered output atomically; stdout gets a receipt |

They are accepted before or after the subcommand.

## Inference flags

Shared by `evaluate`, `decide`, `batch` and `compare`:

| Flag | Effect |
| --- | --- |
| `--pack` | pack ID, `id@version`, or a JSON file |
| `--bundle` | versioned Pack Bundle JSON file |
| `--provider` | binding/provider JSON file, `id@version`, or `mock` |
| `--local-only` | reject any provider that is not local |
| `--no-cache` | bypass the deterministic inference cache |
| `--research` | judgment only: no evidence, no lifecycle writes |
| `--record` | retain decisions so outcomes can be attached later |
| `--workload`, `--scope` | calibration workload and evaluation scope |

`--pack` resolves in order: a file path, this directory's active deployment, a built-in
starter pack, then a registered `id@version`. With no `--provider`, the pack's active
binding is used if the versions match, otherwise the explicit mock fixture.

## Commands

### `evaluate` - bounded judgment

```sh
reflex-axi evaluate --pack filter-context --state state.json
reflex-axi evaluate --bundle bundle.json --state state.json --full
```

Returns the distribution only. Never attaches a policy, and does not record by default.

### `decide` - judgment plus optional recommendation

```sh
reflex-axi decide --pack route-model --state state.json
reflex-axi decide --pack route-model --state state.json --policy none
```

`--policy none` is first-class and suppresses the policy even for packs that define one.
A recommendation always carries `executed: false` and `authorization: external`; Reflex
never performs the action.

`decide` records the decision so outcomes can be attached later, unless `--research` is set.

### `batch` - many states, resumable

```sh
reflex-axi batch --pack filter-context --states states.jsonl --job nightly
reflex-axi batch --bundle bundle.json --states states.jsonl --results
reflex-axi batch --resume nightly --retry-failed
```

One JSON object per non-blank line. Cells are checkpointed durably as
`state_index:pack_index`, so a killed process resumes exactly where it stopped. `--resume`
replays the job's saved inputs and rejects any inference flag, so a resume can never
silently run something different. Partial jobs report `PARTIAL` and exit 1 with successful
cells intact; `--retry-failed` retries only the failures. `--chunk-size` (default 128)
bounds the checkpoint interval.

### `feedback` - record an outcome

```sh
reflex-axi feedback --file outcome.json
```

Silent and idempotent. See [lifecycle](lifecycle.md#evidence).

### `scopes` - evidence permissions

```sh
reflex-axi scopes
reflex-axi scopes --import scope.json
```

### `packs`, `providers` - immutable versioned registries

```sh
reflex-axi packs
reflex-axi providers --import provider.json --full
```

`packs` also lists the built-in starter packs that have not been overridden by a
registration.

### `activate` - initialize an active deployment

```sh
reflex-axi activate --file deployment.json
```

Deliberately one-shot: once a pack is active, changes go through a candidate, not through
re-activation.

### `status`, `events`

```sh
reflex-axi status --full
reflex-axi events --after 0 --limit 100
```

`events` is the external integration surface: a durable, monotonically numbered log with a
cursor. See `examples/event_watch.py`.

### `improve`, `candidates`, `eval`, `shadow`, `promote`, `rollback`, `distill`

The lifecycle commands, documented in [lifecycle](lifecycle.md). Without `--approve`,
`improve` reports readiness only and changes nothing expensive.

### `compare` - explicit provider comparison

```sh
reflex-axi compare --pack filter-context --state state.json \
  --provider a.json --against b.json
```

Runs two pinned bindings and reports agreement plus total-variation distance. Ordinary
inference never does this.

### `setup` - opt-in session integration

```sh
reflex-axi setup --app codex --output codex-hooks.json
reflex-axi setup --app claude --output claude-hooks.json
reflex-axi setup --app opencode --output reflex-plugin.js
```

Writes a snippet for review. It never installs anything, never edits an existing harness
config, and the snippet only runs `reflex-axi status`. `--output` is required.

## Research mode

`--research` makes a run judgment-only: no evidence recording, no lifecycle writes, no
optimization, no promotion, no mutation. Caching and batching stay on, so large offline
sweeps remain cheap and reproducible. It cannot be combined with `--record`, and a research
batch cannot record or decide.

```sh
reflex-axi batch --bundle bundle.json --states corpus.jsonl --research --results --json \
  --output features.json
```

## Library use

Reflex is independently usable without the CLI:

```python
from reflex_axi.catalog import builtin_packs, default_binding
from reflex_axi.engine import Engine
from reflex_axi.store import Store

store = Store("/path/to/state")
engine = Engine(store)
pack = next(p for p in builtin_packs() if p.id == "filter-context")
result = engine.evaluate(pack, {"task": "...", "context": "..."}, default_binding())
print(result.selected, result.confidence, result.distribution)
```

`examples/library_usage.py` is a complete runnable walkthrough covering `evaluate`,
`policy:none`, `decide`, bundles, deterministic cache reuse, batch with resume, and an
explicit cascade.

## Installation

```sh
uv tool install .                                    # from a checkout
uv tool install git+https://github.com/rocitsoftwaresolutions/reflex-axi
uvx --from git+https://github.com/rocitsoftwaresolutions/reflex-axi reflex-axi --version
pip install .                                        # Python 3.11+, any PEP 517 frontend
```

Runtime dependencies are Pydantic and jsonschema only; everything else is the standard
library. POSIX (macOS, Linux); Windows is not currently supported.

For agent harnesses, `skills/reflex-axi/SKILL.md` is a generated on-demand skill, and
`setup --app` produces a reviewable session-start snippet. Both are optional and
independent.

## State and secrets

Runtime state lives in `$REFLEX_STATE_ROOT`, else `$XDG_STATE_HOME/reflex-axi`, else
`~/.local/state/reflex-axi` - never in the source tree. Directories are `0700`, the database
and exports `0600`, and a symlinked state root is refused. Provider credentials are read
only from the environment variable named by `api_key_env`; no secret is ever stored,
logged, or included in an inference identity.

## Provider benchmarks

`benchmark list` discovers suites; `benchmark run --provider <file-or-id> --suite starter`
creates a protected run. `benchmark runs` lists saved runs; `benchmark resume --run <id>`
continues checkpoints; `benchmark show --run <id> --full --json` inspects complete artifacts.
`benchmark compare --run <a> --against <b>` compares only when explicitly invoked.
See [benchmark methodology and flags](benchmarks.md).

OpenRouter/Jev: register `examples/openrouter-jev.json` with `providers --import`. The example
is intentionally disabled pending independent snapshot verification. See
[setup and live limitations](openrouter-jev.md); `providers --full` includes capability caveats.
