# Provider benchmarks

Use this harness to inspect one explicitly selected provider, not to promote it. Benchmark
records are private, protected observations in dedicated SQLite buckets. They never enter
`decisions`, `evidence`, optimizer input, distillation datasets, cache or deployment tables.
Even granting a scope named `benchmark` learning permission cannot make these records
learnable. Comparison is an explicit operation and never invokes another provider.

```sh
reflex-axi benchmark list
reflex-axi benchmark run --provider mock --suite starter --suite robustness \
  --mode single --mode batch --repeats 2 --concurrency 2
reflex-axi benchmark runs
reflex-axi benchmark resume --run <id>
reflex-axi benchmark show --run <id> --full --json --output /private/path/run.json
reflex-axi benchmark compare --run <a> --against <b>
```

`--provider` is required: `mock`, a registered `id@version`, or a provider/binding JSON file.
No implicit active-deployment selection. Default suite/mode selection is all; default repeats
is 2 and concurrency 1. Limits are 20 repeats and 32 workers, additionally capped by the
provider's capability. Calls obey the same durable rate/concurrency controls as normal
inference. The default TOON receipt reports id, status, conformance, accuracy, calls and
failures. `--full` includes metadata, separate metric groups, every case and provider-call
trace. `--json` changes encoding; it does not itself expand the compact schema.

## What is measured

The versioned, reviewable corpus is `src/reflex_axi/benchmarks/agentic-v1.json`. It contains
61 authored scenarios, with all seven starter packs and all three primitives. No model
writes expected labels during a run. These are diagnostic expectations, not a representative
production sample or a statistically sufficient calibration dataset. Ambiguous, contradictory
and out-of-domain cases intentionally have no correctness label; they carry an escalation
expectation and do not contaminate labeled accuracy/Brier/ECE denominators.

| Suite | Coverage |
| --- | --- |
| starter | relevance, sufficiency, retry value, escalation, model/skill/tool routing; all labels |
| robustness | injected instructions, negation, contradiction, OOD, short/long input, structured noise, paraphrases |
| boundaries | ambiguous confidence bands, partial sufficiency, policy:none |
| conformance | schema/type rejection, valid neighbors in partial batches, deterministic association |

Each selected scenario is run singly, in same-pack batches, and in a cross-product bundle
with every selected pack. Additional bundle cells have no invented labels. Native batches
are used only when advertised. Multi-question providers can combine packs; other providers
use ordinary per-pack grouping. State/pack results retain stable cell IDs regardless of
completion order, and raw response IDs/distributions are validated before accepting results.
Repeated identical inputs are deduplicated within one engine call; provider traces show the
actual call/request counts, not inflated logical-cell throughput. Repeats disable the cache.

Results remain separated:

- **Conformance:** expected rejection/success rate, complete finite normalized distributions
  and exact response association, no-action policy safety, repeat/mode consistency and
  paraphrase agreement plus total-variation distance.
- **Quality/calibration:** first mode/first repeat only, each authored case once; labeled
  accuracy, multiclass Brier score, ten-bin ECE, negative log likelihood, ordinal score MAE,
  per-suite/per-pack metrics and selective coverage/risk. At thresholds 0.5, 0.8 and 0.95, report
  ambiguity escalation and an explicit utility schedule: correct +1, wrong -1, defer 0.
  `null` means not measured/undefined, never a perfect score. Policy confidence is not authority.
- **Performance:** actual provider-call timing, whole execution wall time, request throughput,
  largest native batch, observed call concurrency, latency percentiles, process-first and warm calls, process peak RSS, actual reported usage
  and cost. Missing usage is marked incomplete; unknown or failed-call cost is not invented
  as zero. Local inference has zero API cost, not zero hardware/energy cost.
- **Resilience:** expected rejections, error counts by normalized code, unexpected failures,
  successful cells and durable checkpoint chunks. One failed native batch invalidates that
  entire call; independently completed calls remain available.

Confidence-band behavior is also tested deterministically just below/at/above the policy
threshold in CI. CI injects missing/invalid distributions, row reordering, missing IDs,
transport failures, interruptions and provider exceptions. Real loopback HTTP drives the Jev
adapter under native batching and bounded concurrent execution. GLiNER fixtures match its
public raw-logit scoring interface without installing a model or ML runtime. These fault
experiments verify adapter/framework handling; they do **not** claim a live provider was
subjected to network faults or that a model will handle arbitrary outages well.

A model with declared deterministic/seeded capabilities still needs repeat and execution-mode
checks; deterministic settings are pinned in provider identity. Different seeds require
separate provider versions/runs and an explicit comparison. The harness does not mutate
settings or calibration to improve its own score. Supplied calibration must match the
provider, pack, workload `benchmark`, and scope `benchmark`; a calibration for one pack
cannot be applied to a suite covering other packs.

## Reproducibility, checkpoints and limitations

Each run records immutable benchmark/corpus versions and hashes, complete case and pack
snapshots, provider/model/settings/capabilities/hash, calibration, Python/platform/installed
package versions and Reflex implementation hash, concurrency/repeats/modes, timestamps and protection flags. Per-chunk
records include results, errors, raw valid distributions, request associations, timing,
RSS, a process-session identifier and completion time. GLiNER metadata also records
logical model and pinned-runtime file sizes where available (unknown otherwise). Final summaries and result hashes
are immutable. Artifacts live under the ordinary private state root (directory 0700, SQLite
0600); only `--output` explicitly exports them. Treat exported state/results as potentially
sensitive even though the bundled corpus is synthetic.

Checkpoints are atomic chunks under a fenced one-hour lease. An interrupted process can
resume using saved inputs; completed chunks are never called again. An interrupted in-flight
chunk may run again and incur cost. A crashed process's lease expires before another process
can resume; concurrent runners cannot write stale results. `benchmark runs` discovers the
saved ID if interruption prevented a final receipt. Sealed partial runs preserve failures;
start a new run to retry them. Changing the runtime or benchmark version requires a new run.
No retry loop conceals availability problems.

`process-first` means the first actual call of each process session, including lazy local
model loading and manifest verification. It does not mean OS-page-cache cold; no cache flush,
service restart or external lifecycle action is performed. Warm means later calls in that
session, still without the Reflex cache. Resume can therefore add another process-first
sample. Different cases/batch sizes affect timings; use traces rather than treating the
aggregate as a controlled latency study. RSS is the lifetime process high-water mark, not
isolated model memory or GPU memory. Artifact bytes are serialized payload size, not allocated
SQLite pages. Rate waits and local storage contribute to wall throughput, not provider-call
time. An unsupported input/context limit is a visible failure, not a silently omitted case.

`compare` refuses mismatched corpora, packs, cases, modes or repeat counts and requires
completed runs. Environment equality is explicit; concurrency, calibration and settings
remain available in the exported metadata. No opaque aggregate, automatic ranking or
promotion is produced. Small authored corpora support diagnosis, not a production release
decision. Add independently reviewed workload examples and protected holdouts before making
statistical calibration or deployment claims.

For GLiNER availability and the exact capacity blocker, see [gliner.md](gliner.md).
The existing `scripts/benchmark.py` remains the framework overhead gate; it is not this
provider-fitness benchmark.

## OpenRouter/Jev

After independent snapshot verification, use
`reflex-axi benchmark run --provider provider.json --suite starter --mode batch --repeats 1`.
This is paid remote execution. The shipped `examples/openrouter-jev.json` refuses runtime
until revision verification; no live API benchmark was performed. See
[OpenRouter/Jev](openrouter-jev.md) for probability, revision and precision limitations.
Offline validation uses `--provider mock` and an injected OpenRouter byte-stream fixture
through all benchmark execution modes. Neither establishes model accuracy or live latency.
