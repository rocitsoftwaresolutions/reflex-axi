# Implementation verification, 2026-09-26

Implemented the generated skill expansion, optional offline GLiNER adapter, and protected
provider benchmark CLI. Existing global Reflex 0.1.0 installation was confirmed with
`reflex-axi --version`; unrelated configuration and the global installation were left intact.
The replacement wheel was installed and exercised in an isolated environment by the gate.

## Validation

`uv run python scripts/check.py` passed: Ruff lint/format, mypy (21 modules), generated-skill
drift check, **113 tests**, wheel/sdist builds, isolated installed-wheel console smoke, and
the version/cache performance gate. The smoke includes benchmark discovery, execution and
resume. Tests include real loopback HTTP, native batching/concurrency, raw score fixtures,
interruption recovery, stale-lease fencing, immutable records, threshold boundaries and
proof that benchmark data cannot enter learning or mutate deployments.

A console reproduction also found that `--output` changed a partial run's exit status to
success. The fix preserves the original exit status before projection/export; its subprocess
regression test confirms that exporting a partial run still exits 1.

The full deterministic mock benchmark was recorded using:

```sh
uv run reflex-axi benchmark run --provider mock --concurrency 2 --full --json \
  --output .scratch/mock-benchmark.json --state-root .scratch/provider-bench-validation
```

These are ignored validation artifacts, not production state or committed benchmark runs.
Ordinary use defaults to the private runtime state root outside the checkout.

| Observation | Recorded value |
| --- | ---: |
| Scenarios / logical cells | 61 / 1,098 |
| Conformance / unexpected failures | 1.0 / 0 |
| Expected state-contract rejections | 36 |
| Primary labeled scenarios | 51 |
| Mock accuracy / Brier / ECE | 0.411765 / 0.996961 / 0.488235 |
| Actual provider calls / requests | 146 / 1,046 |
| Largest native batch | 64 |
| Measured execution wall time | 0.396 seconds |
| Process peak RSS | 48,218,112 bytes |

Run ID: `750eee24b81e40b0bf7f0e72ca4daf9c`.
Result hash: `fb2f1fd51a893c75f8af093ae68fc87ec759b7a4ecfda24779d98adaca3b2768`.
Timings are local observations, not performance promises. Mock predictions are deliberately
nonsemantic and cannot validate GLiNER or any production provider. Usage is unavailable
for the mock; reported API cost is zero.

## Remaining real-model work

The official model snapshot, source, inference interface, license and dependency metadata
were inspected. Weights were **not downloaded**. The initial 2.64 GiB free-space snapshot
cannot fit weights plus runtime/cache and operating margin. A later read-only disk check
showed roughly 4.06 GiB free, still below the conservative installation budget; no unrelated
files were removed or moved. Exact sizes, source revisions, capacity calculations and a
safe installation/smoke/benchmark sequence are in [gliner.md](gliner.md).

After capacity is provisioned, install the optional runtime, verify the pinned snapshot,
create/register a separate provider manifest, run real inference and all benchmark suites,
and inspect quality/calibration separately from conformance and speed. Real GLiNER fitness,
latency, memory consumption and deployment readiness remain unknown. No deployment was
activated, optimized, promoted or changed by this task.
