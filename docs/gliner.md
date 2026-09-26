# GLiNER2.5-Decide: optional local provider

Status: adapter tested against deterministic fixtures of the public GLiNER2 2.0.0 scoring
contract. **Real weights were not downloaded and real inference has not been validated.**
Do not interpret the fixture benchmark as model-quality evidence.

## Official sources inspected on 2026-09-26

- [Model card and snapshot](https://huggingface.co/fastino/GLiNER2.5-Decide/tree/7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6):
  English operational classifier, DeBERTa-v3-large encoder, advertised as 340M, Apache 2.0.
  Repository metadata actually counts **486,444,053 stored F32 parameters**. The difference
  matters for disk/RAM estimates; marketing parameter counts are not artifact sizes.
- [Runtime repository](https://github.com/fastino-ai/GLiNER2/tree/55656fbfa01d3d4a77485e1a1eeeaf682990ccdf),
  its Apache 2.0 license, loading code, classification schema/scorer and examples.
- [GLiNER2 2.0.0 distribution](https://pypi.org/project/gliner2/2.0.0/): Python >=3.10;
  local extra needs PyTorch, Transformers <5, NumPy, PEFT and safetensors. Reflex requires
  Python >=3.11. The 326,214-byte wheel expands to 1,258,899 bytes; its SHA-256 is
  `6f7c4cba0ef3173636d8bd9404aa94d4e0ccf36d4b4b9a0d27d740dd2d3236c3`.

The model card's examples say **potential outputs**; they are not measured results.
Its reported 60.2% accuracy on fast-decisions does not establish fitness or calibration for
Reflex starter packs. No upstream benchmark dataset was downloaded or used for training.

## Storage decision on this machine

The main APFS data volume had **2,770,388 KiB free (2.64 GiB), 99% capacity** at inspection.
Only public text/metadata, a 326 KB runtime wheel for source inspection, and remote ZIP
central-directory metadata were fetched. No weights, PyTorch wheel payload or ML dependency
installation occurred.

| Component | Bytes | Evidence / reservation |
| --- | ---: | --- |
| model.safetensors | 1,945,828,140 | exact snapshot listing |
| tokenizer.json | 8,333,952 | exact listing |
| other required JSON files | 7,284 | exact listing |
| PyTorch 2.14.0 macOS arm64 CPython 3.11 wheel | 127,277,244 | exact package metadata |
| PyTorch expanded | 523,772,122 | sum of remote ZIP central-directory uncompressed sizes |
| remaining dependencies + bytecode + installer cache | 1,073,741,824 | conservative reservation, not a measured install |
| possible snapshot/download staging duplicate | 1,954,169,376 | reserve one extra snapshot; cache strategy can reduce this |
| operating margin | 2,147,483,648 | minimum retained free space, excluding model RAM/swap growth |

Even one snapshot plus expanded PyTorch leaves only about **0.34 GiB** for everything else.
With the operating margin alone, the lower bound already fails. The conservative plan
needs about **7.3 GiB** of free disk; provision **at least 10 GiB free**, preferably more for
OS activity and model loading/swap. This is a capacity gate, not a claim the complete runtime
was installed or measured. Linux default PyTorch dependencies can include multi-GB CUDA
packages; select an appropriate CPU build and recalculate on that platform.

Do not delete or move unrelated user files. Arrange a larger volume or reclaim space under
separate authorization, then recheck both the runtime and cache volume immediately before
installation. Use macOS 14+ for the currently resolved arm64 PyTorch wheel. RAM must also
accommodate weights, load-time copies and activations; batch size one is the initial smoke
configuration, not a guarantee that a particular machine has sufficient RAM.

## Safe installation after capacity is available

Keep the existing core installation until these checks can pass. A default upgrade does
not install ML dependencies or rewrite provider/deployment configuration:

```sh
uv tool install --reinstall .
# Optional, only after disk/RAM checks, from a reviewed checkout:
uv tool install --reinstall '.[gliner]'
```

For a separate test environment, set `UV_CACHE_DIR` and the venv location on the adequately
provisioned volume. The same applies to the snapshot path. Never store either in this repo.
Use `uv.lock` with Python 3.11 to reproduce this resolution; `uv sync --frozen --extra gliner`
installs the optional runtime into the configured project environment. Core CI installs only
`--extra dev`. Global tool installs resolve dependencies separately, so capture the actual
runtime versions in the provider manifest below.

Download only the required files at the immutable revision (using the optional environment's
Python; this command deliberately downloads weights and must wait for the disk gate):

```python
from huggingface_hub import snapshot_download
from reflex_axi.gliner import MODEL_FILES
snapshot_download(
    "fastino/GLiNER2.5-Decide",
    revision="7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6",
    local_dir="/absolute/private/models/gliner-decide",
    allow_patterns=list(MODEL_FILES),
)
```

The official weight SHA-256 is
`40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997`.
Check it before creating the provider. Hash all required local files and capture installed
runtime versions rather than guessing them:

```python
import json
from pathlib import Path
from reflex_axi.gliner import MODEL_FILES, file_hash, runtime_versions
from reflex_axi.io import atomic_write
from reflex_axi.models import Capabilities, ProviderSpec
root = Path("/absolute/private/models/gliner-decide")
assert file_hash(root / "model.safetensors") == "40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997"
spec = ProviderSpec(
    adapter="gliner", id="gliner-decide-cpu", model="fastino/GLiNER2.5-Decide",
    version="7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6",
    capabilities=Capabilities(local=True, native_batch=True, multi_question=False,
        max_batch=1, concurrency=1, seeded=True, deterministic=True, cost_class="free"),
    settings={"model_path": str(root), "files": {f: file_hash(root / f) for f in MODEL_FILES},
        "runtime": runtime_versions(), "device": "cpu", "dtype": "float32",
        "seed": 0, "max_len": 1024},
)
atomic_write(Path("/absolute/private/gliner-provider.json"), json.dumps(spec.model_dump(), indent=2))
```

```sh
reflex-axi providers --import /absolute/private/gliner-provider.json
reflex-axi evaluate --pack filter-context --state examples/state.json \
  --provider /absolute/private/gliner-provider.json --local-only --research --no-cache --full
reflex-axi benchmark run --provider /absolute/private/gliner-provider.json \
  --suite starter --mode single --repeats 2 --concurrency 1
reflex-axi benchmark show --run <id> --full --json --output /absolute/private/gliner-run.json
```

Then increase `max_batch` to at most 8 in a **new provider version**, run all suites and modes,
and compare runs explicitly when appropriate. Calibration is separate, workload-specific,
and never inherited from another provider or settings version.

## Adapter contract and limits

`Classifier.batch_score` exposes full named per-label logits. Reflex applies stable softmax
across exactly the pack's labels, once, without renormalizing a truncated top-k list. The
basic `classify_text` winner/confidence API cannot supply this distribution. Binary labels
stay `false/true`; choice labels stay unchanged; score labels are exclusive categories and
Reflex computes their expected numeric value. These probabilities are **uncalibrated**.

The normalized state is canonical JSON. The judgment question becomes the schema instruction.
Each provider call contains one judgment and up to eight states. Although GLiNER supports
several heads per text, this adapter advertises `multi_question: false`: changing co-present
heads changes the model's prompt and could invalidate otherwise identical inference keys.
Bundles use the engine's deterministic per-pack grouping. Concurrency is one, CPU float32,
manual seed, deterministic PyTorch algorithms and one CPU thread. Exact cross-machine or
cross-version numerical equality is not promised; runtime version pins and the benchmark's
repeat/mode checks make those differences visible.

The adapter uses a conservative context budget: canonical state bytes plus question/label
bytes and schema-marker overhead must fit `max_len`. Oversized inputs fail `state_too_large`
instead of silently truncating. It deliberately underuses the token window. Schema marker
characters forbidden by GLiNER fail rather than being silently rewritten. Local model paths
and all relevant file/runtime hashes participate in provider identity. Loading is offline;
missing/changed files, missing dependencies, runtime changes, invalid logits, association
errors and memory/runtime failures are normalized without a provider fallback.

Measured model memory, latency, throughput, quality and operational behavior remain **unknown**
until a real run can be performed safely. Do not advertise this fixture-tested adapter as a
validated deployment.
