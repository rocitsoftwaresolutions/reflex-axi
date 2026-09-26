"""Optional offline GLiNER2 classifier. No model or dependency downloads at inference."""

import hashlib
import importlib
import importlib.metadata
import math
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from .errors import ReflexError
from .identity import canonical
from .models import Model, Prediction, ProviderSpec
from .providers import Request

RUNTIME_PACKAGES = (
    "gliner2",
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "numpy",
    "peft",
    "huggingface-hub",
)
MODEL_FILES = (
    "config.json",
    "encoder_config/config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


class GLiNERSettings(Model):
    model_path: str
    files: dict[str, str]
    runtime: dict[str, str]
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    max_len: int = Field(default=512, ge=32, le=4096)


def runtime_versions() -> dict[str, str]:
    try:
        return {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    except importlib.metadata.PackageNotFoundError:
        raise ReflexError(
            "provider_dependency",
            "Install the optional gliner extra after checking disk space; see docs/gliner.md",
        ) from None


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_spec(spec: ProviderSpec) -> GLiNERSettings:
    settings = GLiNERSettings.model_validate(spec.settings)
    cap = spec.capabilities
    if (
        not cap.local
        or not cap.native_batch
        or cap.multi_question
        or cap.concurrency != 1
        or cap.max_batch > 8
        or cap.cost_class != "free"
        or not cap.seeded
        or not cap.deterministic
        or settings.device != "cpu"
        or settings.dtype != "float32"
    ):
        raise ValueError(
            "gliner requires CPU float32, local native batches <=8, concurrency=1, seeded/deterministic=true, multi_question=false, cost_class=free"
        )
    if not Path(settings.model_path).is_absolute() or set(settings.files) != set(MODEL_FILES):
        raise ValueError(
            "gliner needs an absolute model_path and hashes for all documented model files"
        )
    if any(not re.fullmatch(r"[0-9a-f]{64}", x) for x in settings.files.values()):
        raise ValueError("model file hashes must be SHA-256 hex")
    if set(settings.runtime) != set(RUNTIME_PACKAGES) or settings.runtime["gliner2"] != "2.0.0":
        raise ValueError("pin all runtime versions; gliner2 must be 2.0.0")
    return settings


class GLiNERProvider:
    def __init__(self, spec: ProviderSpec) -> None:
        self.spec = spec
        self.settings = validate_spec(spec)
        self.classifier: Any = None
        self.schema_type: Any = None
        self.config_type: Any = None

    def _load(self) -> None:
        if self.classifier is not None:
            return
        s = self.settings
        if runtime_versions() != s.runtime:
            raise ReflexError(
                "provider_version",
                "installed runtime differs from pinned GLiNER runtime; create a new provider version",
            )
        root = Path(s.model_path)
        try:
            if any(file_hash(root / name) != expected for name, expected in s.files.items()):
                raise ReflexError(
                    "provider_version", "local model bytes differ from pinned manifest"
                )
        except OSError:
            raise ReflexError(
                "provider_model_missing",
                "complete local snapshot required; inference never downloads weights",
            ) from None
        torch = importlib.import_module("torch")
        torch.manual_seed(s.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        module = importlib.import_module("gliner2.classification.engine")
        self.schema_type = importlib.import_module(
            "gliner2.classification.schema"
        ).ClassificationSchema
        self.config_type = module.ClassificationConfig
        self.classifier = (
            module.Classifier.from_pretrained(
                s.model_path,
                local_files_only=True,
                device=s.device,
                dtype=s.dtype,
            )
            .to(device=s.device, dtype=s.dtype)
            .eval()
        )

    def infer(self, requests: list[Request]) -> dict[str, Prediction]:
        if not requests:
            return {}
        if (
            len(requests) > self.spec.capabilities.max_batch
            or len({canonical(r.pack.inference_definition()) for r in requests}) != 1
        ):
            raise ReflexError(
                "capability", "gliner batches require one judgment and at most max_batch requests"
            )
        try:
            self._load()
            pack = requests[0].pack
            # One independent head per inference identity. Combining heads changes logits,
            # so multi_question is deliberately false, even though the runtime supports it.
            schema = self.schema_type().single(
                "judgment", pack.judgment.labels, instruction=pack.judgment.question
            )
            texts = [canonical(r.state) for r in requests]
            # Conservative UTF-8 byte budget includes schema markers. Reject rather than
            # silently truncate state. This intentionally underuses the token window.
            schema_bytes = (
                len((pack.judgment.question + "".join(pack.judgment.labels)).encode())
                + 64 * len(pack.judgment.labels)
                + 128
            )
            if any(len(text.encode()) + schema_bytes > self.settings.max_len for text in texts):
                raise ReflexError(
                    "state_too_large",
                    "GLiNER conservative context budget exceeded; shorten state or version max_len",
                )
            rows = self.classifier.batch_score(
                texts,
                schema,
                config=self.config_type(
                    batch_size=self.spec.capabilities.max_batch, max_len=self.settings.max_len
                ),
            )
            if len(rows) != len(requests):
                raise ReflexError("provider_association", "gliner omitted or invented score rows")
            out = {}
            for request, text, row in zip(requests, texts, rows, strict=True):
                if row.text != text or request.id in out:
                    raise ReflexError(
                        "provider_association", "gliner score rows do not match input order"
                    )
                logits = dict(row.tasks["judgment"])
                if set(logits) != set(pack.judgment.labels) or any(
                    isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x)
                    for x in logits.values()
                ):
                    raise ReflexError(
                        "provider_distribution", "gliner must return finite logits for every label"
                    )
                top = max(logits.values())
                weights = {k: math.exp(v - top) for k, v in logits.items()}
                total = sum(weights.values())
                out[request.id] = Prediction(
                    distribution={k: v / total for k, v in weights.items()}, cost=0.0
                )
            return out
        except ReflexError:
            raise
        except ImportError:
            raise ReflexError(
                "provider_dependency",
                "GLiNER runtime unavailable; install the pinned optional extra",
            ) from None
        except MemoryError:
            raise ReflexError("provider_memory", "local model exceeded available memory") from None
        except Exception:
            # ML libraries include input and filesystem paths in arbitrary exception text.
            raise ReflexError(
                "provider_response",
                "GLiNER loading or scoring failed; verify runtime, snapshot and schema; no fallback",
            ) from None
