"""Opt-in OpenRouter Decisions adapter. No SDK, discovery calls or model fallback."""

import math
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any, NoReturn

from pydantic import Field, ValidationError

from .errors import ReflexError
from .models import Model, Prediction, ProviderSpec
from .providers import Request, post, validate_predictions

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
LIMITATIONS = {
    "probabilities": "native; complete choice/score distributions required",
    "revision": "dated snapshot required; immutability and routability unverified live",
    "verification": "offline fixtures only; operator must verify revision before enabling runtime",
    "batching": "one state and one question per call; engine fans out bundles and batches",
    "score": "2-10 strictly increasing levels; exact probability/score consistency required",
    "retry": "no automatic retries; resume failed cells explicitly after rate-limit recovery",
}


class Settings(Model):
    # A dated identifier is evidence of a snapshot, not an immutability guarantee.
    # Require explicit operator attestation before any credential lookup or request.
    revision_verified: bool
    criteria: dict[str, dict[str, str]] = Field(default_factory=dict)


def validate_spec(spec: ProviderSpec) -> Settings:
    def invalid(message: str) -> NoReturn:
        raise ReflexError("provider_config", message, "Read docs/openrouter-jev.md") from None

    if spec.endpoint != ENDPOINT:
        invalid("openrouter-jev requires the documented HTTPS Decisions endpoint")
    if spec.adapter_version != "1":
        invalid("unsupported openrouter-jev adapter version")
    match = re.fullmatch(r"typesafe/jev-[0-9]+\.[0-9]+-([0-9]{8})", spec.model)
    if not match:
        invalid("configure an explicit dated typesafe/jev model snapshot, never a floating alias")
    try:
        datetime.strptime(match[1], "%Y%m%d")
    except ValueError:
        invalid("model snapshot must contain a valid calendar date")
    if spec.version != spec.model.rsplit("-", 1)[1]:
        invalid("provider version must equal the dated model revision")
    if not spec.api_key_env or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", spec.api_key_env):
        invalid("api_key_env must explicitly name a credential environment variable")
    cap = spec.capabilities
    if cap.local or cap.native_batch or cap.multi_question or cap.max_batch != 1:
        invalid("openrouter-jev is remote and supports one request per call")
    if cap.deterministic or cap.seeded:
        invalid("openrouter-jev has no verified deterministic or seeded execution contract")
    try:
        settings = Settings.model_validate(spec.settings)
    except ValidationError:
        invalid("settings require revision_verified (boolean) and optional per-pack criteria")
    for pack, criteria in settings.criteria.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,79}@[^\s@]{1,80}", pack) or not criteria:
            invalid("criteria must be keyed by versioned pack IDs with nonempty label mappings")
        if any(not k or not v.strip() or len(v) > 16000 for k, v in criteria.items()):
            invalid("criteria require nonempty labels and descriptions up to 16000 characters")
    return settings


def number(value: Any, *, maximum: float = math.inf) -> float:
    if type(value) not in (int, float):
        raise ValueError("number required")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= maximum:
        raise ValueError("number out of range")
    return result


def fields(value: Any, required: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or not value.keys() <= required | (optional or set())
    ):
        raise ValueError("invalid object fields")
    return value


class OpenRouterJevProvider:
    def __init__(self, spec: ProviderSpec, *, transport: Callable[..., Any] | None = None) -> None:
        validate_spec(spec)
        self.spec = spec
        self.transport = transport

    def infer(self, requests: list[Request]) -> dict[str, Prediction]:
        settings = validate_spec(self.spec)  # Also defend against mutated nested settings.
        if len(requests) != 1:
            raise ReflexError("capability", "openrouter-jev accepts one request per call")
        if not settings.revision_verified:
            raise ReflexError(
                "provider_version",
                "revision is unverified; confirm snapshot routability and immutability before enabling runtime",
                "Read docs/openrouter-jev.md; do not enable revision_verified based only on the example",
            )
        request = requests[0]
        judgment = request.pack.judgment
        labels = judgment.labels
        descriptions = settings.criteria.get(
            f"{request.pack.id}@{request.pack.version}", {k: k for k in labels}
        )
        if set(descriptions) != set(labels):
            raise ReflexError("provider_config", "criteria must cover exactly the pack labels")
        question: dict[str, Any] = {"instructions": judgment.question}
        if judgment.primitive == "binary":
            question.update(type="noul", criteria=descriptions)
        elif judgment.primitive == "choice":
            question.update(type="choice", criteria=descriptions)
        else:
            values = judgment.values or {}
            if len(labels) > 10 or any(
                values[a] >= values[b] for a, b in zip(labels, labels[1:], strict=False)
            ):
                raise ReflexError(
                    "capability",
                    "Jev score requires 2-10 levels in strictly increasing value order",
                )
            question.update(type="score", criteria=[descriptions[k] for k in labels])
        body = {
            "model": self.spec.model,
            "state": request.state,
            "questions": {"judgment": question},
            "provider": {
                "only": ["TypeSafe"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        raw = post(self.spec, body, transport=self.transport)
        try:
            prediction = self._parse(raw, request, descriptions)
            result = {request.id: prediction}
            validate_predictions(requests, result)
            return result
        except (ValueError, TypeError, KeyError, OverflowError):
            raise ReflexError(
                "provider_response",
                "invalid Decisions response; require complete normalized probabilities and consistent typed answers",
                "Read docs/openrouter-jev.md; no response body is logged",
            ) from None

    def _parse(self, raw: Any, request: Request, descriptions: dict[str, str]) -> Prediction:
        raw = fields(raw, {"model", "answers", "usage"}, {"id", "provider"})
        if raw["model"] != self.spec.model or raw.get("provider", "TypeSafe") != "TypeSafe":
            raise ReflexError(
                "provider_version",
                "response model/provider differs from the configured snapshot; no fallback",
            )
        answer = fields(raw["answers"], {"judgment"})["judgment"]
        judgment = request.pack.judgment
        labels = judgment.labels
        diagnostics: dict[str, str | float] = {
            "probability_source": "native",
            "live_verification": "unverified",
        }
        if "id" in raw:
            # A provider-controlled opaque ID could echo state or a secret. Never persist it.
            if not isinstance(raw["id"], str) or not raw["id"] or len(raw["id"]) > 256:
                raise ValueError("invalid request ID")
            diagnostics["request_id"] = "redacted"
        if judgment.primitive == "binary":
            answer = fields(answer, {"type", "noul"})
            if answer["type"] != "noul":
                raise ValueError("wrong primitive")
            probability = number(answer["noul"], maximum=1)
            distribution = {"false": 1 - probability, "true": probability}
        else:
            selected_field = "choice" if judgment.primitive == "choice" else "score"
            answer = fields(
                answer,
                {"type", selected_field, "probabilities"},
                {"confidence", "legend"} if selected_field == "score" else {"confidence"},
            )
            if answer["type"] != judgment.primitive:
                raise ValueError("wrong primitive")
            keys = labels if selected_field == "choice" else [str(i) for i in range(len(labels))]
            probabilities = fields(answer["probabilities"], set(keys))
            distribution = {
                label: number(probabilities[key], maximum=1)
                for label, key in zip(labels, keys, strict=True)
            }
            if abs(sum(distribution.values()) - 1) > 1e-6:
                raise ValueError("distribution not normalized")
            if "confidence" in answer:
                diagnostics["native_confidence"] = number(answer["confidence"], maximum=1)
            if selected_field == "choice":
                selected = answer["choice"]
                if (
                    not isinstance(selected, str)
                    or selected not in distribution
                    or distribution[selected] != max(distribution.values())
                ):
                    raise ValueError("choice contradicts probabilities")
                diagnostics["native_selected"] = selected
            else:
                score = number(answer["score"], maximum=len(labels) - 1)
                expected = sum(i * distribution[label] for i, label in enumerate(labels))
                if abs(score - expected) > 1e-6:
                    raise ValueError("score contradicts probabilities")
                if "legend" in answer and answer["legend"] != {
                    str(i): descriptions[label] for i, label in enumerate(labels)
                }:
                    raise ValueError("score legend differs from criteria order")
                diagnostics["native_score"] = score
        usage = fields(raw["usage"], {"input_tokens", "output_tokens"}, {"cost"})
        tokens = {}
        for key in ("input_tokens", "output_tokens"):
            if type(usage[key]) is not int:
                raise ValueError("token count must be an integer")
            tokens[key] = number(usage[key])
        cost = number(usage["cost"]) if "cost" in usage else None
        return Prediction(
            distribution=distribution, usage=tokens, cost=cost, diagnostics=diagnostics
        )
