"""Strict, provider-neutral contracts. Each independent component has its own version."""

from typing import Any, Literal, Self

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ReflexError
from .identity import digest


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class StateBuilder(Model):
    version: str = Field(min_length=1)
    schema_: dict[str, Any] = Field(alias="schema")
    fields: list[str] | None = None

    @model_validator(mode="after")
    def valid_schema(self) -> Self:
        Draft202012Validator.check_schema(self.schema_)
        if self.schema_.get("type") != "object":
            raise ValueError("state schema must describe an object")
        if self.schema_.get("additionalProperties") is not False:
            raise ValueError("state schema must reject additionalProperties")

        # No network or filesystem resolution while validating untrusted state.
        def check(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in {"$ref", "$dynamicRef"} and not str(value).startswith("#"):
                        raise ValueError("external schema references are forbidden")
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)

        check(self.schema_)
        if self.fields is not None and len(set(self.fields)) != len(self.fields):
            raise ValueError("duplicate StateBuilder fields")
        return self

    def build(self, state: dict[str, Any]) -> dict[str, Any]:
        # Validate before projection: misspelled input must never disappear silently.
        errors = sorted(
            Draft202012Validator(self.schema_).iter_errors(state), key=lambda e: str(e.path)
        )
        if errors:
            raise ReflexError(
                "state_invalid",
                f"state violates contract at {list(errors[0].path)}",
                "reflex-axi packs --full",
            )
        result = state if self.fields is None else {k: state[k] for k in self.fields if k in state}
        digest(result)  # Reject non-finite values, even in open nested objects.
        return result


class Judgment(Model):
    version: str = Field(min_length=1)
    primitive: Literal["binary", "choice", "score"]
    question: str = Field(min_length=1, max_length=16000)
    labels: list[str] = Field(min_length=2, max_length=100)
    values: dict[str, float] | None = None

    @model_validator(mode="after")
    def valid_labels(self) -> Self:
        if len(set(self.labels)) != len(self.labels) or any(not x for x in self.labels):
            raise ValueError("labels must be distinct nonempty strings")
        if self.primitive == "binary" and self.labels != ["false", "true"]:
            raise ValueError('binary labels must be ["false", "true"]')
        if self.primitive == "score" and (not self.values or set(self.values) != set(self.labels)):
            raise ValueError("score requires a numeric value for every label")
        if self.values is not None and set(self.values) != set(self.labels):
            raise ValueError("values must cover exactly the labels")
        return self


class Policy(Model):
    version: str = "1"
    kind: Literal["none", "recommend"] = "none"
    min_confidence: float = Field(default=0.8, ge=0, le=1)
    actions: dict[str, str] = Field(default_factory=dict)
    uncertain: str = "escalate"


class EvaluationSpec(Model):
    version: str = "1"
    min_replay: int = Field(default=3, ge=1)
    min_shadow: int = Field(default=3, ge=1)
    max_regression: float = Field(default=0.0, ge=0, le=1)
    min_utility: float = Field(default=0.0, ge=0, le=1)
    utility: Literal["accuracy", "squared_error"] = "accuracy"


class Pack(Model):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,79}$")
    version: str = Field(min_length=1, max_length=80)
    state: StateBuilder
    judgment: Judgment
    policy: Policy = Field(default_factory=Policy)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_policy(self) -> Self:
        if not set(self.policy.actions) <= set(self.judgment.labels):
            raise ValueError("policy has unknown labels")
        return self

    def inference_definition(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "state": self.state.model_dump(by_alias=True),
            "judgment": self.judgment.model_dump(),
        }


class Capabilities(Model):
    primitives: list[Literal["binary", "choice", "score"]] = ["binary", "choice", "score"]
    native_batch: bool = False
    multi_question: bool = False
    local: bool = False
    max_batch: int = Field(default=1, ge=1, le=10000)
    concurrency: int = Field(default=4, ge=1, le=256)
    requests_per_second: float = Field(default=10.0, gt=0)
    max_state_bytes: int = Field(default=100000, ge=1)
    latency_class: Literal["low", "medium", "high"] = "medium"
    cost_class: Literal["free", "low", "medium", "high"] = "low"
    seeded: bool = False
    deterministic: bool = False


class ProviderSpec(Model):
    adapter: Literal["mock", "jev", "structured-llm", "local-student", "gliner", "openrouter-jev"]
    id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    version: str = Field(min_length=1)
    adapter_version: str = "1"
    endpoint: str | None = None
    api_key_env: str | None = None
    timeout: float = Field(default=30.0, gt=0, le=300)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compatible(self) -> Self:
        if self.adapter in {"jev", "structured-llm"} and not self.endpoint:
            raise ValueError("remote adapters require an explicit endpoint")
        if self.adapter == "structured-llm" and (
            self.capabilities.native_batch or self.capabilities.multi_question
        ):
            raise ValueError("structured-llm adapter supports single-request execution only")
        if self.adapter in {"mock", "local-student"} and not self.capabilities.local:
            raise ValueError("mock/student capabilities must declare local=true")
        if self.adapter == "openrouter-jev":
            from .openrouter import validate_spec as validate_openrouter_spec

            validate_openrouter_spec(self)
        if self.adapter == "gliner":
            from .gliner import validate_spec

            validate_spec(self)
        return self

    def identity(self) -> str:
        # Credentials affect authorization, never inference. Everything else is pinned.
        return digest(self.model_dump(exclude={"api_key_env"}))


class Calibration(Model):
    id: str
    version: str
    provider_hash: str
    pack_id: str
    pack_version: str
    workload: str
    scope: str
    temperature: float = Field(default=1.0, gt=0, le=100)


class Binding(Model):
    provider: ProviderSpec
    calibration: Calibration | None = None

    def check(self, pack: Pack, workload: str, scope: str) -> None:
        c = self.calibration
        if c and (
            c.provider_hash != self.provider.identity()
            or c.pack_id != pack.id
            or c.pack_version != pack.version
            or c.workload != workload
            or c.scope != scope
        ):
            raise ReflexError(
                "calibration_mismatch",
                "calibration does not match provider, pack, workload and evaluation scope",
                "reflex-axi compare --help",
            )


class ExecutionPolicy(Model):
    local_only: bool = False
    max_cost_class: Literal["free", "low", "medium", "high"] = "high"
    max_latency_class: Literal["low", "medium", "high"] = "high"
    min_batch: int = Field(default=1, ge=1)
    cache: bool = True
    workload: str = "default"
    scope: str = "default"


class Prediction(Model):
    distribution: dict[str, float]
    usage: dict[str, float] = Field(default_factory=dict)
    cost: float | None = Field(default=None, ge=0)
    diagnostics: dict[str, str | float] = Field(default_factory=dict)


class DecisionResult(Model):
    id: str
    pack: str
    version: str
    primitive: str
    raw_distribution: dict[str, float]
    distribution: dict[str, float]
    selected: str
    expected_score: float | None
    confidence: float
    uncertainty: float
    provider: dict[str, str]
    calibration: dict[str, str] | None
    state_hash: str
    inference_hash: str
    latency_ms: float
    cost: float | None
    usage: dict[str, float]
    execution: dict[str, Any]
    policy_result: dict[str, Any] | None = None


class Scope(Model):
    name: str = Field(min_length=1)
    readable: bool = True
    evaluatable: bool = False
    optimizable: bool = False
    allowed_to_mutate: bool = False


class Outcome(Model):
    id: str = Field(min_length=1)
    decision_id: str
    scope: str
    kind: Literal[
        "proof",
        "test",
        "downstream",
        "correction",
        "override",
        "metric",
        "delayed",
        "retrospective",
        "comparison",
    ]
    expected: str | None = None
    value: float | None = None
    success: bool | None = None
    gap: str = "general"
    protected: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def has_signal(self) -> Self:
        if self.expected is None and self.value is None and self.success is None:
            raise ValueError("outcome requires expected, value or success")
        return self


class Deployment(Model):
    pack: Pack
    binding: Binding
    workload: str = "default"
    scope: str = "default"

    @model_validator(mode="after")
    def check_calibration(self) -> Self:
        self.binding.check(self.pack, self.workload, self.scope)
        return self


class Bundle(Model):
    id: str
    version: str
    packs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self) -> Self:
        if len(set(self.packs)) != len(self.packs) or any("@" not in p for p in self.packs):
            raise ValueError("bundle packs must be unique versioned id@version references")
        return self
