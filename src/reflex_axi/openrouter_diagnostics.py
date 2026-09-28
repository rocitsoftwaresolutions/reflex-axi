"""Bounded, opt-in score evidence. Never retain provider-controlled text or request data."""

import math
import uuid
from typing import Any

from .identity import canonical
from .store import Store

MESSAGES = {
    "accepted": "Score distribution accepted; native score is diagnostic only",
    "schema": "Unexpected or missing response object fields",
    "request_id": "Invalid opaque request ID shape",
    "primitive": "Typed answer does not match requested primitive",
    "probability_keys": "Probability keys must exactly cover the requested labels or indices",
    "probability_value": "Probabilities must be finite numbers in [0,1]",
    "probability_sum": "Probabilities must sum to one within 1e-6",
    "native_score": "Native score must be a finite number in [0,N-1]",
    "native_confidence": "Native confidence must be a finite number in [0,1]",
    "legend": "Score legend must match the ordered criteria exactly",
    "choice": "Native choice must name a maximum-probability label",
    "usage": "Usage requires nonnegative integer token counts and finite nonnegative cost",
    "provider_version": "Response model or provider differs from the configured snapshot",
    "provider_distribution": "Universal distribution validation failed",
    "provider_response": "Response exceeds the transport size limit",
    "provider_transport": "Transport or strict JSON decoding failed; no body retained",
    "provider_http": "HTTP request failed; no body or headers retained",
    "provider_credentials": "Configured credential is unavailable",
    "invalid_response": "Response failed structural validation",
    "diagnostic_limit": "Diagnostic record limit reached; subsequent records omitted",
}


class ResponseValidationError(ValueError):
    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(MESSAGES[category])


def finite(value: Any) -> int | float | None:
    if type(value) not in (int, float):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def score_snapshot(raw: Any, label_count: int, model: str) -> dict[str, Any]:
    """Project untrusted JSON onto a fixed numeric schema *before* answer validation.

    Index keys are locally generated, never copied. At most ten entries are retained;
    counts and completeness expose malformed/oversized maps without retaining their keys.
    """
    snapshot: dict[str, Any] = {"label_count": label_count}
    if not isinstance(raw, dict):
        return snapshot
    snapshot["model"] = model if raw.get("model") == model else "mismatch"
    snapshot["provider"] = (
        "absent"
        if "provider" not in raw
        else "TypeSafe"
        if raw["provider"] == "TypeSafe"
        else "mismatch"
    )
    answers = raw.get("answers")
    answer = answers.get("judgment") if isinstance(answers, dict) else None
    if isinstance(answer, dict):
        kind = answer.get("type")
        snapshot["answer_kind"] = kind if kind in ("score", "choice", "noul") else "invalid"
        snapshot["native_score"] = finite(answer.get("score"))
        snapshot["native_confidence"] = finite(answer.get("confidence"))
        legend = answer.get("legend")
        snapshot["legend_count"] = len(legend) if isinstance(legend, dict) else None
        probabilities = answer.get("probabilities")
        if isinstance(probabilities, dict):
            snapshot["probability_count"] = len(probabilities)
            numeric = {
                str(i): value
                for i in range(10)
                if str(i) in probabilities and (value := finite(probabilities[str(i)])) is not None
            }
            snapshot["probabilities"] = numeric
            snapshot["probabilities_complete"] = len(numeric) == len(probabilities)
            keys = {str(i) for i in range(label_count)}
            if set(numeric) == keys and len(probabilities) == label_count:
                snapshot["native_expected_index"] = finite(
                    sum(i * float(numeric[str(i)]) for i in range(label_count))
                )
    usage = raw.get("usage")
    if isinstance(usage, dict):
        snapshot["usage"] = {
            k: finite(usage.get(k)) for k in ("input_tokens", "output_tokens", "cost")
        }
    return snapshot


class ScoreDiagnostics:
    """Private state-root records, capped at 256 x 4096 bytes plus a limit marker.

    Construct explicitly for a diagnostic session or benchmark. No global logger, paths,
    state identifiers, provider settings, inference identity inputs, or raw bodies.
    Store supplies 0700/0600 permissions and atomic cross-process transactions.
    """

    MAX_RECORDS = 256
    MAX_BYTES = 4096

    def __init__(self, store: Store, *, session: str | None = None) -> None:
        self.store = store
        self.bucket = "openrouter_score_diagnostics:" + (session or uuid.uuid4().hex)

    def record(self, snapshot: dict[str, Any], category: str) -> None:
        category = category if category in MESSAGES else "invalid_response"
        row = {**snapshot, "category": category, "message": MESSAGES[category]}
        # Snapshot has a fixed schema. This second bound defends against future extensions.
        if len(canonical(row).encode()) > self.MAX_BYTES:
            row = {
                "category": category,
                "message": MESSAGES[category],
                "diagnostic_truncated": True,
            }
        with self.store.transaction() as db:
            count = db.execute(
                "SELECT count(*) FROM records WHERE bucket=?", (self.bucket,)
            ).fetchone()[0]
            if count > self.MAX_RECORDS:
                return
            if count == self.MAX_RECORDS:
                row = {"category": "diagnostic_limit", "message": MESSAGES["diagnostic_limit"]}
            self.store.put(self.bucket, f"{count:03d}", row, db, immutable=True)
