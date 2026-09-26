"""Outcome adapter boundary: normalize observations, never execute tests or actions."""

from typing import Any, Protocol

from .models import Outcome


class OutcomeAdapter(Protocol):
    id: str
    version: str

    def adapt(self, source: dict[str, Any]) -> Outcome: ...


class RecordedOutcomeAdapter:
    """Strict common format for proofs/tests, downstream/delayed outcomes,
    corrections, overrides, metrics, retrospective judgments and comparisons.
    Consumer-specific adapters implement the same protocol.
    """

    id = "recorded-outcome"
    version = "1"

    def adapt(self, source: dict[str, Any]) -> Outcome:
        return Outcome.model_validate(source)
