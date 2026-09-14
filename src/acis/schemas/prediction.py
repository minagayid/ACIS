"""Prediction contract: scored association, evidence, and abstention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedExample:
    segment_id: str
    recording_id: str
    label: str
    similarity: float
    animal_id: str | None = None
    device_id: str | None = None


@dataclass(frozen=True)
class Prediction:
    segment_id: str
    model_name: str
    model_version: str
    predicted_label: str | None
    predictive_confidence: float
    abstained: bool
    probabilities: dict[str, float]
    neighbors: list[RetrievedExample] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    ood_flags: list[str] = field(default_factory=list)
    confidence_kind: str = "uncalibrated_model_score"
    abstention_reason: str | None = None
    claim_scope: str = "context-associated hypothesis; not a translation or intent claim"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Animal/device IDs are useful for local diagnostics but should not be
        # emitted by a public-facing response by default.
        payload["neighbors"] = [
            {
                "segment_id": item["segment_id"],
                "recording_id": item["recording_id"],
                "label": item["label"],
                "similarity": item["similarity"],
            }
            for item in payload["neighbors"]
        ]
        return payload
