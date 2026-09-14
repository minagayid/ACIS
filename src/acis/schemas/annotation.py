"""Observable annotation contract, deliberately separate from predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

from .common import (
    ValidationError,
    copy_unknown_keys,
    optional_list,
    optional_string,
    require_int,
    require_list,
    require_number,
    require_string,
)


ALLOWED_CONTEXTS = frozenset({
    "play_context", "separation_context", "alert_context", "food_context", "resting_context", "unknown",
})


@dataclass(frozen=True)
class AnnotationSegment:
    segment_id: str
    recording_id: str
    start_ms: int
    end_ms: int
    vocalization_type: str
    context_labels: list[str]
    observed_behavior: list[str]
    annotator_id: str
    annotation_confidence: float
    acoustic_features: dict[str, float] = field(default_factory=dict)
    recorded_trigger: str | None = None
    receiver_id: str | None = None
    response_observed: list[str] = field(default_factory=list)
    response_latency_ms: int | None = None
    quality_flags: list[str] = field(default_factory=list)
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationSegment":
        allowed = {
            "segment_id", "recording_id", "start_ms", "end_ms", "vocalization_type", "context_labels",
            "observed_behavior", "annotator_id", "annotation_confidence", "acoustic_features",
            "recorded_trigger", "receiver_id", "response_observed", "response_latency_ms",
            "quality_flags", "notes",
        }
        copy_unknown_keys(data, allowed)
        start = require_int(data, "start_ms", minimum=0)
        end = require_int(data, "end_ms", minimum=1)
        if end <= start:
            raise ValidationError("end_ms must be greater than start_ms")
        labels = require_list(data, "context_labels")
        if not labels or any(label not in ALLOWED_CONTEXTS for label in labels):
            raise ValidationError(f"context_labels must use only {sorted(ALLOWED_CONTEXTS)}")
        features = data.get("acoustic_features", {})
        if not isinstance(features, dict) or any(
            not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for key, value in features.items()
        ):
            raise ValidationError("acoustic_features must be a mapping of string names to numbers")
        latency = require_int(data, "response_latency_ms", minimum=0) if data.get("response_latency_ms") is not None else None
        return cls(
            segment_id=require_string(data, "segment_id"),
            recording_id=require_string(data, "recording_id"),
            start_ms=start,
            end_ms=end,
            vocalization_type=require_string(data, "vocalization_type"),
            context_labels=labels,
            observed_behavior=require_list(data, "observed_behavior"),
            annotator_id=require_string(data, "annotator_id"),
            annotation_confidence=require_number(data, "annotation_confidence", minimum=0, maximum=1),
            acoustic_features={str(key): float(value) for key, value in features.items()},
            recorded_trigger=optional_string(data, "recorded_trigger"),
            receiver_id=optional_string(data, "receiver_id"),
            response_observed=optional_list(data, "response_observed"),
            response_latency_ms=latency,
            quality_flags=optional_list(data, "quality_flags"),
            notes=optional_string(data, "notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Annotation:
    recording_id: str
    annotator_id: str
    segments: list[AnnotationSegment]
    annotation_version: str = "1.0"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Annotation":
        allowed = {"recording_id", "annotator_id", "segments", "annotation_version"}
        copy_unknown_keys(data, allowed)
        raw_segments = data.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValidationError("segments must be a non-empty list")
        segments = [AnnotationSegment.from_dict(segment) for segment in raw_segments]
        recording_id = require_string(data, "recording_id")
        if any(segment.recording_id != recording_id for segment in segments):
            raise ValidationError("all segments must reference the annotation recording_id")
        return cls(
            recording_id=recording_id,
            annotator_id=require_string(data, "annotator_id"),
            segments=segments,
            annotation_version=require_string(data, "annotation_version") if data.get("annotation_version") is not None else "1.0",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
