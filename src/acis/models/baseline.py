"""Transparent nearest-centroid baseline for context-associated predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from acis.audio.features import FEATURE_NAMES, acoustic_vector, apply_standardization, standardize
from acis.dataset.loaders import TrainingExample
from acis.schemas.annotation import AnnotationSegment


@dataclass(frozen=True)
class BaselineResult:
    label: str
    probabilities: dict[str, float]
    confidence: float
    min_distance: float


class ContextBaseline:
    """A readable baseline that makes it difficult to hide data leakage.

    It learns one centroid per operational context. Scaling statistics are fit
    on the training partition only and can be serialized with the model.
    """

    name = "standardized-nearest-centroid"
    version = "0.1.0"
    artifact_schema_version = "1.0"
    feature_pipeline_version = "acoustic-summary-v1"

    def __init__(self, *, abstention_threshold: float = 0.55, temperature: float = 1.0) -> None:
        if not math.isfinite(abstention_threshold) or not 0 < abstention_threshold <= 1:
            raise ValueError("abstention_threshold must be in (0, 1]")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive")
        self.abstention_threshold = abstention_threshold
        self.temperature = temperature
        self.means: tuple[float, ...] = ()
        self.stds: tuple[float, ...] = ()
        self.centroids: dict[str, tuple[float, ...]] = {}
        self.training_examples: list[TrainingExample] = []
        self.artifact_metadata: dict[str, Any] = {}

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.centroids))

    def fit(self, examples: list[TrainingExample], *, allow_excluded: bool = False) -> "ContextBaseline":
        single_label = [
            example for example in examples
            if example.is_single_label and example.target_label != "unknown"
        ]
        excluded = [example.annotation.segment_id for example in examples if not example.is_single_label or example.target_label == "unknown"]
        if excluded and not allow_excluded:
            raise ValueError(
                "baseline training excludes multi-label/unknown annotations; pass allow_excluded=True explicitly: "
                + ", ".join(excluded)
            )
        if len(single_label) < 2:
            raise ValueError("at least two eligible single-label examples are required")
        if any(example.recording.consent_status != "documented" for example in single_label):
            raise ValueError("training requires documented consent for every training recording")
        vectors = []
        incomplete = []
        for example in single_label:
            try:
                vectors.append(acoustic_vector(example.annotation))
            except ValueError:
                incomplete.append(example.annotation.segment_id)
        if incomplete:
            raise ValueError(f"training requires complete acoustic features; missing in {incomplete}")
        normalized, self.means, self.stds = standardize(vectors)
        grouped: dict[str, list[tuple[float, ...]]] = {}
        for example, vector in zip(single_label, normalized):
            grouped.setdefault(example.label, []).append(vector)
        if len(grouped) < 2:
            raise ValueError("at least two context labels are required")
        self.centroids = {
            label: tuple(sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0])))
            for label, rows in grouped.items()
        }
        self.training_examples = single_label
        return self

    def _distance(self, vector: tuple[float, ...], centroid: tuple[float, ...]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, centroid)))

    def predict_distribution(self, segment: AnnotationSegment) -> BaselineResult:
        if not self.centroids:
            raise RuntimeError("model is not fitted")
        vector = apply_standardization(acoustic_vector(segment), self.means, self.stds)
        distances = {label: self._distance(vector, centroid) for label, centroid in self.centroids.items()}
        scores = {label: math.exp(-distance / self.temperature) for label, distance in distances.items()}
        total = sum(scores.values()) or 1.0
        probabilities = {label: score / total for label, score in scores.items()}
        label = max(probabilities, key=probabilities.get)
        return BaselineResult(label, probabilities, probabilities[label], distances[label])

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "model_name": self.name,
            "model_version": self.version,
            "feature_pipeline_version": self.feature_pipeline_version,
            "abstention_threshold": self.abstention_threshold,
            "temperature": self.temperature,
            "feature_names": list(FEATURE_NAMES),
            "means": list(self.means),
            "stds": list(self.stds),
            "centroids": {label: list(vector) for label, vector in self.centroids.items()},
            "metadata": self.artifact_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextBaseline":
        if not isinstance(data, dict):
            raise ValueError("model artifact must be a JSON object")
        if data.get("artifact_schema_version") != cls.artifact_schema_version:
            raise ValueError("unsupported model artifact schema version")
        if data.get("model_name") != cls.name:
            raise ValueError("unsupported model artifact")
        if data.get("feature_pipeline_version") != cls.feature_pipeline_version:
            raise ValueError("unsupported feature pipeline version")
        if data.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("model feature_names do not match this package")
        model = cls(
            abstention_threshold=float(data.get("abstention_threshold", 0.55)),
            temperature=float(data.get("temperature", 1.0)),
        )
        try:
            model.means = tuple(float(value) for value in data["means"])
            model.stds = tuple(float(value) for value in data["stds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("model means/stds are invalid") from exc
        if len(model.means) != len(FEATURE_NAMES) or len(model.stds) != len(FEATURE_NAMES):
            raise ValueError("model means/stds have the wrong feature dimension")
        if any(not math.isfinite(value) for value in (*model.means, *model.stds)) or any(value <= 0 for value in model.stds):
            raise ValueError("model means/stds must be finite and stds must be positive")
        centroids = data.get("centroids")
        if not isinstance(centroids, dict):
            raise ValueError("model centroids are invalid")
        try:
            model.centroids = {
                str(label): tuple(float(value) for value in vector)
                for label, vector in centroids.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("model centroids are invalid") from exc
        if not model.centroids or "unknown" in model.centroids:
            raise ValueError("model centroids must contain eligible context labels only")
        if any(
            len(vector) != len(FEATURE_NAMES) or any(not math.isfinite(value) for value in vector)
            for vector in model.centroids.values()
        ):
            raise ValueError("model centroid dimensions or values are invalid")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("model metadata must be an object")
        if "training_segment_ids" in metadata and (
            not isinstance(metadata["training_segment_ids"], list)
            or not all(isinstance(item, str) for item in metadata["training_segment_ids"])
        ):
            raise ValueError("model training_segment_ids metadata is invalid")
        model.artifact_metadata = metadata
        return model
