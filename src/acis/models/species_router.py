"""Species-routed baselines for extending ACIS without forcing one animal language."""

from __future__ import annotations

from typing import Any

from acis.dataset.loaders import TrainingExample
from acis.models.baseline import ContextBaseline
from acis.schemas.annotation import AnnotationSegment


class SpeciesRoutedBaseline:
    """Keep preprocessing and centroids independent for every species.

    Acoustic scales and context cues are not assumed to transfer between
    species. The router therefore shares only the operational label vocabulary
    and API contract; each species gets its own leakage-aware baseline.
    """

    name = "species-routed-standardized-nearest-centroid"
    version = "0.2.0"
    artifact_schema_version = "1.0"

    def __init__(self, *, abstention_threshold: float = 0.55, temperature: float = 1.0) -> None:
        self.abstention_threshold = abstention_threshold
        self.temperature = temperature
        self.models: dict[str, ContextBaseline] = {}
        self.artifact_metadata: dict[str, Any] = {}

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({label for model in self.models.values() for label in model.labels}))

    @property
    def training_examples(self) -> list[TrainingExample]:
        return [example for species in sorted(self.models) for example in self.models[species].training_examples]

    def fit(self, examples: list[TrainingExample], *, allow_excluded: bool = False) -> "SpeciesRoutedBaseline":
        grouped: dict[str, list[TrainingExample]] = {}
        for example in examples:
            grouped.setdefault(example.species, []).append(example)
        if len(grouped) < 2:
            raise ValueError("species-routed training requires at least two species")

        models: dict[str, ContextBaseline] = {}
        for species, species_examples in sorted(grouped.items()):
            model = ContextBaseline(
                abstention_threshold=self.abstention_threshold,
                temperature=self.temperature,
            )
            try:
                model.fit(species_examples, allow_excluded=allow_excluded)
            except ValueError as exc:
                raise ValueError(f"cannot train species {species}: {exc}") from exc
            models[species] = model
        self.models = models
        return self

    def for_species(self, species: str | None) -> ContextBaseline:
        if not species:
            raise ValueError("species is required for a species-routed model")
        try:
            return self.models[species]
        except KeyError as exc:
            raise ValueError(f"no model is trained for species {species}") from exc

    def predict_distribution(self, segment: AnnotationSegment, *, species: str | None = None):
        return self.for_species(species).predict_distribution(segment)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "model_name": self.name,
            "model_version": self.version,
            "abstention_threshold": self.abstention_threshold,
            "temperature": self.temperature,
            "species_models": {species: model.to_dict() for species, model in sorted(self.models.items())},
            "metadata": self.artifact_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeciesRoutedBaseline":
        if not isinstance(data, dict):
            raise ValueError("model artifact must be a JSON object")
        if data.get("artifact_schema_version") != cls.artifact_schema_version:
            raise ValueError("unsupported model artifact schema version")
        if data.get("model_name") != cls.name:
            raise ValueError("unsupported species-routed model artifact")
        raw_models = data.get("species_models")
        if not isinstance(raw_models, dict) or not raw_models:
            raise ValueError("species_models must be a non-empty object")
        model = cls(
            abstention_threshold=float(data.get("abstention_threshold", 0.55)),
            temperature=float(data.get("temperature", 1.0)),
        )
        model.models = {species: ContextBaseline.from_dict(artifact) for species, artifact in raw_models.items()}
        if any(not isinstance(species, str) or not species.strip() for species in model.models):
            raise ValueError("species model keys must be non-empty strings")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("model metadata must be an object")
        model.artifact_metadata = metadata
        return model
