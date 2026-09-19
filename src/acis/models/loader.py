"""Model factory used by training artifacts and the local service."""

from __future__ import annotations

from typing import Any

from acis.models.baseline import ContextBaseline
from acis.models.species_router import SpeciesRoutedBaseline


MODEL_TYPES = {
    ContextBaseline.name: ContextBaseline,
    SpeciesRoutedBaseline.name: SpeciesRoutedBaseline,
}


def create_model(name: str, *, abstention_threshold: float, temperature: float = 1.0):
    try:
        model_type = MODEL_TYPES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported configured model: {name}") from exc
    return model_type(abstention_threshold=abstention_threshold, temperature=temperature)


def load_model_artifact(data: dict[str, Any]):
    if not isinstance(data, dict):
        raise ValueError("model artifact must be a JSON object")
    try:
        model_type = MODEL_TYPES[data.get("model_name")]
    except KeyError as exc:
        raise ValueError(f"unsupported model artifact: {data.get('model_name')}") from exc
    return model_type.from_dict(data)
