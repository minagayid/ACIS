"""Evidence retrieval using the same feature representation as the baseline."""

from __future__ import annotations

import math

from acis.audio.features import acoustic_vector, apply_standardization
from acis.dataset.loaders import TrainingExample
from acis.schemas.annotation import AnnotationSegment
from acis.schemas.prediction import RetrievedExample


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def retrieve_neighbors(
    query: AnnotationSegment,
    examples: list[TrainingExample],
    *,
    top_k: int = 3,
    exclude_animal_id: str | None = None,
    exclude_recording_id: str | None = None,
    exclude_session_id: str | None = None,
    species: str | None = None,
    means: tuple[float, ...] = (),
    stds: tuple[float, ...] = (),
) -> list[RetrievedExample]:
    if top_k < 1:
        return []
    query_vector = apply_standardization(acoustic_vector(query), means, stds)
    ranked: list[tuple[float, TrainingExample]] = []
    for example in examples:
        if (
            (species and example.species != species)
            or
            (exclude_animal_id and example.animal_id == exclude_animal_id)
            or (exclude_recording_id and example.recording.recording_id == exclude_recording_id)
            or (exclude_session_id and example.session_id == exclude_session_id)
            or not example.is_single_label
            or example.target_label == "unknown"
        ):
            continue
        example_vector = apply_standardization(acoustic_vector(example.annotation), means, stds)
        distance = _distance(query_vector, example_vector)
        ranked.append((distance, example))
    ranked.sort(key=lambda item: (item[0], item[1].annotation.segment_id))
    return [
        RetrievedExample(
            segment_id=example.annotation.segment_id,
            recording_id=example.recording.recording_id,
            label=example.label,
            similarity=round(1 / (1 + distance), 6),
            animal_id=example.animal_id,
            device_id=example.recording.device_id,
        )
        for distance, example in ranked[:top_k]
    ]
