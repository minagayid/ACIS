"""Convert a baseline output into the public evidence-aware prediction contract."""

from __future__ import annotations

from acis.dataset.loaders import TrainingExample
from acis.models.baseline import ContextBaseline
from acis.retrieval.nearest_examples import retrieve_neighbors
from acis.schemas.annotation import AnnotationSegment
from acis.schemas.prediction import Prediction


def _selected_model(model, species: str | None):
    """Return the species-specific model while keeping the legacy API intact."""
    selector = getattr(model, "for_species", None)
    return selector(species) if selector is not None else model


def predict_with_evidence(
    model: ContextBaseline,
    segment: AnnotationSegment,
    *,
    segment_id: str | None = None,
    query_recording_animal_id: str | None = None,
    query_recording_device_id: str | None = None,
    query_recording_id: str | None = None,
    query_session_id: str | None = None,
    species: str | None = None,
    neighbors: list[TrainingExample] | None = None,
    training_segment_ids: set[str] | None = None,
    top_k: int = 3,
) -> Prediction:
    selected = _selected_model(model, species)
    result = selected.predict_distribution(segment)
    ood_flags: list[str] = []
    if result.min_distance > 4.0:
        ood_flags.append("far_from_training_centroids")
    abstained = result.confidence < selected.abstention_threshold or bool(ood_flags)
    predicted_label = None if abstained else result.label
    abstention_reason = None
    if abstained:
        abstention_reason = "out_of_distribution_heuristic" if ood_flags else "low_confidence"
    evidence = {
        "same_animal_examples": 0,
        "same_device_examples": 0,
        "training_examples": len(selected.training_examples) or len(selected.artifact_metadata.get("training_segment_ids", [])),
        "min_centroid_distance": round(result.min_distance, 4),
    }
    retrieved = []
    if neighbors is not None:
        retrieved = retrieve_neighbors(
            segment,
            neighbors,
            top_k=top_k,
            exclude_animal_id=query_recording_animal_id,
            exclude_recording_id=query_recording_id,
            exclude_session_id=query_session_id,
            species=species,
            means=selected.means,
            stds=selected.stds,
        )
        evidence["same_animal_examples"] = sum(
            1 for item in retrieved if query_recording_animal_id and item.animal_id == query_recording_animal_id
        )
        evidence["same_device_examples"] = sum(
            1 for item in retrieved if query_recording_device_id and item.device_id == query_recording_device_id
        )
        known_training_ids = training_segment_ids if training_segment_ids is not None else {
            item.annotation.segment_id for item in selected.training_examples
        }
        evidence["query_in_training_set"] = segment.segment_id in known_training_ids
        evidence["same_animal_retrieval_excluded"] = query_recording_animal_id is not None
    return Prediction(
        segment_id=segment_id or segment.segment_id,
        model_name=model.name,
        model_version=model.version,
        predicted_label=predicted_label,
        predictive_confidence=round(result.confidence, 6),
        abstained=abstained,
        probabilities={label: round(probability, 6) for label, probability in result.probabilities.items()},
        neighbors=retrieved,
        evidence=evidence,
        ood_flags=ood_flags,
        abstention_reason=abstention_reason,
    )
