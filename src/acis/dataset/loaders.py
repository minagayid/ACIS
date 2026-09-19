"""Join raw recording metadata with independently maintained annotations."""

from __future__ import annotations

from dataclasses import dataclass

from acis.schemas.annotation import AnnotationSegment
from acis.schemas.recording import Recording


@dataclass(frozen=True)
class TrainingExample:
    recording: Recording
    annotation: AnnotationSegment

    @property
    def animal_id(self) -> str:
        return self.recording.animal_id

    @property
    def session_id(self) -> str:
        return self.recording.session_id

    @property
    def site_id(self) -> str:
        return self.recording.site_id

    @property
    def species(self) -> str:
        return self.recording.species

    @property
    def label(self) -> str:
        if len(self.annotation.context_labels) != 1:
            raise ValueError(f"segment {self.annotation.segment_id} is multi-label")
        return self.annotation.context_labels[0]

    @property
    def is_single_label(self) -> bool:
        return len(self.annotation.context_labels) == 1

    @property
    def target_label(self) -> str | None:
        return self.annotation.context_labels[0] if self.is_single_label else None


def join_training_examples(recordings: list[Recording], annotations: list[AnnotationSegment]) -> list[TrainingExample]:
    by_id = {recording.recording_id: recording for recording in recordings}
    examples: list[TrainingExample] = []
    for annotation in annotations:
        recording = by_id.get(annotation.recording_id)
        if recording is None:
            raise ValueError(f"annotation {annotation.segment_id} references unknown recording")
        examples.append(TrainingExample(recording, annotation))
    return examples
