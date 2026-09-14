"""JSONL ingestion with duplicate and cross-file integrity checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from acis.schemas.annotation import AnnotationSegment
from acis.schemas.common import ValidationError
from acis.schemas.recording import Recording


def _read_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, value


def load_recordings(path: str | Path) -> list[Recording]:
    recordings: list[Recording] = []
    seen: set[str] = set()
    for line_number, value in _read_jsonl(path):
        try:
            recording = Recording.from_dict(value)
        except ValidationError as exc:
            raise ValidationError(f"{path}:{line_number}: {exc}") from exc
        if recording.recording_id in seen:
            raise ValidationError(f"{path}:{line_number}: duplicate recording_id {recording.recording_id}")
        seen.add(recording.recording_id)
        recordings.append(recording)
    if not recordings:
        raise ValidationError(f"{path}: no recordings found")
    return recordings


def load_annotations(path: str | Path) -> list[AnnotationSegment]:
    annotations: list[AnnotationSegment] = []
    seen: set[str] = set()
    for line_number, value in _read_jsonl(path):
        try:
            segment = AnnotationSegment.from_dict(value)
        except ValidationError as exc:
            raise ValidationError(f"{path}:{line_number}: {exc}") from exc
        if segment.segment_id in seen:
            raise ValidationError(f"{path}:{line_number}: duplicate segment_id {segment.segment_id}")
        seen.add(segment.segment_id)
        annotations.append(segment)
    if not annotations:
        raise ValidationError(f"{path}: no annotations found")
    return annotations


def validate_dataset(recordings_path: str | Path, annotations_path: str | Path) -> tuple[list[Recording], list[AnnotationSegment]]:
    recordings = load_recordings(recordings_path)
    annotations = load_annotations(annotations_path)
    recording_by_id = {item.recording_id: item for item in recordings}
    for segment in annotations:
        if segment.recording_id not in recording_by_id:
            raise ValidationError(f"segment {segment.segment_id} references unknown recording {segment.recording_id}")
        recording = recording_by_id[segment.recording_id]
        if recording.duration_ms is not None and segment.end_ms > recording.duration_ms:
            raise ValidationError(
                f"segment {segment.segment_id} ends at {segment.end_ms}ms beyond recording {recording.recording_id} duration"
            )
    return recordings, annotations
