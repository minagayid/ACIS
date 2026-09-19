import unittest
import math

from acis.ingest.manifest import load_annotations, load_recordings, validate_dataset
from acis.schemas.annotation import AnnotationSegment
from acis.schemas.common import ValidationError
from acis.schemas.recording import Recording


class SchemaValidationTests(unittest.TestCase):
    def test_example_dataset_loads_and_joins(self):
        recordings, annotations = validate_dataset(
            "data/examples/recordings.jsonl", "data/examples/annotations.jsonl"
        )
        self.assertEqual(len(recordings), 16)
        self.assertEqual(len(annotations), 16)

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            AnnotationSegment.from_dict({
                "segment_id": "seg",
                "recording_id": "rec",
                "start_ms": 0,
                "end_ms": 100,
                "vocalization_type": "bark",
                "context_labels": ["alert_context"],
                "observed_behavior": [],
                "annotator_id": "a",
                "annotation_confidence": 0.5,
                "unexpected": True,
            })

    def test_multi_label_is_valid_but_unknown_label_is_not(self):
        base = {
            "segment_id": "seg",
            "recording_id": "rec",
            "start_ms": 0,
            "end_ms": 100,
            "vocalization_type": "bark",
            "context_labels": ["alert_context", "play_context"],
            "observed_behavior": [],
            "annotator_id": "a",
            "annotation_confidence": 0.5,
        }
        self.assertEqual(len(AnnotationSegment.from_dict(base).context_labels), 2)
        base["context_labels"] = ["happy"]
        with self.assertRaises(ValidationError):
            AnnotationSegment.from_dict(base)

    def test_nonfinite_values_and_naive_timestamps_are_rejected(self):
        base = {
            "segment_id": "seg",
            "recording_id": "rec",
            "start_ms": 0,
            "end_ms": 100,
            "vocalization_type": "bark",
            "context_labels": ["alert_context"],
            "observed_behavior": [],
            "annotator_id": "a",
            "annotation_confidence": math.nan,
        }
        with self.assertRaises(ValidationError):
            AnnotationSegment.from_dict(base)

    def test_recording_requires_timezone_and_explicit_consent(self):
        base = {
            "recording_id": "rec",
            "animal_id": "dog",
            "species": "canis_lupus_familiaris",
            "file_uri": "data/raw/a.wav",
            "captured_at": "2026-01-01T00:00:00",
            "session_id": "session",
            "site_id": "site",
            "device_id": "device",
            "location_class": "home",
        }
        with self.assertRaises(ValidationError):
            Recording.from_dict(base)
        base["captured_at"] = "2026-01-01T00:00:00Z"
        base["consent_status"] = "invented"
        with self.assertRaises(ValidationError):
            Recording.from_dict(base)

    def test_recording_accepts_non_sensitive_taxonomy_metadata(self):
        base = {
            "recording_id": "rec",
            "animal_id": "animal-1",
            "species": "tursiops_truncatus",
            "file_uri": "data/raw/a.wav",
            "captured_at": "2026-01-01T00:00:00Z",
            "session_id": "session",
            "site_id": "site",
            "device_id": "device",
            "location_class": "coastal",
            "consent_status": "documented",
            "population": "bay-a",
            "region": "north",
            "habitat": "nearshore",
            "wild_or_domestic": "wild",
            "social_group_id": "group-1",
        }
        recording = Recording.from_dict(base)
        self.assertEqual(recording.population, "bay-a")
        self.assertEqual(recording.social_group_id, "group-1")
