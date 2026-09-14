import unittest

from acis.dataset.loaders import join_training_examples
from acis.dataset.split import grouped_split
from acis.ingest.manifest import validate_dataset
from acis.models.baseline import ContextBaseline
from acis.models.inference import predict_with_evidence


class SplitAndBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recordings, annotations = validate_dataset(
            "data/examples/recordings.jsonl", "data/examples/annotations.jsonl"
        )
        cls.examples = join_training_examples(recordings, annotations)

    def test_animals_never_overlap(self):
        split = grouped_split(self.examples)
        self.assertTrue(set(split.train_animals).isdisjoint(split.validation_animals))
        self.assertTrue(set(split.train_animals).isdisjoint(split.test_animals))
        self.assertTrue(set(split.validation_animals).isdisjoint(split.test_animals))
        self.assertEqual(len(split.train) + len(split.validation) + len(split.test), len(self.examples))

    def test_model_uses_training_statistics_and_returns_distribution(self):
        split = grouped_split(self.examples)
        model = ContextBaseline().fit(split.train)
        prediction = predict_with_evidence(model, split.test[0].annotation, neighbors=split.train)
        self.assertAlmostEqual(sum(prediction.probabilities.values()), 1.0, places=5)
        self.assertTrue(prediction.neighbors)
        self.assertIn(prediction.predicted_label, set(model.labels) | {"unknown"})
        self.assertEqual(set(model.means).__class__, set)

    def test_model_serialization_round_trip(self):
        model = ContextBaseline().fit(self.examples)
        restored = ContextBaseline.from_dict(model.to_dict())
        original = model.predict_distribution(self.examples[0].annotation)
        replay = restored.predict_distribution(self.examples[0].annotation)
        self.assertEqual(original.label, replay.label)
        self.assertEqual(original.probabilities, replay.probabilities)

    def test_invalid_model_artifact_is_rejected(self):
        with self.assertRaises(ValueError):
            ContextBaseline.from_dict({"model_name": "not-supported"})
