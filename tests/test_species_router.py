import unittest
from dataclasses import replace

from acis.dataset.loaders import join_training_examples
from acis.ingest.manifest import validate_dataset
from acis.models.inference import predict_with_evidence
from acis.models.species_router import SpeciesRoutedBaseline


class SpeciesRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recordings, annotations = validate_dataset(
            "data/examples/recordings.jsonl", "data/examples/annotations.jsonl"
        )
        dog_examples = join_training_examples(recordings, annotations)
        bird_examples = [
            replace(example, recording=replace(example.recording, species="taeniopygia_guttata"))
            for example in dog_examples
        ]
        cls.examples = dog_examples + bird_examples

    def test_each_species_gets_an_independent_model(self):
        model = SpeciesRoutedBaseline().fit(self.examples)
        self.assertEqual(set(model.models), {"canis_lupus_familiaris", "taeniopygia_guttata"})
        self.assertEqual(len(model.models["canis_lupus_familiaris"].means), 6)
        self.assertEqual(len(model.models["taeniopygia_guttata"].means), 6)

    def test_prediction_retrieval_is_species_scoped(self):
        model = SpeciesRoutedBaseline().fit(self.examples)
        example = self.examples[0]
        prediction = predict_with_evidence(
            model,
            example.annotation,
            species=example.species,
            neighbors=self.examples,
        )
        self.assertTrue(prediction.neighbors)
        dog_segment_ids = {item.annotation.segment_id for item in self.examples if item.species == example.species}
        self.assertTrue(all(item.segment_id in dog_segment_ids for item in prediction.neighbors))

    def test_router_round_trip(self):
        model = SpeciesRoutedBaseline().fit(self.examples)
        restored = SpeciesRoutedBaseline.from_dict(model.to_dict())
        original = model.predict_distribution(
            self.examples[0].annotation,
            species="canis_lupus_familiaris",
        )
        replay = restored.predict_distribution(
            self.examples[0].annotation,
            species="canis_lupus_familiaris",
        )
        self.assertEqual(original.label, replay.label)
        self.assertEqual(original.probabilities, replay.probabilities)

    def test_unknown_species_fails_closed(self):
        model = SpeciesRoutedBaseline().fit(self.examples)
        with self.assertRaises(ValueError):
            model.for_species("unknown_species")


if __name__ == "__main__":
    unittest.main()
