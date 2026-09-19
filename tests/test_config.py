import unittest

from acis.config import load_config


class ConfigTests(unittest.TestCase):
    def test_multispecies_scope_is_loaded(self):
        config = load_config("configs/multispecies_baseline.json")
        self.assertEqual(config.species, "multi_species")
        self.assertEqual(len(config.species_scope), 4)
        self.assertEqual(config.model, "species-routed-standardized-nearest-centroid")


if __name__ == "__main__":
    unittest.main()
