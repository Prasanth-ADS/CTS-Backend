import unittest

from app.llm.symptom_extraction import extract_observations


class SymptomExtractionTests(unittest.TestCase):
    def test_extracts_positive_and_unknown_observations(self):
        observations = extract_observations("Leaves have brown spots and water-soaked patches.")

        self.assertIs(observations["brown_patches"], True)
        self.assertIs(observations["wet_lesions"], True)
        self.assertIsNone(observations["yellow_halo"])

    def test_negative_mentions_override_positive_terms(self):
        observations = extract_observations("There are brown spots but no yellow halo or white mold.")

        self.assertIs(observations["brown_patches"], True)
        self.assertIs(observations["yellow_halo"], False)
        self.assertIs(observations["white_mold"], False)
