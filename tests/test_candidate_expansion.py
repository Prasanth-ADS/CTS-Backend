import math
import unittest

from app.reasoning.candidates import expand_candidate_distribution


class CandidateExpansionTests(unittest.TestCase):
    def test_adds_new_kb_matches_and_renormalizes_distribution(self):
        expanded = expand_candidate_distribution(
            {"Potato___Late_blight": 0.6, "Tomato___Late_blight": 0.4},
            ["Potato___Late_blight", "Alternaria_Solani"],
        )

        self.assertEqual(set(expanded), {"Potato___Late_blight", "Tomato___Late_blight", "Alternaria_Solani"})
        self.assertTrue(math.isclose(sum(expanded.values()), 1.0))
        self.assertGreater(expanded["Potato___Late_blight"], expanded["Alternaria_Solani"])

    def test_keeps_distribution_unchanged_when_all_matches_already_exist(self):
        expanded = expand_candidate_distribution(
            {"Potato___Late_blight": 0.6, "Tomato___Late_blight": 0.4},
            ["Tomato___Late_blight"],
        )

        self.assertEqual(expanded, {"Potato___Late_blight": 0.6, "Tomato___Late_blight": 0.4})
