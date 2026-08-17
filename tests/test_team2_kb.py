import asyncio
import unittest
from unittest.mock import patch

from app.core.config import Settings
from app.integrations import team2_kb


class Team2KBTests(unittest.TestCase):
    def test_match_symptoms_uses_local_fallback_without_team2_url(self):
        settings = Settings(team2_kb_url=None)
        observations = {"wet_lesions": True, "white_mold": True, "yellow_halo": False}

        with patch.object(team2_kb, "get_settings", return_value=settings):
            matched = asyncio.run(team2_kb.match_symptoms(observations))

        self.assertEqual(matched, ["Potato___Late_blight", "Tomato___Leaf_Mold"])

    def test_match_symptoms_uses_http_client_when_team2_url_is_configured(self):
        settings = Settings(team2_kb_url="https://team2.example.test", team2_kb_timeout_seconds=1)

        with (
            patch.object(team2_kb, "get_settings", return_value=settings),
            patch.object(team2_kb, "_post_symptom_match", return_value=["Alternaria_Solani"]) as post_match,
        ):
            matched = asyncio.run(team2_kb.match_symptoms({"brown_patches": True}))

        self.assertEqual(matched, ["Alternaria_Solani"])
        post_match.assert_called_once_with("https://team2.example.test", {"brown_patches": True}, 1)


    def test_fetch_qa_knowledge_uses_local_fallback_without_team2_url(self):
        settings = Settings(team2_kb_url=None)

        with patch.object(team2_kb, "get_settings", return_value=settings):
            questions = asyncio.run(team2_kb.fetch_qa_knowledge(["Potato___Late_blight"]))

        self.assertEqual(questions[0]["question_id"], "q_water_soaked")
        self.assertEqual(set(questions[0]["support"]), {"Potato___Late_blight"})

    def test_fetch_qa_knowledge_uses_http_client_when_team2_url_is_configured(self):
        settings = Settings(team2_kb_url="https://team2.example.test", team2_kb_timeout_seconds=1)

        with (
            patch.object(team2_kb, "get_settings", return_value=settings),
            patch.object(team2_kb, "_get_qa_knowledge", return_value=[{"question_id": "q1"}]) as get_qa,
        ):
            questions = asyncio.run(team2_kb.fetch_qa_knowledge(["A", "B"]))

        self.assertEqual(questions, [{"question_id": "q1"}])
        get_qa.assert_called_once_with("https://team2.example.test", ["A", "B"], 1)
