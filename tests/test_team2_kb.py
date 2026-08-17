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
